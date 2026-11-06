"""Official CNN-DP and the specified finite-window Mamba extension."""
import torch
from torch import nn
from torch.nn import functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.vision.model_getter import get_resnet


def build_policy(variant, shape_meta, cfg):
    encoder = MultiImageObsEncoder(
        shape_meta, get_resnet('resnet18', weights=None), crop_shape=(76,76),
        random_crop=True, use_group_norm=True, share_rgb_model=False, imagenet_norm=True)
    cls = DiffusionUnetImagePolicy if variant == 'M0' else MambaPastPolicy
    return cls(shape_meta=shape_meta, obs_encoder=encoder,
        noise_scheduler=DDPMScheduler(**cfg['ddpm']), horizon=cfg['horizon'],
        n_action_steps=cfg['n_action_steps'], n_obs_steps=2, num_inference_steps=100,
        obs_as_global_cond=True, diffusion_step_embed_dim=128,
        down_dims=(512,1024,2048), kernel_size=5, n_groups=8, cond_predict_scale=True)


class MambaPastPolicy(DiffusionUnetImagePolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from mamba_ssm import Mamba
        self.history_projection = nn.Linear(self.obs_feature_dim, 256)
        self.history_mamba = Mamba(d_model=256, d_state=16, d_conv=4, expand=2)
        self.history_norm = nn.LayerNorm(256)
        self.past_head = nn.Sequential(nn.Linear(256,256), nn.GELU(),
                                       nn.Linear(256,8*self.action_dim))
        self.model = ConditionalUnet1D(input_dim=self.action_dim, local_cond_dim=None,
            global_cond_dim=2*self.obs_feature_dim+256, diffusion_step_embed_dim=128,
            down_dims=(512,1024,2048), kernel_size=5, n_groups=8, cond_predict_scale=True)

    def encode_history(self, obs):
        assert all(v.shape[1] == 16 for v in obs.values())
        nobs = self.normalizer.normalize(obs)
        b = next(iter(nobs.values())).shape[0]
        # One shared encoder pass. The last two features are exactly the short window.
        z = self.obs_encoder({k:v.flatten(0,1) for k,v in nobs.items()}).reshape(b,16,-1)
        h = self.history_norm(self.history_mamba(self.history_projection(z))[:,-1])
        return torch.cat((z[:,-2:].flatten(1),h),dim=-1), h

    def past_loss(self, h, past_action, past_mask):
        label = self.normalizer['action'].normalize(past_action)
        pred = self.past_head(h).reshape(-1,8,self.action_dim)
        mask = past_mask.unsqueeze(-1).to(pred.dtype)
        return ((pred-label).square()*mask).sum()/(mask.sum()*self.action_dim).clamp_min(1)

    def compute_loss(self, batch):
        cond,h = self.encode_history(batch['obs'])
        actions = self.normalizer['action'].normalize(batch['action'])
        noise = torch.randn_like(actions)
        t = torch.randint(0,self.noise_scheduler.config.num_train_timesteps,
                          (len(actions),),device=actions.device).long()
        noisy = self.noise_scheduler.add_noise(actions,noise,t)
        loss_ddpm = F.mse_loss(self.model(noisy,t,global_cond=cond),noise)
        loss_past = self.past_loss(h,batch['past_action'],batch['past_mask'])
        self.loss_parts = (loss_ddpm.detach(),loss_past.detach())
        return loss_ddpm+0.1*loss_past

    @torch.no_grad()
    def predict_action(self, obs_dict):
        cond,_ = self.encode_history(obs_dict)
        data = torch.zeros((len(cond),self.horizon,self.action_dim),device=self.device,dtype=self.dtype)
        sample = self.conditional_sample(data,torch.zeros_like(data,dtype=torch.bool),global_cond=cond)
        action = self.normalizer['action'].unnormalize(sample)
        # Keep n_obs_steps=2 alignment even though the history contains 16 frames.
        return {'action':action[:,1:1+self.n_action_steps], 'action_pred':action}
