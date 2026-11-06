"""Stanford CNN-DP and continuous Mamba policies for RMBench.

Mamba dimensions and the past-action auxiliary head follow the 3.2 Coffee
implementation at github.com/Lin-zh-handsome/memory, d4e1749b004f.
"""
from dataclasses import dataclass, field, replace
from math import log1p, prod

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


MEMORY_CAPACITY = 8
HISTORY_DIM = 256
PAST_STEPS = 8


def build_policy(variant, shape_meta, cfg):
    vision = cfg.get("vision", {})
    crop = vision.get("crop", (76, 76))
    encoder = MultiImageObsEncoder(
        shape_meta, get_resnet("resnet18", weights=None),
        crop_shape=None if crop is None else tuple(crop),
        random_crop=vision.get("random_crop", True),
        use_group_norm=vision.get("group_norm", True),
        share_rgb_model=vision.get("share_rgb_model", False),
        imagenet_norm=vision.get("imagenet_norm", True),
    )
    unet = {
        "down_dims": (512, 1024, 2048),
        "diffusion_step_embed_dim": 128,
        "kernel_size": 5,
        "n_groups": 8,
        "cond_predict_scale": True,
    }
    unet.update(cfg.get("unet", {}))
    common = dict(
        shape_meta=shape_meta, obs_encoder=encoder, noise_scheduler=DDPMScheduler(**cfg["ddpm"]),
        horizon=cfg["horizon"], n_action_steps=cfg["n_action_steps"], n_obs_steps=2,
        num_inference_steps=cfg.get("num_inference_steps", 100), obs_as_global_cond=True,
    )
    if variant == "M0":
        return DiffusionUnetImagePolicy(**common, **unet)
    if variant in ("M2", "M3"):
        return ContinuousMemoryPolicy(cfg, unet, use_events=variant == "M3", **common)
    raise ValueError(f"Unknown model {variant}")


def mamba_recurrence(core, tokens, cache=None):
    """Run Mamba1 with functional, differentiable convolution and SSM caches."""
    if cache is None:
        cache = core.allocate_inference_cache(tokens.shape[0], max_seqlen=1)
    conv_state, ssm_state = cache
    x, gate = core.in_proj(tokens).chunk(2, dim=-1)
    conv_input = torch.cat((conv_state[..., 1:], x.transpose(1, 2)), dim=-1)
    x = F.conv1d(conv_input, core.conv1d.weight, core.conv1d.bias, groups=core.d_inner)
    x = F.silu(x.transpose(1, 2))
    dt_low_rank, b, c = core.x_proj(x).split((core.dt_rank, core.d_state, core.d_state), dim=-1)
    dt_raw = F.linear(dt_low_rank, core.dt_proj.weight)
    a = -core.A_log.float().exp()
    dt = F.softplus(dt_raw.float() + core.dt_proj.bias.float())

    if tokens.is_cuda and tokens.shape[1] > 1:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        def channels_first(value):
            return value.transpose(1, 2).contiguous()

        # The stock CUDA scan starts from zero. Add the homogeneous solution
        # induced by the incoming state, using the same discretization.
        zero_start = selective_scan_fn(
            channels_first(x), channels_first(dt_raw), a, channels_first(b),
            channels_first(c), core.D.float(), z=channels_first(gate),
            delta_bias=core.dt_proj.bias.float(), delta_softplus=True,
        ).transpose(1, 2)
        elapsed = dt.cumsum(dim=1)
        initial_contribution = torch.exp(elapsed[..., None] * a) * ssm_state[:, None]
        correction = torch.einsum("btdn,btn->btd", initial_contribution, c.float())
        y = zero_start + correction.to(x.dtype) * F.silu(gate)
        # selective_scan_fn does not backpropagate its returned final state.
        # Compute that state explicitly so this API can carry gradients too.
        remaining = elapsed[:, -1:] - elapsed
        decay = torch.exp(remaining[..., None] * a)
        driven = torch.einsum("btdn,btd,btn->bdn", decay, dt * x.float(), b.float())
        ssm_state = initial_contribution[:, -1] + driven
    else:
        output = []
        for x_t, gate_t, dt_t, b_t, c_t in zip(
            x.unbind(1), gate.unbind(1), dt.unbind(1), b.unbind(1), c.unbind(1)
        ):
            d_a = torch.exp(dt_t[..., None] * a)
            driven = (dt_t * x_t.float())[..., None] * b_t.float()[:, None]
            ssm_state = ssm_state * d_a + driven
            y_t = torch.einsum("bdn,bn->bd", ssm_state, c_t.float())
            y_t = (y_t.to(x.dtype) + core.D.to(x.dtype) * x_t) * F.silu(gate_t)
            output.append(y_t)
        y = torch.stack(output, dim=1)

    conv_state = conv_input[..., -core.d_conv:].contiguous()
    return core.out_proj(y), (conv_state, ssm_state)


@dataclass(frozen=True)
class Event:
    visual: torch.Tensor
    history: torch.Tensor
    visual_score: np.ndarray
    history_score: np.ndarray
    time: int
    protected: bool

    def similarity(self, other):
        return float(self.visual_score @ other.visual_score), float(self.history_score @ other.history_score)

    def map_tensors(self, transform):
        return replace(self, visual=transform(self.visual), history=transform(self.history))


class OrderedEvents:
    """Keep the protected initial event and causally select later events."""

    def __init__(self, thresholds, records=()):
        self.visual_threshold, self.history_threshold = thresholds
        self.records = list(records)

    def read(self, now, reference, age_scale):
        values = [
            torch.cat((
                event.visual, event.history,
                reference.new_tensor([log1p(now - event.time) / age_scale, 1.0]),
            ))
            for event in self.records
        ]
        values += [reference.new_zeros(2 * HISTORY_DIM + 2)] * (MEMORY_CAPACITY - len(values))
        return torch.cat(values)

    def write(self, event):
        if not self.records:
            self.records.append(event)
            return True, None, None

        visual, history = event.similarity(self.records[-1])
        scores = (visual, history)
        if visual >= self.visual_threshold and history >= self.history_threshold:
            return False, None, scores

        removed = None
        if len(self.records) == MEMORY_CAPACITY:
            # Iterate in chronological order so ties evict the earliest member
            # of the first equally redundant non-protected pair.
            first = max(
                range(1, MEMORY_CAPACITY - 1),
                key=lambda i: min(self.records[i].similarity(self.records[i + 1])),
            )
            removed = self.records.pop(first).time
        self.records.append(event)
        return True, removed, scores

    def map_tensors(self, transform):
        return OrderedEvents(
            (self.visual_threshold, self.history_threshold),
            [event.map_tensors(transform) for event in self.records],
        )


@dataclass
class EpisodeState:
    conv: torch.Tensor
    ssm: torch.Tensor
    events: OrderedEvents | None = None
    previous_feature: torch.Tensor | None = None
    history: torch.Tensor | None = None
    condition: torch.Tensor | None = None
    time: int = 0
    writes: list | None = None
    capacities: list = field(default_factory=list)

    def _map_tensors(self, transform):
        return replace(
            self,
            conv=transform(self.conv), ssm=transform(self.ssm),
            events=None if self.events is None else self.events.map_tensors(transform),
            previous_feature=None if self.previous_feature is None else transform(self.previous_feature),
            history=None if self.history is None else transform(self.history),
            condition=None if self.condition is None else transform(self.condition),
        )

    def detach(self):
        """Cut the graph at a TBPTT boundary while preserving all state values."""
        return self._map_tensors(torch.Tensor.detach)

    def to(self, device):
        return self._map_tensors(lambda value: value.to(device))

    def diagnostics(self):
        records = [] if self.events is None else self.events.records
        writes = self.writes or []
        return {
            "observations": self.time,
            "write_count": len(writes),
            "replacement_count": sum(row["replaced_time"] is not None for row in writes),
            "initial_retained": bool(records and records[0].protected and records[0].time == 0),
            "capacities": self.capacities.copy(),
            "events": writes.copy(),
            "final_event_times": [event.time for event in records],
        }


class ContinuousMemoryPolicy(DiffusionUnetImagePolicy):
    def __init__(self, cfg, unet, use_events, **common):
        super().__init__(**common, **unet)
        from mamba_ssm import Mamba

        self.use_events = use_events
        self.history_projection = nn.Linear(self.obs_feature_dim, HISTORY_DIM)
        self.history_mamba = Mamba(d_model=HISTORY_DIM, d_state=16, d_conv=4, expand=2)
        self.history_norm = nn.LayerNorm(HISTORY_DIM)
        self.past_head = nn.Sequential(
            nn.Linear(HISTORY_DIM, HISTORY_DIM), nn.GELU(),
            nn.Linear(HISTORY_DIM, PAST_STEPS * self.action_dim),
        )
        condition_dim = self.n_obs_steps * self.obs_feature_dim + HISTORY_DIM
        if use_events:
            # Stanford's encoder puts all RGB features before proprioception.
            proprio_dim = sum(
                prod(common["shape_meta"]["obs"][key]["shape"]) for key in self.obs_encoder.low_dim_keys
            )
            self.visual_dim = self.obs_feature_dim - proprio_dim
            self.event_visual_projection = nn.Linear(self.visual_dim, HISTORY_DIM)
            self.memory_projection = nn.Linear(MEMORY_CAPACITY * (2 * HISTORY_DIM + 2), HISTORY_DIM, bias=False)
            self.event_thresholds = cfg.get("event_thresholds")
            self.age_scale = log1p(cfg.get("max_episode_steps", 500))
            condition_dim += HISTORY_DIM
        self.encoder_chunk_size = cfg.get("history_encoder_chunk_size", 64)
        self.past_loss_weight = cfg.get("past_loss_weight", 0.1)
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim, local_cond_dim=None, global_cond_dim=condition_dim, **unet,
        )
        self._online = {}

    def encode_observations(self, obs):
        normalized = self.normalizer.normalize(obs)
        keys = tuple(normalized)
        chunks = zip(*(normalized[key].split(self.encoder_chunk_size) for key in keys))
        encoded = []
        for values in chunks:
            chunk = dict(zip(keys, values))
            if self.training and torch.is_grad_enabled():
                encoded.append(checkpoint(self.obs_encoder, chunk, use_reentrant=False))
            else:
                encoded.append(self.obs_encoder(chunk))
        return torch.cat(encoded, dim=0)

    def history_sequence(self, features, cache=None):
        history, cache = mamba_recurrence(self.history_mamba, self.history_projection(features), cache)
        return self.history_norm(history), cache

    def event_features(self, features, history):
        return self.event_visual_projection(features[..., :self.visual_dim]), history

    def _new_episode(self, diagnostics=False):
        conv, ssm = self.history_mamba.allocate_inference_cache(batch_size=1, max_seqlen=1)
        events = None
        if self.use_events:
            if self.event_thresholds is None:
                raise ValueError("M3 requires independent visual and history thresholds calibrated on training data.")
            events = OrderedEvents((self.event_thresholds["visual"], self.event_thresholds["history"]))
        return EpisodeState(conv, ssm, events, writes=[] if diagnostics else None)

    def _advance_episode(self, features, history, cache, state):
        previous = features[:1] if state.previous_feature is None else state.previous_feature[None]
        recent = torch.cat((previous, features[:-1]), dim=0)
        components = [recent, features, history]
        if state.events is not None:
            vision, history_content = self.event_features(features, history)
            with torch.no_grad():
                scores = torch.cat((
                    F.normalize(vision.float(), dim=-1),
                    F.normalize(history_content.float(), dim=-1),
                ), dim=-1).cpu().numpy()
            memory_rows = []
            for i, (v, h, score) in enumerate(zip(vision, history_content, scores)):
                now = state.time + i
                memory_rows.append(state.events.read(now, v, self.age_scale))
                event = Event(v, h, score[:HISTORY_DIM], score[HISTORY_DIM:], now, now == 0)
                written, removed, similarity = state.events.write(event)
                if state.writes is not None:
                    state.capacities.append(len(state.events.records))
                    if written:
                        state.writes.append({
                            "time": now, "replaced_time": removed,
                            "visual_similarity_to_latest": None if similarity is None else similarity[0],
                            "history_similarity_to_latest": None if similarity is None else similarity[1],
                        })
            components.append(self.memory_projection(torch.stack(memory_rows)))
        conditions = torch.cat(components, dim=-1)
        state.conv, state.ssm = cache
        state.previous_feature = features[-1]
        state.history = history[-1]
        state.condition = conditions[-1]
        state.time += len(features)
        return conditions

    def segment_condition(self, obs, carry=None):
        features = self.encode_observations(obs)
        state = self._new_episode() if carry is None else carry
        history, cache = self.history_sequence(features[None], (state.conv, state.ssm))
        conditions = self._advance_episode(features, history[0], cache, state)
        return conditions, history[0], state

    def past_loss(self, history, past_action, past_mask):
        label = self.normalizer["action"].normalize(past_action)
        prediction = self.past_head(history).reshape(-1, PAST_STEPS, self.action_dim)
        mask = past_mask.unsqueeze(-1).to(prediction.dtype)
        return ((prediction - label).square() * mask).sum() / (mask.sum() * self.action_dim).clamp_min(1)

    def compute_loss(self, batch, carry=None):
        condition, history, carry = self.segment_condition(batch["obs"], carry)
        if "anchors" in batch:
            condition, history = condition[batch["anchors"]], history[batch["anchors"]]
        action = self.normalizer["action"].normalize(batch["action"])
        noise = torch.randn_like(action)
        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (len(action),), device=action.device)
        noisy = self.noise_scheduler.add_noise(action, noise, timestep)
        prediction = self.model(noisy, timestep, global_cond=condition)
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = action
        else:
            raise ValueError(f"Unsupported prediction type {prediction_type}")
        loss_ddpm = F.mse_loss(prediction, target)
        loss_past = self.past_loss(history, batch["past_action"], batch["past_mask"])
        self.loss_parts = (loss_ddpm.detach(), loss_past.detach())
        return loss_ddpm + self.past_loss_weight * loss_past, carry

    def reset_memory(self, env_ids):
        for env_id in env_ids:
            self._online[env_id] = self._new_episode(diagnostics=True)

    def _online_states(self, env_ids):
        env_ids = tuple(env_ids)
        if not env_ids or len(set(env_ids)) != len(env_ids):
            raise ValueError("Provide a non-empty list of distinct environment IDs.")
        return [self._online[env_id] for env_id in env_ids]

    @torch.no_grad()
    def update_observation(self, obs, env_ids):
        """Advance each listed environment by one actual observation."""
        states = self._online_states(env_ids)
        features = self.encode_observations(obs)
        if len(features) != len(states):
            raise ValueError("The observation batch must correspond to the provided environment IDs.")
        cache = (
            torch.cat([state.conv for state in states]),
            torch.cat([state.ssm for state in states]),
        )
        history, (conv, ssm) = self.history_sequence(features[:, None], cache)
        for i, state in enumerate(states):
            self._advance_episode(features[i:i + 1], history[i], (conv[i:i + 1], ssm[i:i + 1]), state)

    def get_history_feature(self, env_ids):
        states = self._online_states(env_ids)
        if any(state.history is None for state in states):
            raise RuntimeError("Update each environment with its initial observation before reading memory.")
        return torch.stack([state.history for state in states])

    def memory_diagnostics(self, env_ids):
        return [state.diagnostics() for state in self._online_states(env_ids)]

    @torch.no_grad()
    def predict_stream(self, env_ids, generator=None):
        states = self._online_states(env_ids)
        if any(state.condition is None for state in states):
            raise RuntimeError("Update each environment with its initial observation before predicting.")
        condition = torch.stack([state.condition for state in states])
        data = condition.new_zeros(len(states), self.horizon, self.action_dim)
        sample = self.conditional_sample(
            data, torch.zeros_like(data, dtype=torch.bool),
            global_cond=condition, generator=generator, **self.kwargs,
        )
        action = self.normalizer["action"].unnormalize(sample)
        start = self.n_obs_steps - 1
        return {"action": action[:, start:start + self.n_action_steps], "action_pred": action}

    def predict_action(self, obs_dict=None, *, env_ids, generator=None):
        if obs_dict is not None:
            raise ValueError("Pass each new observation to update_observation before predicting.")
        return self.predict_stream(env_ids, generator=generator)

