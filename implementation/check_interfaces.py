"""Necessary data alignment, auxiliary-only gradient, and CUDA kernel checks; no optimizer steps."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply
from coffee_data import CoffeeDataset,window_indices,RGB_KEYS
from coffee_models import build_policy
from train import shape_metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root);cfg=json.loads((root/'configs/protocol.json').read_text())
    torch.set_num_threads(4);torch.manual_seed(0)
    ds0=CoffeeDataset(root/'data/visual',2);ds2=CoffeeDataset(root/'data/visual',16)
    # Check beginning, middle, end and transition to next episode on the real corpus.
    probes=[0,1,8,15,int(ds0.ends[0]-1),int(ds0.ends[0])]
    for index in probes:
        m0,m2=ds0[index],ds2[index]
        for k in m0['obs']:np.testing.assert_array_equal(m0['obs'][k],m2['obs'][k][-2:])
        np.testing.assert_array_equal(m0['action'],m2['action'])
        episode=int(np.searchsorted(ds0.ends,index,side='right'))
        t=index-(int(ds0.ends[episode-1]) if episode else 0)
        arrays=ds0.episode(episode)
        oi,ai,pi,mask=window_indices(len(arrays['action']),t,16)
        np.testing.assert_array_equal(m0['action'],arrays['action'][ai])
        for k in m2['obs']:
            expected=np.array(arrays[k][oi],dtype=np.float32)
            if k in RGB_KEYS:expected=np.moveaxis(expected,-1,1)/255
            np.testing.assert_array_equal(m2['obs'][k],expected)
        np.testing.assert_array_equal(m2['past_mask'],mask)
        np.testing.assert_array_equal(m2['past_action'][mask],arrays['action'][pi[mask]])
        np.testing.assert_array_equal(m0['action'][1],arrays['action'][t])
        assert mask.sum()==min(t,8)
    meta=shape_metadata(ds2);normalizer=ds2.normalizer();records={}
    # Real valid labels (t>=8), to distinguish connected gradients from all-masked labels.
    samples=[ds2[15],ds2[16]]
    from torch.utils.data._utils.collate import default_collate
    batch=dict_apply(default_collate(samples),lambda x:x.cuda())
    policy=build_policy('M2',meta,cfg);policy.set_normalizer(normalizer);policy.cuda()
    policy.train();_,h=policy.encode_history(batch['obs'])
    aux=policy.past_loss(h,batch['past_action'],batch['past_mask']);aux.backward()
    norms={}
    for name,module in [('mamba',policy.history_mamba),('encoder',policy.obs_encoder),('projection',policy.history_projection)]:
        total=sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)
        assert np.isfinite(total) and total>0,(name,total)
        norms[name]=total
    for name,module in policy.obs_encoder.key_model_map.items():
        total=sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)
        assert np.isfinite(total) and total>0,(name,total)
        norms['encoder_'+name]=total
    policy.zero_grad(set_to_none=True)
    records['auxiliary_gradient_abs_sum']=norms
    del policy,batch;torch.cuda.empty_cache()
    for variant,ds in [('M0',ds0),('M2',ds2)]:
        policy=build_policy(variant,meta,cfg);policy.set_normalizer(normalizer);policy.cuda()
        batch=dict_apply(default_collate([ds[15],ds[16]]),lambda x:x.cuda())
        policy.train();torch.cuda.synchronize();began=time.monotonic()
        loss=policy.compute_loss(batch);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss)
        records[variant]={'parameters_total':sum(p.numel() for p in policy.parameters()),
            'trainable_parameters':sum(p.numel() for p in policy.parameters() if p.requires_grad),
            'interface_forward_backward_batch2_seconds':time.monotonic()-began}
        policy.zero_grad(set_to_none=True);policy.eval()
        torch.cuda.synchronize();began=time.monotonic()
        with torch.no_grad():prediction=policy.predict_action(batch['obs'])
        torch.cuda.synchronize()
        torch.testing.assert_close(prediction['action'],prediction['action_pred'][:,1:9])
        records[variant]['inference_batch2_seconds']=time.monotonic()-began
        if variant=='M2':
            records[variant]['auxiliary_head_parameters']=sum(p.numel() for p in policy.past_head.parameters())
            records[variant]['mamba_parameters']=sum(p.numel() for p in policy.history_mamba.parameters())
            records[variant]['history_projection_parameters']=sum(p.numel() for p in policy.history_projection.parameters())
            records[variant]['history_layernorm_parameters']=sum(p.numel() for p in policy.history_norm.parameters())
        del policy,batch;torch.cuda.empty_cache()
    records['notes']='Interface pass timing, not warmed-up benchmark; end-to-end epoch timings recorded during formal training.'
    records['alignment_probes']=probes
    (root/'logs/interface_checks.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2),flush=True)


if __name__=='__main__':main()
