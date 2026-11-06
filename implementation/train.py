"""Full 250-epoch official CNN-DP protocol, resumable per run and per evaluation episode."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.common.pytorch_util import dict_apply
from coffee_data import CoffeeDataset
from coffee_models import build_policy
from evaluate import evaluate


def shape_metadata(dataset):
    from coffee_data import RGB_KEYS,STATE_KEYS
    a=dataset.episode(0)
    return {'action':{'shape':[a['action'].shape[-1]]},'obs':{
        **{k:{'shape':[3,84,84],'type':'rgb'} for k in RGB_KEYS},
        **{k:{'shape':list(a[k].shape[1:]),'type':'low_dim'} for k in STATE_KEYS}}}


def save_atomic(payload,path):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model',choices=['M0','M2'],required=True)
    p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    root=Path(args.root);cfg=json.loads((root/'configs/protocol.json').read_text())
    seed=args.seed;variant=args.model;tag=f'{variant}_seed{seed}'
    torch.set_num_threads(4)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    ds=CoffeeDataset(root/'data/visual',2 if variant=='M0' else 16)
    meta=shape_metadata(ds)
    policy=build_policy(variant,meta,cfg);policy.set_normalizer(ds.normalizer());policy.cuda()
    ema_policy=copy.deepcopy(policy)
    ema=EMAModel(ema_policy,power=0.75)
    optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-4,betas=(0.95,0.999),eps=1e-8,weight_decay=1e-6)
    loader=DataLoader(ds,batch_size=64,shuffle=True,num_workers=4,pin_memory=True,
                      persistent_workers=True,drop_last=False)
    lr=get_scheduler('cosine',optimizer,num_warmup_steps=500,num_training_steps=len(loader)*250)
    ckdir=root/'checkpoints'/tag;ckdir.mkdir(exist_ok=True)
    latest=ckdir/'latest.pt';start=1;global_step=0
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        policy.load_state_dict(state['policy']);ema_policy.load_state_dict(state['ema'])
        policy.normalizer.cuda();ema_policy.normalizer.cuda()
        optimizer.load_state_dict(state['optimizer']);lr.load_state_dict(state['lr'])
        for s in optimizer.state.values():
            for k,v in s.items():
                if torch.is_tensor(v):s[k]=v.cuda()
        ema.optimization_step=state['global_step'];global_step=state['global_step'];start=state['epoch']+1
        random.setstate(state['random']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state_all(state['cuda'])
        print(json.dumps({'resumed_epoch':start,'global_step':global_step}),flush=True)
    (root/f'configs/{tag}.json').write_text(json.dumps({**cfg,'model':variant,'seed':seed,'shape_meta':meta},indent=2))
    # Finish an interrupted checkpoint evaluation before advancing training.
    for epoch in cfg['eval_epochs']:
        path=ckdir/f'epoch_{epoch}.pt'
        result=root/f'eval/{tag}_epoch{epoch}.json'
        if path.exists() and not result.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            ema_policy.load_state_dict(saved['ema']);ema_policy.normalizer.cuda()
            evaluate(ema_policy,root,variant,seed,epoch,cfg['n_eval_envs'])
    if latest.exists():
        ema_policy.load_state_dict(state['ema'])
        ema_policy.normalizer.cuda()
    for epoch in range(start,251):
        began=time.monotonic();policy.train();totals=np.zeros(3);count=0
        torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            batch=dict_apply(batch,lambda x:x.cuda(non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss=policy.compute_loss(batch);loss.backward();optimizer.step();lr.step();ema.step(policy)
            parts=policy.loss_parts if variant=='M2' else (loss.detach(),torch.zeros((),device='cuda'))
            totals+=np.array([loss.item(),parts[0].item(),parts[1].item()])*len(batch['action'])
            count+=len(batch['action']);global_step+=1
            if global_step%100==0:
                print(json.dumps({'model':variant,'seed':seed,'epoch':epoch,'step':global_step,
                    'loss':loss.item(),'lr':lr.get_last_lr()[0]}),flush=True)
        row={'model':variant,'seed':seed,'epoch':epoch,'global_step':global_step,
             'loss':totals[0]/count,'ddpm_loss':totals[1]/count,'past_loss':totals[2]/count,
             'seconds':time.monotonic()-began,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated()}
        logfile=root/f'logs/{tag}_epochs.csv';exists=logfile.exists()
        with logfile.open('a',newline='') as f:
            w=csv.DictWriter(f,list(row))
            if not exists:w.writeheader()
            w.writerow(row)
        payload={'epoch':epoch,'global_step':global_step,'policy':policy.state_dict(),
                 'ema':ema_policy.state_dict(),'optimizer':optimizer.state_dict(),'lr':lr.state_dict(),
                 'random':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                 'cuda':torch.cuda.get_rng_state_all(),'cfg':cfg,'shape_meta':meta}
        save_atomic(payload,latest)
        print(json.dumps({'epoch_complete':row}),flush=True)
        if epoch in cfg['eval_epochs']:
            save_atomic({'epoch':epoch,'ema':ema_policy.state_dict(),'cfg':cfg,'shape_meta':meta},ckdir/f'epoch_{epoch}.pt')
            evaluate(ema_policy,root,variant,seed,epoch,cfg['n_eval_envs'])
    (root/f'logs/{tag}_complete.json').write_text(json.dumps({'epochs':250,'global_step':global_step}))


if __name__=='__main__':main()
