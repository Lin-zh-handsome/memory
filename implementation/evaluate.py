"""Batched closed-loop inference with independent official MimicGen simulators."""
import argparse
from collections import deque
import csv
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import random
import time
import numpy as np
import torch
from coffee_data import RGB_KEYS, STATE_KEYS,observation_tensor
from coffee_env import make_env,metadata,raw_obs,seed_reset,stage_metrics


def worker(conn,meta):
    env=make_env(meta)
    frames=[];ever={};steps=0;success=False
    try:
        while True:
            cmd,arg=conn.recv()
            if cmd=='close':break
            if cmd=='reset':
                env.reset();env.reset_to(arg)
                frames=[];ever={};steps=0;success=False
                obs=raw_obs(env); frames.append(obs[RGB_KEYS[0]])
                conn.send(obs)
            if cmd=='step':
                observations=[]
                for action in arg:
                    env.step(action)
                    steps+=1
                    obs=raw_obs(env);observations.append(obs)
                    metrics=stage_metrics(env)
                    for k,v in metrics.items():ever[k]=ever.get(k,False) or v
                    success=success or bool(env.is_success()['task'])
                    if steps%4==0: frames.append(obs[RGB_KEYS[0]])
                    if success or steps>=800:break
                done=success or steps>=800
                conn.send({'observations':observations,'done':done,'success':success,
                           'steps':steps,'stages':ever,'frames':frames if done else None})
    finally:
        env.env.close();conn.close()


def init_states(root):
    root=Path(root);target=root/'eval/initial_states.pkl'
    if target.exists():return
    env=make_env(metadata(root/'data/coffee_preparation_d0.hdf5'))
    states=[]
    for seed in range(100000,100050):
        state=seed_reset(env,seed)
        # Store full model XML as well as simulator state: fixtures can live in model arrays.
        env.reset_to(state)
        restored=env.get_state()
        np.testing.assert_allclose(state['states'],restored['states'],atol=1e-10,rtol=0)
        states.append({'seed':seed,'state':state})
    env.env.close()
    with target.open('wb') as f:pickle.dump(states,f)
    (root/'eval/initialization.json').write_text(json.dumps({
        'seeds':list(range(100000,100050)),
        'method':'Official D0 reset sampled once; save and replay model XML and flattened simulator state.'},indent=2))


def evaluate(policy,root,variant,train_seed,epoch,n_envs=10):
    root=Path(root)
    episode_file=root/f'eval/{variant}_seed{train_seed}_epoch{epoch}_episodes.csv'
    fields=['model','train_seed','epoch','env_seed','success','steps','stages']
    rows=[]
    if episode_file.exists():
        with episode_file.open() as f:rows=list(csv.DictReader(f))
    completed={int(x['env_seed']) for x in rows}
    with (root/'eval/initial_states.pkl').open('rb') as f:states=pickle.load(f)
    todo=[x for x in states if x['seed'] not in completed]
    meta=metadata(root/'data/coffee_preparation_d0.hdf5')
    ctx=mp.get_context('spawn'); pairs=[]
    for _ in range(min(n_envs,len(todo))):
        parent,child=ctx.Pipe();proc=ctx.Process(target=worker,args=(child,meta));proc.start();child.close()
        pairs.append((parent,proc))
    policy.eval();policy.reset()
    device=policy.device;history=2 if variant=='M0' else 16
    try:
        for start in range(0,len(todo),n_envs):
            group=todo[start:start+n_envs]
            active={};histories={}
            for i,spec in enumerate(group):pairs[i][0].send(('reset',spec['state']))
            for i,spec in enumerate(group):
                obs=pairs[i][0].recv();histories[i]=deque([obs]*16,maxlen=16);active[i]=spec
            while active:
                ids=list(active)
                batch=[observation_tensor(histories[i],device,history) for i in ids]
                obs={k:torch.cat([x[k] for x in batch]) for k in batch[0]}
                with torch.no_grad():actions=policy.predict_action(obs)['action'].cpu().numpy()
                for i,a in zip(ids,actions):pairs[i][0].send(('step',a))
                for i in ids:
                    result=pairs[i][0].recv()
                    histories[i].extend(result['observations'])
                    if result['done']:
                        row=dict(model=variant,train_seed=train_seed,epoch=epoch,
                            env_seed=active[i]['seed'],success=int(result['success']),steps=result['steps'],
                            stages=json.dumps(result['stages'],sort_keys=True))
                        exists=episode_file.exists()
                        with episode_file.open('a',newline='') as f:
                            w=csv.DictWriter(f,fields)
                            if not exists:w.writeheader()
                            w.writerow(row)
                        rows.append(row)
                        outcome='success' if result['success'] else 'failure'
                        video=root/f'videos/{variant}_{outcome}.mp4'
                        if not video.exists():
                            import imageio.v2 as imageio
                            imageio.mimsave(video,result['frames'],fps=5,macro_block_size=1)
                            video.with_suffix('.json').write_text(json.dumps(row,indent=2))
                        print(json.dumps({'evaluation_episode':row}),flush=True)
                        del active[i]
    finally:
        for conn,proc in pairs:
            if proc.is_alive():conn.send(('close',None))
        for conn,proc in pairs:proc.join(timeout=30);conn.close()
    assert len(rows)==50 and len({int(x['env_seed']) for x in rows})==50
    success=sum(int(x['success']) for x in rows)
    result={'model':variant,'train_seed':train_seed,'epoch':epoch,
            'successful_episodes':success,'episodes':50,'success_rate':success/50}
    (root/f'eval/{variant}_seed{train_seed}_epoch{epoch}.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--init',action='store_true')
    args=p.parse_args()
    if args.init:init_states(args.root)
