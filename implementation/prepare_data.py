"""Inspect official data and replay each pre-action state to aligned RGB observations."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
from coffee_data import RGB_KEYS, STATE_KEYS


def convert_shard(task):
    source,out,names,gpu = task
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['MUJOCO_EGL_DEVICE_ID']=str(gpu)
    os.environ['MUJOCO_GL']='egl'
    from coffee_env import make_env,raw_obs,metadata
    env = None
    result=[]
    with h5py.File(source,'r') as f:
        for name in names:
            dst=Path(out)/name
            if (dst/'complete.json').exists():
                result.append(json.loads((dst/'complete.json').read_text())); continue
            demo=f['data'][name]
            n=len(demo['actions'])
            dst.mkdir(parents=True,exist_ok=True)
            if 'obs' in demo and all(k in demo['obs'] for k in RGB_KEYS+STATE_KEYS):
                for k in RGB_KEYS+STATE_KEYS:
                    values=np.asarray(demo['obs'][k],dtype=(np.uint8 if k in RGB_KEYS else np.float32))
                    assert len(values)==n,(name,k,values.shape,n)
                    if k in RGB_KEYS:assert values.shape[1:]==(84,84,3),values.shape
                    np.save(dst/(k+'.npy'),values)
                np.save(dst/'action.npy',np.array(demo['actions'],dtype=np.float32))
                record={'name':name,'length':n,'observation_source':'official_stored_rgb'}
                (dst/'complete.json').write_text(json.dumps(record))
                result.append(record)
                print(json.dumps({'copied_official_observations':name,'length':n}),flush=True)
                continue
            if env is None:env=make_env(metadata(source))
            env.reset()
            initial={'states':demo['states'][0], 'model':demo.attrs['model_file']}
            env.reset_to(initial)
            arrays={}
            for t in range(n):
                # states[t] is the pre-action state paired with actions[t].
                env.reset_to({'states':demo['states'][t]})
                obs=raw_obs(env)
                if t==0:
                    arrays={k:np.lib.format.open_memmap(dst/(k+'.npy'),mode='w+',
                        dtype=(np.uint8 if k in RGB_KEYS else np.float32),shape=(n,*obs[k].shape))
                        for k in RGB_KEYS+STATE_KEYS}
                for k in arrays: arrays[k][t]=obs[k]
            for a in arrays.values(): a.flush()
            np.save(dst/'action.npy',np.array(demo['actions'],dtype=np.float32))
            record={'name':name,'length':n}
            (dst/'complete.json').write_text(json.dumps(record))
            result.append(record)
            print(json.dumps({'converted':name,'length':n,'gpu':gpu}),flush=True)
    if env is not None:env.env.close()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--gpus',default='1,2,3,5')
    args=p.parse_args(); root=Path(args.root); source=root/'data/coffee_preparation_d0.hdf5'
    part=source.with_suffix('.hdf5.part')
    if not source.exists():
        # Require the download process to have explicitly completed before renaming.
        raise FileNotFoundError(source)
    with h5py.File(source,'r') as f:
        names=sorted(f['data'],key=lambda x:int(x.split('_')[-1]))
        d=f['data'][names[0]]
        report={'episodes':len(names),'total_steps':sum(len(f['data'][n]['actions']) for n in names),
                'first_episode':names[0], 'env_args':json.loads(f['data'].attrs['env_args']),
                'structure':{}}
        def visit(n,o):
            if isinstance(o,h5py.Dataset): report['structure'][n]={'shape':o.shape,'dtype':str(o.dtype)}
        d.visititems(visit)
        report['action_dim']=d['actions'].shape[-1]
        print(json.dumps(report,indent=2),flush=True)
        (root/'configs/data_inspection.json').write_text(json.dumps(report,indent=2))
    out=root/'data/visual'; out.mkdir(exist_ok=True)
    gpus=[int(x) for x in args.gpus.split(',')]
    tasks=[(str(source),str(out),names[i::len(gpus)],g) for i,g in enumerate(gpus)]
    import multiprocessing as mp
    with concurrent.futures.ProcessPoolExecutor(len(gpus),mp_context=mp.get_context('spawn')) as pool:
        records=sum(list(pool.map(convert_shard,tasks)),[])
    records.sort(key=lambda x:int(x['name'].split('_')[-1]))
    manifest={'episodes':records,'source':str(source),'action_dim':report['action_dim'],
              'cameras':['agentview','robot0_eye_in_hand'],'resolution':[84,84],
              'alignment':'official stored obs[t] paired with actions[t]; verified against state replay; raw OSC delta action'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
