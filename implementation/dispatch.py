"""Run the six authorized jobs only on GPUs with no compute processes."""
import argparse
import csv
import ctypes
import errno
import json
import os
import select
from pathlib import Path
import subprocess
import sys
import time


class AdoptedRun:
    def __init__(self,pid,completion_file):
        self.pid=pid
        self.completion_file=completion_file

    def poll(self):
        try:
            os.kill(self.pid,0)
            return None
        except ProcessLookupError:
            return 0 if self.completion_file.exists() else 1


def available_gpus():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    occupied=set(procs.splitlines())
    return [(int(i),u.strip()) for i,u,mem,util in csv.reader(gpu.splitlines())
            if u.strip() not in occupied and int(mem)<500 and int(util)<5]


def wait_for_completion_or_hour(running):
    descriptors=[]
    try:
        libc=ctypes.CDLL(None,use_errno=True)
        libc.syscall.restype=ctypes.c_long
        for gpu,proc,log in running.values():
            # The Conda Python build lacks os.pidfd_open; use the kernel interface.
            descriptor=libc.syscall(434,proc.pid,0)
            if descriptor<0:
                code=ctypes.get_errno()
                if code==errno.ESRCH:return
                raise OSError(code,os.strerror(code))
            descriptors.append(descriptor)
        if descriptors:select.select(descriptors,[],[],3600)
        else:time.sleep(3600)
    finally:
        for descriptor in descriptors:os.close(descriptor)


def shared_seed0_gpu(running,allow_remaining=False):
    if len(running)!=1:return None
    job,(gpu,proc,log)=next(iter(running.items()))
    if job[1]!=0 and not allow_remaining:return None
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    uuid,free=next(csv.reader(row.splitlines()))
    if int(free)<24000:return None
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    own_pids={p.pid for g,p,l in running.values()}
    if any(u.strip()==uuid.strip() and int(pid) not in own_pids for u,pid in csv.reader(apps.splitlines())):
        return None
    return gpu,uuid.strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--max-gpus',type=int,default=1,choices=[1,2,3,4])
    p.add_argument('--adopt-pid',type=int);p.add_argument('--adopt-gpu',type=int)
    p.add_argument('--adopt-model',choices=['M0','M2']);p.add_argument('--adopt-seed',type=int)
    p.add_argument('--parallel-seed0',action='store_true')
    p.add_argument('--parallel-remaining',action='store_true')
    p.add_argument('--exclusive', action='store_true')
    args=p.parse_args();root=Path(args.root)
    config=json.loads((root/'configs/protocol.json').read_text())
    args.parallel_seed0=args.parallel_seed0 or config.get('resource_policy',{}).get('parallel_seed0_pair',False)
    args.parallel_remaining=args.parallel_remaining or config.get('resource_policy',{}).get('parallel_remaining_pair',False)
    # User priority: finish the seed-0 pair first, with M2 seed 0 first.
    jobs=[('M2',0),('M0',0)]+[(m,s) for s in [1,2] for m in ['M2','M0']]
    jobs=[j for j in jobs if not (root/f'logs/{j[0]}_seed{j[1]}_complete.json').exists()]
    running={};failed=[];here=Path(__file__).parent
    if args.adopt_pid:
        job=(args.adopt_model,args.adopt_seed)
        cmd=Path(f'/proc/{args.adopt_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert 'coffee_dp_official/implementation/train.py' in cmd and f'--model {job[0]}' in cmd and f'--seed {job[1]}' in cmd
        jobs.remove(job)
        running[job]=(args.adopt_gpu,AdoptedRun(args.adopt_pid,root/f'logs/{job[0]}_seed{job[1]}_complete.json'),None)
        print(json.dumps({'adopted':job,'pid':args.adopt_pid,'gpu':args.adopt_gpu}),flush=True)
    print(json.dumps({'pending_queue':jobs,'max_gpus':args.max_gpus,'parallel_seed0':args.parallel_seed0}),flush=True)
    while jobs or running:
        for job,(gpu,proc,log) in list(running.items()):
            code=proc.poll()
            if code is not None:
                if log is not None:log.close()
                del running[job]
                print(json.dumps({'finished':job,'gpu':gpu,'exit_code':code}),flush=True)
                if code:failed.append({'job':job,'exit_code':code})
        physical_gpus={g for g,p,l in running.values()}
        candidates=[]
        if jobs and not args.exclusive and ((args.parallel_seed0 and jobs[0][1]==0) or args.parallel_remaining):
            shared=shared_seed0_gpu(running,args.parallel_remaining)
            if shared is not None:candidates.append(shared)
        seed0_active=any(job[1]==0 for job in running)
        if jobs and not candidates and len(physical_gpus)<args.max_gpus and not (seed0_active and jobs[0][1]!=0):
            candidates=available_gpus()
        for gpu,uuid in candidates:
            if not jobs:break
            if gpu in physical_gpus:
                if not (len(running)==1 and ((args.parallel_seed0 and jobs[0][1]==0) or args.parallel_remaining)):continue
            elif len(physical_gpus)>=args.max_gpus:break
            if any(job[1]==0 for job in running) and jobs[0][1]!=0:break
            model,seed=jobs.pop(0)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu))
            log=(root/f'logs/{model}_seed{seed}.log').open('a')
            proc=subprocess.Popen([sys.executable,'-u',str(here/'train.py'),'--root',str(root),'--model',model,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
            running[(model,seed)]=(gpu,proc,log)
            physical_gpus.add(gpu)
            print(json.dumps({'started':[model,seed],'gpu':gpu,'uuid':uuid,'pid':proc.pid}),flush=True)
        with (root/'logs/gpu_resources.jsonl').open('a') as f:
            record=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            f.write(json.dumps({'time':time.time(),'gpus':record,'running':{f'{j[0]}_seed{j[1]}':{'gpu':g,'pid':p.pid} for j,(g,p,l) in running.items()}})+'\n')
        if jobs or running:wait_for_completion_or_hour(running)
    (root/'logs/dispatch_finished.json').write_text(json.dumps({'failed':failed}))
    subprocess.run([sys.executable,str(here/'summarize.py'),'--root',str(root)],check=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
