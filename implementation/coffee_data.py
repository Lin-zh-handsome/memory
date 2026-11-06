"""Episode-local windows with raw OSC delta actions, identical for both policies."""
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

RGB_KEYS = ['agentview_image','robot0_eye_in_hand_image']
STATE_KEYS = ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']


def window_indices(length,t,history):
    obs = np.maximum(np.arange(t-history+1,t+1),0)
    action = np.clip(np.arange(t-1,t-1+16),0,length-1)
    past = np.arange(t-8,t)
    return obs,action,np.clip(past,0,length-1),past>=0


class CoffeeDataset(Dataset):
    def __init__(self,root,history):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.history = history
        self.episodes = self.manifest['episodes']
        # Official SequenceSampler: horizon=16, pad_before=1, pad_after=7.
        self.ends = np.cumsum([x['length']-7 for x in self.episodes])
        self.arrays = OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def episode(self,i):
        if i not in self.arrays:
            p = self.root/self.episodes[i]['name']
            self.arrays[i] = {k:np.load(p/(k+'.npy'),mmap_mode='r')
                             for k in RGB_KEYS+STATE_KEYS+['action']}
            if len(self.arrays)>16:self.arrays.popitem(last=False)
        self.arrays.move_to_end(i)
        return self.arrays[i]

    def __getitem__(self,index):
        i = int(np.searchsorted(self.ends,index,side='right'))
        t = int(index-(self.ends[i-1] if i else 0))
        a = self.episode(i)
        oi,ai,pi,mask = window_indices(len(a['action']),t,self.history)
        obs = {k:np.array(a[k][oi],dtype=np.float32) for k in STATE_KEYS}
        obs.update({k:np.moveaxis(np.array(a[k][oi]),-1,1).astype(np.float32)/255 for k in RGB_KEYS})
        return {'obs':obs,'action':np.array(a['action'][ai]),
                'past_action':np.array(a['action'][pi]),'past_mask':mask.astype(np.float32)}

    def normalizer(self):
        norm = LinearNormalizer()
        for k in STATE_KEYS+['action']:
            # Numeric statistics are small; load owning arrays so the temporary list
            # does not retain one open memory mapping per episode.
            values = np.concatenate([np.load(self.root/x['name']/(k+'.npy')) for x in self.episodes])
            norm[k] = SingleFieldLinearNormalizer.create_fit(values,mode='limits',last_n_dims=1)
        for k in RGB_KEYS:
            norm[k] = SingleFieldLinearNormalizer.create_identity()
        return norm


def observation_tensor(hist,device,history):
    seq = list(hist)[-history:]
    out = {}
    for k in RGB_KEYS+STATE_KEYS:
        v = np.stack([x[k] for x in seq])
        if k in RGB_KEYS:
            v = np.moveaxis(v,-1,1).astype(np.float32)/255
        out[k] = torch.as_tensor(np.ascontiguousarray(v),device=device,dtype=torch.float32)[None]
    return out
