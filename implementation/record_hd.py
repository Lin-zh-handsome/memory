from pathlib import Path
from collections import deque
import json,pickle,torch,numpy as np,random,imageio.v2 as imageio
from coffee_models import build_policy
from coffee_env import make_env,metadata,raw_obs
from coffee_data import observation_tensor
root=Path('/home/linzhenghong/project/coffee_dp_official/experiments/coffee_preparation_d0')
torch.set_num_threads(2)
saved=torch.load(root/'checkpoints/M2_seed0/epoch_50.pt',map_location='cpu',weights_only=False)
policy=build_policy('M2',saved['shape_meta'],saved['cfg'])
policy.load_state_dict(saved['ema']);policy.cuda();policy.normalizer.cuda();policy.eval()
with (root/'eval/initial_states.pkl').open('rb') as f:states=pickle.load(f)
spec=next(x for x in states if x['seed']==100006)
env=make_env(metadata(root/'data/coffee_preparation_d0.hdf5'))
try:
 for attempt in range(5):
  random.seed(attempt);np.random.seed(attempt);torch.manual_seed(attempt);torch.cuda.manual_seed_all(attempt)
  env.reset();env.reset_to(spec['state']);policy.reset();hist=deque([raw_obs(env)]*16,maxlen=16)
  path=root/f'videos/M2_epoch50_HD_attempt{attempt}.mp4'
  writer=imageio.get_writer(path,fps=10,codec='libx264',quality=9,macro_block_size=1)
  success=False;steps=0
  try:
   writer.append_data(env.env.sim.render(width=1280,height=720,camera_name='agentview')[::-1])
   while steps<800 and not success:
    with torch.no_grad():actions=policy.predict_action(observation_tensor(hist,'cuda',16))['action'][0].cpu().numpy()
    for a in actions:
     env.step(a);steps+=1;hist.append(raw_obs(env));success=bool(env.is_success()['task'])
     if steps%2==0 or success:writer.append_data(env.env.sim.render(width=1280,height=720,camera_name='agentview')[::-1])
     if success or steps>=800:break
    if steps%80<8:print(json.dumps({'attempt':attempt,'steps':steps,'success':success}),flush=True)
  finally:writer.close()
  result={'model':'M2','train_seed':0,'epoch':50,'env_seed':100006,'demo_rng_seed':attempt,'success':success,'steps':steps,'resolution':[1280,720],'purpose':'HD demonstration only; excluded from formal evaluations','video':str(path)}
  path.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
  if success:
   (root/'videos/HD_success_result.json').write_text(json.dumps(result,indent=2));break
finally:env.env.close()
