"""Use MimicGen's registered CoffeePreparation_D0 and original controller metadata."""
import copy
import json
import random
import numpy as np
import h5py
from coffee_data import RGB_KEYS, STATE_KEYS


def metadata(path):
    with h5py.File(path,'r') as f:
        return json.loads(f['data'].attrs['env_args'])


def make_env(meta):
    import mimicgen
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.env_utils as EnvUtils
    ObsUtils.initialize_obs_utils_with_obs_specs({'obs':{'rgb':RGB_KEYS,'low_dim':STATE_KEYS}})
    m = copy.deepcopy(meta)
    assert m['env_name'] == 'CoffeePreparation_D0',m['env_name']
    m['env_kwargs'].update(camera_names=['agentview','robot0_eye_in_hand'],
        camera_heights=84,camera_widths=84,camera_depths=False,
        use_camera_obs=True,has_renderer=False,has_offscreen_renderer=True,
        ignore_done=True,horizon=800)
    return EnvUtils.create_env_from_metadata(m,render=False,render_offscreen=True,use_image_obs=True)


def raw_obs(env):
    # Raw robosuite observations (HWC uint8), identical in conversion and evaluation.
    # reset_to and step already refresh sensors; do not render a second time.
    obs = env.env._get_observations(force_update=False)
    return {k:np.array(obs[k][::-1] if k in RGB_KEYS else obs[k],copy=True)
            for k in RGB_KEYS+STATE_KEYS}


def seed_reset(env,seed):
    random.seed(seed)
    np.random.seed(seed)
    env.reset()
    return env.get_state()


def stage_metrics(env):
    return {k:bool(v) for k,v in env.env._get_partial_task_metrics().items()}
