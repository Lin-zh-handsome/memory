"""robomimic 0.3's legacy mujoco_py import is unused with robosuite 1.4's native MuJoCo."""
from pathlib import Path
import robomimic
p=Path(robomimic.__file__).parent/'envs/env_robosuite.py'
s=p.read_text()
if '\nimport mujoco_py\n' in s:
    s=s.replace('\nimport mujoco_py\n','\ntry:\n    import mujoco_py\nexcept ImportError:\n    mujoco_py = None  # legacy robosuite 0.x only; experiment uses native MuJoCo\n')
    p.write_text(s)
print(p)
