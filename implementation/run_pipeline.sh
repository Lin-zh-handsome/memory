#!/bin/bash
set -euo pipefail
ROOT=$HOME/project/coffee_dp_official
EXP=$ROOT/experiments/coffee_preparation_d0
export PYTHONNOUSERSITE=1
export PATH="$HOME/miniconda3/envs/coffee-dp/bin:$PATH"
export PYTHONPATH="$ROOT/source/diffusion_policy:$ROOT/source/mimicgen:$ROOT/implementation"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MUJOCO_GL=egl
cd "$ROOT/implementation"
# The pipeline may be restarted without launching a second active dispatcher.
exec 9>"$EXP/logs/pipeline.lock"
flock -n 9 || exit 0
test -f "$EXP/logs/environment_ready"
python fix_robomimic_import.py
python - <<'PY'
import json,subprocess,sys
from pathlib import Path
import torch,torchvision,diffusers,robosuite,robomimic,mamba_ssm,mujoco
root=Path.home()/'project/coffee_dp_official'
versions={k:m.__version__ for k,m in locals().copy().items() if k in ['torch','torchvision','diffusers','robosuite','mamba_ssm','mujoco']}
versions.update(python=sys.version,cuda=torch.version.cuda,robomimic='0.3.0',
  diffusion_policy_commit=subprocess.check_output(['git','-C',str(root/'source/diffusion_policy'),'rev-parse','HEAD'],text=True).strip(),
  mimicgen_commit=subprocess.check_output(['git','-C',str(root/'source/mimicgen'),'rev-parse','HEAD'],text=True).strip())
(root/'experiments/coffee_preparation_d0/configs/code_versions.json').write_text(json.dumps(versions,indent=2))
PY
if [ ! -f "$EXP/data/coffee_preparation_d0.hdf5" ]; then
  # HDF5 rejects a truncated file using its stored EOF; do not accept a partial download.
  python - "$EXP/data/coffee_preparation_d0.hdf5.part" <<'PY'
import sys,time
from pathlib import Path
import h5py
p=Path(sys.argv[1])
while True:
 try:
  with h5py.File(p,'r') as f:
   names=list(f['data']);last=f['data'][names[-1]]
   last['actions'][-1];last['states'][-1]
  break
 except (OSError,KeyError):time.sleep(30)
p.rename(p.with_suffix(''))
PY
fi
if [ ! -f "$EXP/data/visual/manifest.json" ]; then
  GPUS=$(python - <<'PY'
from dispatch import available_gpus
print(','.join(str(i) for i,u in available_gpus()[:1]))
PY
)
  test -n "$GPUS"
  python -u prepare_data.py --root "$EXP" --gpus "$GPUS" > "$EXP/logs/prepare.log" 2>&1
fi
GPU=$(python - <<'PY'
from dispatch import available_gpus
print(available_gpus()[0][0])
PY
)
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
if [ ! -f "$EXP/logs/interface_checks.json" ]; then
  python -u check_interfaces.py --root "$EXP" > "$EXP/logs/interface_checks.log" 2>&1
fi
python -u evaluate.py --root "$EXP" --init > "$EXP/logs/init_states.log" 2>&1
unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID
python -u dispatch.py --root "$EXP" > "$EXP/logs/dispatch.log" 2>&1
