# M5 下一关键状态视觉辅助监督

M5 在 M4 上更换辅助任务。当前观测、连续 Mamba 历史和四头注意力读出形成残差潜在向量。动作网络用这个 256 维向量接收记忆条件。辅助预测头只在训练时从这份向量预测下一操作完成的 512 维视觉特征。

## 数据对齐与目标

RMBench 的 `_take_picture()` 每保存一张原始采集帧就增加一次官方语言标注计数。`create_xpolicylab_hdf5` 使用 `state[:-1]` 和 `RGB[:-1]` 作为观察，使用下一帧 `state[1:]` 作为关节动作标签。因此每条示范的 0 基索引保持原样，只有采集时间轴最后的 RGB 不在训练数组里。事件边界保持原始间隔，并把最后一个事件映射到最后一个有配对动作的 RGB。每个事件选择 `end_frame - 1`，监督覆盖事件前 `floor(length/2)` 个锚点。同一事件内的连续语言描述合并。标签只进入辅助损失。

50 条示范通过官方与准备数据的关节、动作全数组核对，以及每个事件的首尾 RGB 对照。共 300 个目标、51,077 个配对帧及 25,478 个有效辅助锚点。见 [目标统计](annotations/statistics.json) 和 [逐事件索引](annotations/events.json)。目标特征可从原始数据和公开权重生成；索引中的 `goal_feature` 是离线特征矩阵行号，`valid_mask` 用半开区间表示。

## 训练与评测

使用原 M4 种子对应的已训练数据标定阈值，训练权重独立从头初始化。保持同一批 50 条示范、250 轮、每 64 锚点截断、相同动作和采样协议。第 1 轮权重为 0，第 10 轮达到 0.02，随后固定。取五个完整检查点的最高成功率，同分取较早的轮次。种子 0 达到 40% 后自动执行种子 1。

下面的命令从本仓库服务器工作树执行。沿用 M4 的服务器环境和 RMBench 补丁，路径变量指向既有环境、归档和准备数据。

```bash
CODE=implementation/rmbench
ROOT=/path/to/server/experiments/rmbench/m5_goal_keyframe_aux
RMBENCH=/path/to/official/RMBench
DATA=/path/to/cover_blocks/prepared
ARCHIVE=/path/to/cover_blocks/demo_clean.zip
DPSOURCE=/path/to/official/diffusion_policy
CUROBO=/path/to/curobo/src
PYTHON=/path/to/conda/env/bin/python
mkdir -p "$ROOT/teacher"
curl -fL https://download.pytorch.org/models/resnet18-f37072fd.pth \
  -o "$ROOT/teacher/resnet18-f37072fd.pth"
PYTHONPATH="$RMBENCH" "$PYTHON" "$CODE/goal_targets.py" \
  --archive "$ARCHIVE" --data "$DATA" --weights "$ROOT/teacher/resnet18-f37072fd.pth" \
  --output "$ROOT/annotations"
export PYTHONPATH="$DPSOURCE:$CUROBO"
"$PYTHON" "$CODE/test_models.py" --device cuda
"$PYTHON" "$CODE/run_m5.py" --root "$ROOT" --data "$DATA" --rmbench "$RMBENCH" --two-gpus
```

中断后用同一 `run_m5.py` 命令继续。训练恢复模型、优化器、学习率、随机数及分段状态，评测沿用已有正式回合。使用 `--publish` 可在闭环、种子门槛及报告全部完成后发布紧凑结果。大型数据、权重和视频在服务器保留。
