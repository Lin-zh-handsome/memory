# M5 空间视觉子目标辅助监督

独立实验从 bfaec49 派生，并保留旧 M5 的 512 维全局监督。空间版从头训练，U-Net（动作去噪网络）仍接收 256 维共享特征。只在训练中将共享特征送给辅助头，预测 3×3 个空间位置的 512 维特征。每个位置分别算余弦距离，并只平均事件前半段的有效样本。第 1 轮权重为 0，第 10 轮达到 0.02。

[关键帧审查](KEYFRAME_AUDIT.md) · [教师特征质量](FEATURE_QUALITY.md) · [模型检查](CHECKS.md) · [任务规划](task_plan.md)

## 复现

从本工作树根目录执行。沿用 M4 的 RMBench 环境、资产和仿真补丁。准备数据和官方示范归档与原 M4 一致。旧 ImageNet 权重仅用于离线特征对照。

```bash
CODE=implementation/rmbench
ROOT=/path/to/experiments/rmbench/m5_goal_spatial_r3m
RMBENCH=/path/to/RMBench
DATA=/path/to/cover_blocks/prepared
ARCHIVE=/path/to/cover_blocks/demo_clean.zip
DPSOURCE=/path/to/diffusion_policy
CUROBO=/path/to/curobo/src
PYTHON=/path/to/conda/bin/python
mkdir -p "$ROOT/teacher"
curl -fL 'https://hf-mirror.com/surajnair/r3m-18/resolve/1a4f077fe01db52c8f5d9f8d6641b6e03f688420/pytorch_model.bin' \
  -o "$ROOT/teacher/r3m18.bin"
curl -fL 'https://download.pytorch.org/models/resnet18-f37072fd.pth' \
  -o "$ROOT/teacher/resnet18-f37072fd.pth"
PYTHONPATH="$RMBENCH" "$PYTHON" "$CODE/spatial_goal_targets.py" \
  --archive "$ARCHIVE" --data "$DATA" --r3m-weights "$ROOT/teacher/r3m18.bin" \
  --global-weights "$ROOT/teacher/resnet18-f37072fd.pth" --output "$ROOT/annotations"
export PYTHONPATH="$DPSOURCE:$CUROBO"
"$PYTHON" "$CODE/test_models.py" --device cuda
"$PYTHON" "$CODE/run_m5.py" --root "$ROOT" --data "$DATA" --rmbench "$RMBENCH" \
  --spatial-r3m --stage-results
```

训练、评测与回放诊断合计默认使用一张物理卡。按同一命令可恢复，五个检查点各跑固定的 50 个环境种子。仅当种子 0 的最佳成功率达到 40% 才启动种子 1。闭环结果、训练表格和阶段统计全部出现后，生成正式报告和服务器提交。权重、冻结教师文件、派生特征数组及大型视频留在服务器。
