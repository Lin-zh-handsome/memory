# RMBench 连续 Mamba 实验

[实验结果](../../experiments/rmbench/README.md)涵盖方块归位（M0、M2、M3）和方块交换（M0、M3）。M0 是 Stanford CNN-DP（卷积扩散策略），M2 使用连续 Mamba（状态空间序列模型）和过去动作重构，M3 额外使用选择性情景记忆。

## 环境与数据

- 从 [RMBench 官方仓库](https://github.com/RoboTwin-Platform/RMBench)检出 `7d1cdfc`，按官方说明准备 SAPIEN（物理仿真器）、CuRobo（机器人运动规划器）和场景资产。
- 从 [Stanford Diffusion Policy](https://github.com/real-stanford/diffusion_policy)检出 `5ba07ac`，安装其依赖及 `mamba-ssm`，并将相关源码加入 `PYTHONPATH`。方块交换的实际依赖版本见[来源配置](../../experiments/rmbench/swap_blocks/configs/source.json)。
- 从 [RMBench 官方数据](https://huggingface.co/datasets/TianxingChen/RMBench)获取对应任务的 `demo_clean.zip`。所有模型使用相同的 96×96 头部彩色图像及 14 维绝对关节和夹爪状态。

## 训练与评测

以下示例在本仓库根目录运行。设置 `BENCH` 为 RMBench 检出路径、`ARCHIVE` 为官方示范压缩包、`RUN` 为新的运行目录、`RENDER_PCI` 为评测显卡的 PCI 地址。按需设置 `CUDA_VISIBLE_DEVICES`。

```bash
CODE="$PWD/implementation/rmbench"
TASK=swap_blocks
RESULT="$PWD/experiments/rmbench/swap_blocks"
MODEL=M3
PREPARED="$RUN/prepared"

# 仅在干净的 RMBench 检出中应用补丁。
git -C "$BENCH" apply "$CODE/rmbench_render_device.patch"
git -C "$BENCH" apply "$CODE/rmbench_qpos_workers.patch"

mkdir -p "$RUN/configs" "$RUN/evaluation"
cp "$RESULT/configs/protocol.json" "$RUN/configs/"
cp "$RESULT/evaluation/reset_seeds.json" "$RUN/evaluation/"
python "$CODE/data.py" "$ARCHIVE" "$PREPARED" "$BENCH" --task "$TASK"
python "$CODE/train.py" --root "$RUN" --data "$PREPARED" --model "$MODEL"

for epoch in 50 100 150 200 250; do
  python "$CODE/evaluate.py" --experiment "$RUN" --rmbench "$BENCH" \
    --data "$PREPARED" --render-pci "$RENDER_PCI" --camera-shader rt \
    rollout --model "$MODEL" --epoch "$epoch" --batch-size 2
done
```

方块归位使用 `TASK=put_back_block`，并将 `RESULT` 指向 `experiments/rmbench/put_back_block_m0_m3` 或 `experiments/rmbench/put_back_block_m2`。逐节点的实际批大小、种子顺序和随机数规则见随结果发布的 `evaluation_protocol.json`。原始方块归位 M0/M3 采用其 `configs/source.json` 中记录的历史训练和评测提交及渲染补丁。方块交换和 M2 的关节空间评测使用 CPU TOPP（时间最优路径参数化）和原始夹爪插值。

评测使用第 50、100、150、200、250 轮的 EMA（指数移动平均）权重。五个节点复用该任务的 50 个初始条件，方块归位每回合最多 500 步，方块交换最多 1,000 步。
