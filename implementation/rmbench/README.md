# RMBench Put Back Block 回放验证

使用同一批 50 条官方 demo_clean 专家示范，比较原始 Stanford CNN-DP（卷积扩散策略）M0 和连续 Mamba（状态空间序列模型）加情景记忆的 M3。M2 共享连续递推和过去动作辅助损失，支持独立训练与推理，本轮只验证其实现。

## 复现

服务器使用 `coffee-dp` Conda（环境管理器）环境。Stanford Diffusion Policy（扩散策略）源码为 5ba07ac，RMBench（记忆操作基准）源码为 7d1cdfc，过去动作辅助监督沿用 memory 仓库 3.2 的 d4e1749。

下载 [官方示范归档](https://huggingface.co/datasets/TianxingChen/RMBench/resolve/main/data/put_back_block/demo_clean.zip) 到 `data/put_back_block_demo_clean.zip`。RMBench 的 `assets/embodiments` 和 `assets/objects` 必须包含官方 Aloha AgileX 机器人、005_button 和对象元数据。原数据包含头部与腕部视角。本实验对所有模型使用同一 96×96 头部 RGB（彩色）图像和同一 14 维绝对关节及夹爪状态。数据清单记录原始相机、动作顺序及回合长度。

```bash
BASE="$HOME/project/rmbench_m0_m3"
DP="$HOME/project/coffee_dp_official/source/diffusion_policy"
CUROBO="$HOME/project/benchmark/RoboTwin/envs/curobo/src"
BENCH="$BASE/source/RMBench"
EXP="$BASE/experiments/rmbench_put_back_block_m0_m3_seed0"
PY="$HOME/miniconda3/envs/coffee-dp/bin/python"
CODE="$BASE/source/memory-rmbench/implementation/rmbench"
export PYTHONNOUSERSITE=1 PYTHONPATH="$DP:$CUROBO"
export GPU_UUID=GPU-66d3f503-bf4f-56dd-bb79-74ca7cda66e5
export CUDA_VISIBLE_DEVICES="$GPU_UUID"

# 只对未应用补丁的干净 RMBench 检出执行。
git -C "$BENCH" apply "$CODE/rmbench_render_device.patch"
mkdir -p "$EXP/configs"
cp -n "$CODE/configs/"*.json "$EXP/configs/"
"$PY" "$CODE/data.py" "$BASE/data/put_back_block_demo_clean.zip" "$BASE/data/prepared" "$BENCH"
"$PY" "$CODE/test_models.py"
"$PY" "$CODE/calibrate.py" --root "$EXP" --data "$BASE/data/prepared"
"$PY" "$CODE/evaluate.py" --experiment "$EXP" --rmbench "$BENCH" --data "$BASE/data/prepared" \
  --render-pci 00000000:DB:00.0 --camera-shader rt verify
"$PY" "$CODE/campaign.py" --root "$EXP" --data "$BASE/data/prepared" \
  --rmbench "$BENCH" --primary "$GPU_UUID"
```

M0 和 M3 每轮各使用同样的 17,212 个监督样本和 300 次优化更新。Mamba 的卷积缓存和 SSM（状态空间模型）状态跨 64 步截断反向传播边界保留，回合切换时重置。每次在线观测只更新一次状态，M3 先读出保序记忆，再写入当前候选事件。

评测使用第 50、100、150、200、250 轮的 EMA（指数移动平均）权重。所有节点共享同一组 50 个通过官方专家检查的初始条件，使用 500 步上限和官方成功判定。仅在五个节点完整评测后汇报该模型的最佳成绩。

支持在限时资源调度中并行训练和筛选初始条件。辅助卡上的训练保存优化器、随机状态和采样位置后迁移到主卡。候选分片只向固定种子清单提交连续且经过专家检查的前缀。正式评测和 Vulkan（图形接口）渲染始终绑定到主卡。
