# RMBench 消融运行快照

截至北京时间 2026-11-25，九任务、四模型、三个训练种子共 108 组，已有 **7 组**完成全部五个闭环评测节点。M5 完成 **2/27 组**，分别是方块覆盖和方块归位的训练种子 0。

本轮于北京时间 **07:28:42** 正常停止。恢复时从已有训练断点和已记录评测回合接续，优先补齐 M5 的其余 **25 组**，随后继续原始扩散策略、纯 Mamba 和八槽情景记忆。

## 闭环评测节点

均为训练种子 0。同一任务各组使用同一批 50 个固定初态，每组评完五个节点后按最佳值计入正式总表。未评满节点记录已完成回合及成功数。

| 任务 | 模型 | 第 50 轮 | 第 100 轮 | 第 150 轮 | 第 200 轮 | 第 250 轮 | 完整节点 |
|---|---|---|---|---|---|---|---|
| 方块覆盖 | 纯 Mamba | [5/50（10%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/mamba_only/epoch_050/summary.json) | [10/50（20%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/mamba_only/epoch_100/summary.json) | [7/50（14%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/mamba_only/epoch_150/summary.json) | [6/50（12%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/mamba_only/epoch_200/summary.json) | [6/50（12%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/mamba_only/epoch_250/summary.json) | 5/5 |
| 方块覆盖 | 八槽情景记忆 | [7/50（14%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/memory_only/epoch_050/summary.json) | [9/50（18%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/memory_only/epoch_100/summary.json) | [8/50（16%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/memory_only/epoch_150/summary.json) | 待评 | 待评 | 3/5 |
| 方块归位 | 完整 M5 | [50/50（100%）](checkpoint_snapshots/put_back_block/seed0/evaluation/M5/epoch_050/summary.json) | [50/50（100%）](checkpoint_snapshots/put_back_block/seed0/evaluation/M5/epoch_100/summary.json) | [50/50（100%）](checkpoint_snapshots/put_back_block/seed0/evaluation/M5/epoch_150/summary.json) | [50/50（100%）](checkpoint_snapshots/put_back_block/seed0/evaluation/M5/epoch_200/summary.json) | [50/50（100%）](checkpoint_snapshots/put_back_block/seed0/evaluation/M5/epoch_250/summary.json) | 5/5 |
| 方块交换 | 完整 M5 | [48/50（96%）](checkpoint_snapshots/swap_blocks/seed0/evaluation/M5/epoch_050/summary.json) | 待评 | 待评 | 待评 | 待评 | 1/5 |
| 电池试插 | 完整 M5 | [已评 38/50，成功 14](checkpoint_snapshots/battery_try/seed0/evaluation/M5/epoch_050/summary.json) | 待评 | 待评 | 待评 | 待评 | 0/5 |
| 观察后抓取 | 原始扩散策略 | [已评 48/50，成功 1](checkpoint_snapshots/observe_and_pickup/seed0/evaluation/M0/epoch_050/summary.json) | [已评 10/50，成功 0](checkpoint_snapshots/observe_and_pickup/seed0/evaluation/M0/epoch_100/summary.json) | 待评 | 待评 | 待评 | 0/5 |

## 已保存训练断点

| 任务 | 模型 | 轮次 | 批次 | 累计训练步数 |
|---|---|---:|---:|---:|
| 方块交换 | 完整 M5 | 79 | 462 | 39,462 |
| 电池试插 | 完整 M5 | 61 | 29 | 33,209 |
| 观察后抓取 | 完整 M5 | 81 | 20 | 12,020 |
| 方块重排 | 原始扩散策略 | 174 | 23 | 60,573 |

[正式结果总表](REPORT.md) · [实验计划](task_plan.md) · [执行进度](progress.md)
