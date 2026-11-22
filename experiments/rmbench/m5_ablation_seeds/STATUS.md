# 全量消融发布快照

核对时间为 2026-11-22 21:57 CST（北京时间）。调度器按用户要求于 2026-11-22 00:29 暂停训练和评测。

九任务、四模型、三个训练种子共 108 组。现有 5 组完成五个检查点，均为复用结果。新增评测已跑满 6 个检查点，分布在下表 3 组。

## 新增评测与断点

下表四组均使用训练种子 0。跑满的检查点报告成功回合数和成功率，未跑满的检查点报告已记录回合数。

| 任务 | 模型 | 训练进度 | 50 轮 | 100 轮 | 150 轮 | 200 轮 | 250 轮 | 完整节点 |
|---|---|---|---|---|---|---|---|---|
| 方块覆盖 | 纯 Mamba | 已完成 250 轮 | [5/50（10%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/mamba_only/epoch_050/summary.json) | [10/50（20%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/mamba_only/epoch_100/summary.json) | [2/50 回合](checkpoint_snapshots/cover_blocks/seed0/evaluation/mamba_only/epoch_150/summary.json) | [2/50 回合](checkpoint_snapshots/cover_blocks/seed0/evaluation/mamba_only/epoch_200/summary.json) | [2/50 回合](checkpoint_snapshots/cover_blocks/seed0/evaluation/mamba_only/epoch_250/summary.json) | 2/5 |
| 方块覆盖 | 八槽情景记忆 | 第 244 轮第 650 批暂停 | [7/50（14%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/memory_only/epoch_050/summary.json) | [9/50（18%）](checkpoint_snapshots/cover_blocks/seed0/evaluation/memory_only/epoch_100/summary.json) | [40/50 回合](checkpoint_snapshots/cover_blocks/seed0/evaluation/memory_only/epoch_150/summary.json) | 待评 | 待评 | 2/5 |
| 方块归位 | 完整 M5 | 第 249 轮第 294 批暂停 | [50/50（100%）](checkpoint_snapshots/put_back_block/seed0/evaluation/M5/epoch_050/summary.json) | [50/50（100%）](checkpoint_snapshots/put_back_block/seed0/evaluation/M5/epoch_100/summary.json) | [46/50 回合](checkpoint_snapshots/put_back_block/seed0/evaluation/M5/epoch_150/summary.json) | [8/50 回合](checkpoint_snapshots/put_back_block/seed0/evaluation/M5/epoch_200/summary.json) | 待评 | 2/5 |
| 方块交换 | 完整 M5 | 第 2 轮第 4 批暂停 | 待评 | 待评 | 待评 | 待评 | 待评 | 0/5 |

## 已完成的五组

| 任务 | 模型 | 训练种子 | 五节点最佳 |
|---|---|---|---|
| 方块归位 | M0 原始扩散策略 | 0 | 26%（250 轮） |
| 方块交换 | M0 原始扩散策略 | 0 | 18%（50 轮） |
| 电池试插 | M0 原始扩散策略 | 0 | 86%（250 轮） |
| 方块覆盖 | M0 原始扩散策略 | 0 | 16%（100 轮） |
| 方块覆盖 | 完整 M5 | 0 | 80%（150 轮） |

完整组按五个检查点各 50 回合取最佳，同分取较早轮次。新增组待五个检查点全部完成后进入最佳成功率汇总。

[正式结果总表](REPORT.md) · [评测协议](task_plan.md) · [执行进度](progress.md)
