# 论文第三章 · 进度和来源

## 2026-11-18 同步

- 核对了当前项目的 RMBench 实验对话、方案与写作对话、汇报对话，以及 Coffee 和较早几何方案记录。实验状态以服务器规划、报告和逐回合数据为准。这里汇总研究决策与进度，原对话各自保存。
- 修正三套滞后的服务器计划。方块覆盖/电池试插后续队列、M4 种子 0 实验 B、M4 种子 1 复现均已进入完成状态。诊断各自保留原始执行记录。
- 核实 GitHub 远端 codex/rmbench-memory-validation 当前为 66202a4，3.2 当前为 d4e1749。本轮只同步服务器文档。下次发布时补齐规划文件，在交接页给网页版直接入口。
- 原 M4 实验和诊断已完成。M5 另见下文。

## 服务器规划索引

以下从 rmbench_m0_m3 根目录读取，每项含 task_plan.md、findings.md、progress.md。诊断和复现实验另有各自 REPORT.md。

| 工作 | 规划目录 |
|---|---|
| 方块归位早期 M0/M3 | .planning/rmbench-put-back-block-m0-m3-seed0/ |
| 方块交换与扩展队列、M2 排程 | experiments/rmbench_memory_validation/ |
| 方块覆盖与电池试插 | experiments/rmbench_mn_validation/ |
| 方块覆盖训练与评测审计 | .planning/rmbench-cover-training-audit/ |
| M3 第 100 轮种子 100000 重放 | .planning/rmbench-cover-seed100000-replay/ |
| M4 实验 B，种子 0 | .planning/rmbench-m4/ |
| M4 首次揭盖诊断 | experiments/rmbench/m4_first_uncover_diagnosis/ |
| M4 目标形成与决策前干预 | experiments/rmbench/m4_prechoice_diagnosis/ |
| M4 七回合逐动作块复查 | experiments/rmbench/m4_memory_followthrough/ |
| M4 第 504 步动作条件替换 | experiments/rmbench/m4_step504_readout_swap/ |
| M4 双种子复现 | experiments/rmbench/m4_cover_blocks_seed1/ |
| 发布记录 | .planning/rmbench-github-publication/ |
| Coffee 历史线 | ../coffee_dp_official/.planning/coffee-preparation-d0-m0-m2/ |

## 网页版研判入口

先读本目录的 task_plan.md 和 findings.md，再读对应报告。RMBench 已发布的报告目录见 [实验分支](https://github.com/Lin-zh-handsome/memory/tree/codex/rmbench-memory-validation/experiments/rmbench)。原 M4 实验 B 正式结果在 mn_validation/m4_cover_blocks/，M3 单回合重放在 mn_validation/diagnostics/seed_100000/。首次揭盖时间线与历史替换使用后续独立诊断报告，可配合逐回合记录复核。

下次发布时，本目录及五个 M4 实验的三件套沿原路径加入 Git。对旧 .planning 和旧队列目录，以实验为单位放入本目录的 history/，继续保留任务计划、研究发现和进度三份文件，把记录中的过期安排明确写成历史状态。Coffee 仅在第三章交接中引用已发布的 3.2 分支，实验文件仍在该历史分支。

## M5 下一关键状态辅助监督

用户已批准 M5，保留原 M4 写入和动作协议，把过去动作重构改为预测下一关键帧视觉特征。独立规划见 [M5 进度](../m5_goal_keyframe_aux/progress.md)。当前已完成实现与训练前检查。
