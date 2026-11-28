# 进度

旧队列已保存断点，新队列开始 CPU 阈值校准。

2026-11-26T20:23:42+08:00 按用户要求使用修正 RGB 从头训练剩余六项，GPU 5 设置四个并行槽。

## 2026-11-26 记忆写入阈值文献核对

逐任务训练集统计校准可以作为实现设定。初始化M3到已训练M5的评分一致性、特征分布变化及相邻帧/最近入库事件比较差异，需补机制证据。20%分位数不代表20%写入率。当前训练、阈值和官方规则均未调整。
文献与验证建议：[调研报告](../memory_write_threshold_literature_20261126/REPORT.md)。

## 2026-11-26 双卡运行至11月27日07:30前

使用5号、6号H200，每卡4个任务槽；六项M5种子0均已启动。原四项保存断点并精确恢复，两项新训练已有有效日志。第50轮权重可用后空闲槽加入评测。服务器用户cron已设置北京时间07:25保存停止、07:29清理，仅匹配本批进程，独立于本地联网状态。
[运行与定时说明](../dual_gpu_until_20261127_0730/REPORT.md)。

## 定时停止 2026-11-27T07:29:01.528829+08:00

{"status": "paused", "reason": "scheduled_0730_beijing", "time": "2026-11-27T07:29:01.528829+08:00", "target_time": "2026-11-27T07:30:00+08:00", "forced": false, "remaining": [], "checkpoints": [{"root": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/swap_blocks/seed0", "checkpoint": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/swap_blocks/seed0/checkpoints/M5/latest.pt", "last_resume_record": {"resume_saved": true, "next_epoch": 141, "batch": 419, "global_step": 70419, "checkpoint_seconds": 14.563732109032571}}, {"root": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/battery_try/seed0", "checkpoint": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/battery_try/seed0/checkpoints/M5/latest.pt", "last_resume_record": {"resume_saved": true, "next_epoch": 126, "batch": 369, "global_step": 69494, "checkpoint_seconds": 14.423877286724746}}, {"root": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/rearrange_blocks/seed0", "checkpoint": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/rearrange_blocks/seed0/checkpoints/M5/latest.pt", "last_resume_record": {"resume_saved": true, "next_epoch": 210, "batch": 4, "global_step": 73154, "checkpoint_seconds": 14.044576055370271}}, {"root": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/swap_T/seed0", "checkpoint": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/swap_T/seed0/checkpoints/M5/latest.pt", "last_resume_record": {"resume_saved": true, "next_epoch": 245, "batch": 265, "global_step": 73465, "checkpoint_seconds": 14.301723374053836}}, {"root": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/blocks_ranking_try/seed0", "checkpoint": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/blocks_ranking_try/seed0/checkpoints/M5/latest.pt", "last_resume_record": {"resume_saved": true, "next_epoch": 21, "batch": 818, "global_step": 24118, "checkpoint_seconds": 12.833711391314864}}, {"root": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/press_button/seed0", "checkpoint": "/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench/m5_seed0_rgb_corrected/press_button/seed0/checkpoints/M5/latest.pt", "last_resume_record": {"resume_saved": true, "next_epoch": 57, "batch": 249, "global_step": 24329, "checkpoint_seconds": 15.68125313334167}}]}

## 2026-11-27 07:30前运行窗口结束

07:25保存并正常停止，07:29检查无残留，未强杀；六项轮内断点均存在，5号和6号卡已释放。修正版方块交换第50轮完整48/50（96%），第100轮已评18/50、成功17；电池第50轮已评12/50、成功3；T形块交换第50轮完整2/50（4%）。完整训练轮次依次为方块交换140、电池125、重排209、T形块244、排序20、按钮56。训练与评测保持暂停。
最新汇总：[状态](../m5_seed0_rgb_corrected/STATUS.md)。

## 2026-11-27 单卡优先完成T形块交换

按最新指令只运行修正RGB的M5种子0 T形块交换：GPU 5，1训练+3评测，训练完成后4评测。已从第245轮中途global_step73465恢复；第100、150、200轮评测已进入闭环，第50轮2/50完整结果保留。目标为250轮训练及五节点各50回合全完成。其他任务暂停。此前07:30截止已执行，本轮不设置新的截止时间。
[单任务运行记录](../focus_swap_T_20261127/REPORT.md)。

## 2026-11-27 17:20 T形块交换进度

250轮训练已完成。第50轮2/50（4%）、第100轮11/50（22%）均完整；第150轮已评46/50、成功21，第200轮已评44/50、成功33，第250轮已评28/50、成功26，三节点仍在进行。总计已评218/250回合。单卡GPU 5上四评测进程继续运行，其他任务暂停。

## T形块交换单任务 completed

{"time": "2026-11-27T18:09:19.378908+08:00", "task": "swap_T", "model": "M5", "seed": 0, "rgb_normalization": "identity", "training_complete": true, "evaluations": [{"epoch": 50, "episodes": 50, "successes": 2, "complete": true, "success_rate": 0.04}, {"epoch": 100, "episodes": 50, "successes": 11, "complete": true, "success_rate": 0.22}, {"epoch": 150, "episodes": 50, "successes": 23, "complete": true, "success_rate": 0.46}, {"epoch": 200, "episodes": 50, "successes": 38, "complete": true, "success_rate": 0.76}, {"epoch": 250, "episodes": 50, "successes": 45, "complete": true, "success_rate": 0.9}], "complete": true}

## 2026-11-27 18:09 T形块交换全部完成

修正RGB版本M5种子0已完成250轮训练和五节点各50回合评测：50轮2/50（4%），100轮11/50（22%），150轮23/50（46%），200轮38/50（76%），250轮45/50（90%）。最佳为第250轮90%。调度器和任务进程已正常退出，GPU 5已释放，其他任务继续暂停。
结果入口：[T形块最终结果](../focus_swap_T_20261127/REPORT.md)。

## 2026-11-27T21:29:54.186402+08:00 双卡每卡四进程继续M5种子0

按最新指令使用GPU 5、6，每卡4槽，当前5训练+3评测，从原断点继续其余五项。T形块90%完整结果保留，其他已完成结果保留；模型、训练预算、RGB修正配置、阈值与官方规则不变。训练结束后空闲槽自动执行评测。未设置新的截止时间，任务独立于本地连接。调度PID 634460，监督PID 634399。
运行记录：[双卡恢复](../resume_m5_dual_gpu_20261127/REPORT.md)。

## 2026-11-28 16:35 单卡进度与方块重排完成

按最新指令已于11:26保存四项训练断点并切为仅GPU 5，每卡4槽；当前方块交换、电池、排序三项训练和按钮50轮评测，GPU 6无本批进程。按钮训练保存于第90轮第418步global_step38688，排队继续。无新截止时间。

九任务5/9已全部完成，31/45节点评完。方块重排250轮训练与五节点均完整，成功率32%、56%、100%、100%、100%，同分取较早为150轮100%。方块交换已完成203轮，完整节点当前最佳100%；电池181轮，当前最佳92%；排序40轮暂无评测；按钮89轮，50轮已评16/50、成功0，仍未完成。

[九任务状态](../m5_seed0_rgb_corrected/STATUS.md) · [方块重排完整结果](../complete_rearrange_blocks_20261128/REPORT.md)。
