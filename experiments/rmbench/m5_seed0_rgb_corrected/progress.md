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
