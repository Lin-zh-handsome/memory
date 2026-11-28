# 进度

已读取上一轮规划、调度器停止机制和当前GPU进程；开始保存断点和切换单卡。

## 2026-11-28T11:26:40.380978+08:00 单卡启动

[{"task": "swap_blocks", "resume_saved": true, "next_epoch": 185, "batch": 118, "global_step": 92118, "checkpoint_seconds": 6.636988709680736}, {"task": "battery_try", "resume_saved": true, "next_epoch": 165, "batch": 225, "global_step": 90917, "checkpoint_seconds": 10.13133220281452}, {"task": "blocks_ranking_try", "resume_saved": true, "next_epoch": 33, "batch": 714, "global_step": 37994, "checkpoint_seconds": 10.186827630735934}, {"task": "press_button", "resume_saved": true, "next_epoch": 90, "batch": 418, "global_step": 38688, "checkpoint_seconds": 10.391824984923005}]
监督PID：1235402。

## 2026-11-28 16:35 单卡进度与方块重排完成

按最新指令已于11:26保存四项训练断点并切为仅GPU 5，每卡4槽；当前方块交换、电池、排序三项训练和按钮50轮评测，GPU 6无本批进程。按钮训练保存于第90轮第418步global_step38688，排队继续。无新截止时间。

九任务5/9已全部完成，31/45节点评完。方块重排250轮训练与五节点均完整，成功率32%、56%、100%、100%、100%，同分取较早为150轮100%。方块交换已完成203轮，完整节点当前最佳100%；电池181轮，当前最佳92%；排序40轮暂无评测；按钮89轮，50轮已评16/50、成功0，仍未完成。

[九任务状态](../m5_seed0_rgb_corrected/STATUS.md) · [方块重排完整结果](../complete_rearrange_blocks_20261128/REPORT.md)。
