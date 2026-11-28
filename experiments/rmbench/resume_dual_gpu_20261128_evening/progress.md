# 进度

已读取单卡规划及切换脚本，准备保存当前训练断点后恢复双卡。

## 2026-11-28T21:29:39.468880+08:00 双卡启动

[{"task": "swap_blocks", "resume_saved": true, "next_epoch": 219, "batch": 439, "global_step": 109439, "checkpoint_seconds": 7.621903425082564}, {"task": "battery_try", "resume_saved": true, "next_epoch": 196, "batch": 220, "global_step": 108055, "checkpoint_seconds": 9.416245584376156}, {"task": "blocks_ranking_try", "resume_saved": true, "next_epoch": 47, "batch": 932, "global_step": 54522, "checkpoint_seconds": 9.345158068463206}]
监督PID：1457813。

## 2026-11-28T21:31:42.238226+08:00 恢复双卡每卡四槽

最新指令为两张卡、每卡四个并行槽。已保存原单卡三项训练断点，并于21:29在GPU 5/6恢复。当前4训练+3评测：GPU 5有方块交换、电池训练以及按钮50轮和交换200轮两项评测；GPU 6有排序、按钮训练以及交换200轮后段评测。当前可运行工作7项，下一批权重生成后自动补第8槽。

四项训练均已确认恢复，三项评测均进入真实闭环，21:30两卡计算利用率均100%。调度PID 1457873，监督PID 1457813。断点、结果保留，模型、阈值、预算、初态及评测协议不变。无新增截止时间，本地断网继续。
[本轮双卡记录](../resume_dual_gpu_20261128_evening/REPORT.md)。
