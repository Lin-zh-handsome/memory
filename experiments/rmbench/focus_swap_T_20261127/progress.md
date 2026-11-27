# 进度

已检查空闲GPU、断点、完整50轮结果和当前暂停状态；开始独立单任务队列，原队列保持暂停。

已确认GPU 5四个有效GPU进程，利用率100%。训练从global_step73465恢复并完成第245轮；第100、150、200轮评测均已进入真实闭环步骤。调度PID：161605。

## 2026-11-27 17:20 T形块交换进度

250轮训练已完成。第50轮2/50（4%）、第100轮11/50（22%）均完整；第150轮已评46/50、成功21，第200轮已评44/50、成功33，第250轮已评28/50、成功26，三节点仍在进行。总计已评218/250回合。单卡GPU 5上四评测进程继续运行，其他任务暂停。

## 2026-11-27T18:09:19.379870+08:00 completed

{"time": "2026-11-27T18:09:19.378908+08:00", "task": "swap_T", "model": "M5", "seed": 0, "rgb_normalization": "identity", "training_complete": true, "evaluations": [{"epoch": 50, "episodes": 50, "successes": 2, "complete": true, "success_rate": 0.04}, {"epoch": 100, "episodes": 50, "successes": 11, "complete": true, "success_rate": 0.22}, {"epoch": 150, "episodes": 50, "successes": 23, "complete": true, "success_rate": 0.46}, {"epoch": 200, "episodes": 50, "successes": 38, "complete": true, "success_rate": 0.76}, {"epoch": 250, "episodes": 50, "successes": 45, "complete": true, "success_rate": 0.9}], "complete": true}

## 2026-11-27 18:09 T形块交换全部完成

修正RGB版本M5种子0已完成250轮训练和五节点各50回合评测：50轮2/50（4%），100轮11/50（22%），150轮23/50（46%），200轮38/50（76%），250轮45/50（90%）。最佳为第250轮90%。调度器和任务进程已正常退出，GPU 5已释放，其他任务继续暂停。
结果入口：[T形块最终结果](../focus_swap_T_20261127/REPORT.md)。
