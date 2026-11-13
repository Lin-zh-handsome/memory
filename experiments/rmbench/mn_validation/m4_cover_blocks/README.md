# M4 方块覆盖对照

M4 按当前观测和连续 Mamba 状态查询历史事件。训练队列于 11 月 12 日暂停，状态为等待空闲卡期间已停止。完成训练轮数为 0，正式评测回合为 0。

待评估协议使用 50 条官方示范、训练种子 0 和 M3 的相同环境种子。配置见 [protocol.json](configs/protocol.json)，固定环境初态见 [reset_seeds.json](evaluation/reset_seeds.json)。[实现](../../../../implementation/rmbench/models.py) · [调度入口](../../../../implementation/rmbench/run_m4.py)
