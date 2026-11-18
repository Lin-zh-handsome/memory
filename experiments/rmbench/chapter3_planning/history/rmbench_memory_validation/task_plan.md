# RMBench 两项扩展及原结果审计
1. 快照和 Put Back Block 真实性、公平性审计 [complete]
2. Swap Blocks 数据适配、两模型训练与十个闭环节点 [complete]
3. Put Back Block 连续 Mamba M2 种子 0 在 Swap Blocks 完成后单独续训与评测 [pending]
4. Swap Blocks 与方块归位 M0/M2/M3 对比结果 [pending]

先通过 Put Back Block 审计，Swap Blocks 完成后才启动 Cover Blocks。种子 0，M0/M3，每模型 250 轮，五节点各 50 回合。默认一张卡，必要时最多两张。
- 用户授权有空闲资源时四卡并行。训练维持 GPU0/7 各一组，评测额外允许空闲 GPU1/3。
- 最新资源限制为单张 GPU0，替代之前的四卡配置。
- 最新资源限制恢复为合计最多四张空闲 GPU。
- 最新资源限制为训练和评测合计最多两张空闲 GPU。
- 当前调度跳过 Cover Blocks。Swap Blocks 继续使用 GPU0/1；用户另授权 GPU3/7 并行执行方块归位 M2。
- 最新安排为 Swap Blocks 先跑完，再由 GPU0/1 接续 M2。GPU3/7 不再分配给该实验。
