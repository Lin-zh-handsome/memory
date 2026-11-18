# Findings
Both experiment worktrees use Lin-zh-handsome/memory as origin.
Publish on codex/rmbench-memory-validation; no matching branch currently advertised.
All 25 evaluated nodes have a summary, protocol and per-episode records.
Benchmark worktree has three modified files; check what must accompany source.
The two benchmark patches match the run environment. rg is absent on the server; used grep.
M(n) source 50acf20 descends from published 60d0eec; update existing codex/rmbench-memory-validation without touching the active worktree.

GitHub 汇报分支已包含 845cf3a 的四任务结果、诊断、M4 源码和停训状态。RMBench 旧索引仍称方块覆盖已跳过，根 README 的咖啡实验状态也已过时。

11 月 15 日 M4 完成 250 轮训练和 250 回合正式评测。回合留出读出已落盘，因果干预在末端运动学一致性检查中失败。

11 月 16 日诊断完成，异位置干预 3/4 换盖、2/4 正确换盖。用户要求完整训练评测或诊断完成后逐次发布。
