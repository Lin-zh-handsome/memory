# Progress
Located the original and extension source branches plus the benchmark worktree.
Inventoried code, configurations, reports and the 1,250 rollouts.
Staged release content in a detached server-side worktree. Included configurations, epoch metrics and all 25 per-checkpoint filesets.
Git flagged CRLF CSV lines during staging; normalized only the publication copies. Committed source lineage and 1,250 evaluation records. No datasets, weights or videos in the release delta.
Published and confirmed Lin-zh-handsome/memory branch codex/rmbench-memory-validation at 60d0eec.
Committed a point-in-time M(n) result snapshot, training statistics and seed 100000 diagnostic metrics. No data, checkpoints, videos or raw feature tensors in the publication.
Direct GitHub push stalled; existing HTTP proxy at 127.0.0.1:7892 returned HTTP 200 and advertised the previous branch head. Retry with scoped proxy and timeout.
Verified M(n) snapshot at 495576b on refs/heads/codex/rmbench-memory-validation.
Reviewed six Codex project chats and fourteen linked web project chats. Reconciled completed Battery, stopped Cover, paused untrained M4 and stopped historical Coffee against server artifacts.
Test module uses sibling imports; set its directory on PYTHONPATH for the targeted CPU check.
Committed 958 Cover and Battery evaluation records, paired statistics, issues, M4 code and a planning prompt. Combined source passes CPU attention test.
GitHub accepted and advertised 845cf3a on the existing branch. HANDOFF.md is linked from the repository and RMBench READMEs.

2026-11-13 汇报发布：已确认 GitHub 现有结果分支和服务端发布工作树。
已核对服务器与 GitHub 的 45 个检查点种子及成功记录，总计 2,208 回合。已汇总进 PROGRESS.md，修正两个索引的历史状态。
服务端 HTTPS 推送缺少 GitHub 凭据；切换 Windows 已配置的 Git 凭据。
GitHub 已接受 49b8e22，远端 codex/rmbench-memory-validation 已确认。仅传输四个 Markdown 文件对应的 Git 提交。

11 月 15 日任务为发布当前 M4 结果，GitHub 仍在 49b8e22。服务器原训练产物保持在原实验目录。
已整理 250 回合评测结果、阶段与配对明细、留出读出和训练轮次统计。原始视频、检查点及特征张量保留服务器。
暂存清单的初始预计文件数少算一份，已逐项确认是预期的结果文件与汇报索引。
GitHub 远端分支 codex/rmbench-memory-validation 已确认指向 5a949d3。发布五个检查点、250 回合、阶段配对和留出读出，标明历史替换尚未完成。

11 月 16 日确认远端 codex/rmbench-memory-validation 指向 7614bf3。正式发布 M4 首次揭盖报告、50 回合时刻、留出读出、注意力权重汇总、动作反事实和实际换盖明细。AGENTS.md 已加入每次完整训练评测或诊断完成后推送一次的约定。

11月16日 M4 目标形成前诊断已发布，远端分支确认为 5f3c054。

七个 M4 失败回合逐动作块复查已发布，远端 codex/rmbench-memory-validation 为 b4297b3。

11月16日 M4 第504步动作头记忆替换报告已推送，远端确认为 a9ae1f9。
- 2026-11-18 用户要求下次 GitHub 发布同步 planning-with-files 的 task_plan.md、findings.md、progress.md；本轮服务器汇总和待发布范围见 experiments/rmbench/chapter3_planning/。
