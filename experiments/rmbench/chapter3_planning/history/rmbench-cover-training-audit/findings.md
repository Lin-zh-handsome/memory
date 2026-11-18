# Findings
All 50 first-frame RGB layouts matched annotation and all converted actions exactly matched the next state. Six permutations appear 3–11 times. The M3 four completed checkpoints produced 166 invalid-order failures and three uncover timeouts.
The official XPolicyLab runner also sends joint outputs to the same take_action interface. Our batch and head-only inputs differ from the Mem-0 paper, which uses high-level planning plus subtask-end detection. The live robot patch bypasses unused pose planning, while TOPP and the success predicate remain on the original path.
Expert-action replay in the current evaluation controller: 3/3 complete successes, seeds 0, 1 and 5; one per first target position. The initial joint vectors matched exactly. TOPP fallback counters were nonzero even for the successful expert trajectories.


## 2026-11-12 复核

M0/M3 使用同一批 50 条官方示范，均训练 250 轮。现有数据和检查点加载未发现错位，50 个环境种子的重复评测初态一致。已有同控制链专家回放 3/3 成功。M3 前四个完整检查点 200/200 进入揭盖阶段，124/200 首揭顺序错误，75/200 首揭正确，31/200 整回合成功。种子 100000 的重放对齐，首帧颜色位置可读，置换首帧位置未改变首揭目标。
训练状态跨段传递但每 64 帧截断梯度，示范首次揭盖在约第 510 帧，首揭损失无法回传到首次写入 E0 的计算。事件阈值用未训练模型的特征校准，训练后各完整检查点的事件写入率中位数约 72%–78%。尚未隔离两处因素对成功率的影响。无独立离线验证集。
