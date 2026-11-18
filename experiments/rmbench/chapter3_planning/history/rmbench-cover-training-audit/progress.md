# Progress
已暂停电池试插。
Training logs show finite, steadily decreasing loss and 204,500 updates per model. The prepared format uses official XPolicyLab state/action arrays, head-camera RGB and the next-state action alignment check. Inspect all 50 color layouts and evaluate controller fallback distribution.
Started expert action replay with current Cover controller for three natural starts (one red position each). Battery queue remains paused.
Finished the audit and stored the three controller replays in expert_qpos_replay.json.


## 2026-11-12 复核

读取实际配置、训练 CSV、四个完整检查点的逐回合数据、正式评测源码及已存专家重放。合并训练与评测两项独立复核。
