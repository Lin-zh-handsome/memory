# 第三章当前模型实现审查

审查日期：2026-11-28。只读实际模型、配置和现有单回合日志；唯一写入为本审查文件。未改实验代码/配置，未启动 GPU（图形处理器），未停止作业，未下载或复制实验文件到本地。

路径约定：
- `SRC` = `/home/linzhenghong/project/rmbench_m0_m3/source/m5-goal-spatial-r3m`。
- 下文 `models.py` 等均位于 `SRC/implementation/rmbench/`。
- `DP` = `/home/linzhenghong/project/coffee_dp_official/source/diffusion_policy/diffusion_policy`。
- `EXP` = `/home/linzhenghong/project/rmbench_m0_m3/experiments/rmbench`。

## 结论

**在模型前向、缓存和辅助标签路径中，没有发现未来完成图注入评测、跨回合共享记忆、动作块内漏更新历史等可直接造成虚高成绩的错误。发现一个已确认的历史 RGB 预处理错接，以及当前新旧结果的协议不一致；这些会混淆模型增益归因。**

现有高分本身不能证明模型或评测正确，也不能证明存在作弊。成功判定、官方重置/控制器偏差由评测审查分工覆盖。本文件只对已覆盖的模型与调用路径作判断。

## 1. 已确认：旧 RGB 两次改变输入分布；当前只修正部分 M5

调用路径：`train.py:33-38` / `evaluate.py:230-236` 将 uint8 RGB 除以 255 → `data.py:235-249` 默认 range（范围变换）归一化 → `models.py:33-40` 启用 ImageNet（图像预训练基准）归一化 → `DP/model/vision/multi_image_obs_encoder.py:22-24,104-110`。

- `DP/common/normalize_util.py:23-35` 的 image range normalizer 为 `2*x-1`。
- 观测编码器明确假定 RGB 为 `[0,1]`，而旧配置 `imagenet_norm=true`、未写 `rgb_normalization`，所以送入主干的是 `((2*x-1)-mean)/std`，与编码器预期的 `(x-mean)/std` 不同。该错接发生在训练和在线评测两端，不属于未来信息泄漏；网络从头训练，实际影响方向和大小需要同协议对照，不能仅从公式推断。
- 六个修正任务（swap_blocks、battery_try、rearrange_blocks、swap_T、blocks_ranking_try、press_button）的 `configs/protocol.json` 与 `configs/M5.json` 已逐字段核对，共有协议字段一致，均为 `rgb_normalization=identity`、`imagenet_norm=true`。因此当前六任务 M5 使用正确的 `[0,1]`→ImageNet 路径。
- 证据例：`EXP/m5_seed0_rgb_corrected/swap_T/seed0/configs/protocol.json:42-47,86-88,120-124`；旧 `EXP/m5_ablation_seeds/swap_T/seed0/configs/protocol.json:42-46,85-89,123-125` 未指定 identity。
- `prepare_m5_rgb_corrected.py:43-50,69-74` 明确只给未完成 M5 更换 RGB 并重校准阈值；`EXP/m5_seed0_rgb_corrected/result_sources.json` 继续复用 put_back_block、cover_blocks、observe_and_pickup 的旧 range 结果。旧消融根现有 M0/mamba_only/memory_only 配置也保留 range；多数任务的其他消融尚无实际模型配置，不能把计划列表当成已训练结果。

影响：修正 M5 与旧 range 基线比较，同时改变了 RGB 输入、记忆阈值和模型方法，不能把增益全部归给 M5 辅助监督；跨任务汇总也混合了两种 RGB 协议。修正六任务 M5 内部训练/评测一致，没有“训练一种 RGB、评测另一种 RGB”的当前错接。

建议：结果表明确标明 RGB 协议；需要四组严格消融时使用一致的 identity RGB 和相同协议。无需因此停止当前训练，也不应在评测时给旧检查点直接切换 normalizer（归一化器）。

## 2. 结构核实：递进定义成立，但 memory_only 同时加了融合网络

`models.py:30-62` 是实际入口；当前 RMBench 模型不调用 `SRC/implementation/coffee_models.py` 的有限 16 帧窗口编码，而是独立实现连续历史，并复用 Stanford DP（扩散策略）的编码器、Unet（U形去噪网络）和采样器。

| 组别 | 实际动作条件 | 辅助监督 |
|---|---|---|
| M0 | 前后相邻两帧编码；直接返回原 `DiffusionUnetImagePolicy` | 只有动作扩散损失 |
| mamba_only | 两帧短窗 + 连续 Mamba 历史 | 只有动作扩散损失 |
| memory_only | 上述条件 + 八槽交叉注意力 + 共享融合 | 只有动作扩散损失 |
| M5 | 与 memory_only 相同的动作网络 | 再加训练期 R3M（机器人表征预训练模型）关键帧预测损失 |

- M0 是原 CNN Diffusion Policy（卷积视觉扩散策略）类，不是自写简化版：`models.py:54-55`；底层 `DP/policy/diffusion_unet_image_policy.py:15-80,123-186,192-260`。其训练采样器、任务输入、裁剪和超参数由本项目适配，因此“原算法类”不等于“完整官方训练运行协议逐项不变”。
- `memory_only` 不仅增加八槽：`models.py:290-295` 增加 `goal_latent` 和 `goal_norm`，`368-371` 直接把当前视觉/本体特征、Mamba 历史与检索拼接做残差融合。即使检索向量为 0，该融合也能产生非零输出。因此 mamba_only→memory_only 的成绩差不能全部解释成八槽检索的因果贡献。这是已确认的架构差异，不是实现错误。
- M5 与 memory_only 的共享动作路径、目标维度和构造顺序一致；后者先构造辅助 head（预测头）以匹配动作网络初始化随机序列，再删除该头：`models.py:290-307`。因此 memory_only→M5 是此四组中最干净的辅助监督对照，前提是训练/评测协议也一致。
- mamba_only/memory_only 没有沿用 M2/M3/M4 的过去动作重建损失：`models.py:25-27,58-61,427-432`。

建议：论文准确称第三组为“八槽检索与共享融合”。若要证明八槽本身必要，可利用已有模型做读槽干预；无需为当前代码审查添加模型模块或启动训练。

## 3. 连续 Mamba 双缓存、因果性、梯度和训练/在线一致

- `models.py:66-117` 同时接收并返回卷积缓存和 SSM（状态空间模型）缓存。卷积拼接旧缓存最后三步和当前序列，不重复吃上一段；最终保留最后四步。状态递推只使用截至当前的 token（时序特征）。
- CUDA（显卡计算平台）分段扫描用零初态扫描加传入 SSM 初态的齐次解校正，并显式重算最终状态以保留梯度：`80-102`。未发现漏传初始 SSM 状态或仅保留一类缓存的问题。
- 训练按同一示范连续 64 个锚点分段，状态在新示范清空，段间保值并 detach（切断梯度），每轮无跨示范遗留：`data.py:273-309`；`train.py:191-214,242-243`。`EpisodeState.detach()` 同时处理双缓存、事件张量、上一帧、历史和条件：`models.py:195-207`。
- 这是 TBPTT（截断反向传播）：当前段内历史和事件可求梯度；跨段老事件的数值可供读取，但后段损失不会反向训练 64 步前的写入特征。这是当前训练设计，不能描述为整回合端到端梯度。
- `segment_condition` 的上一帧由 carry（跨段状态）保留，第一帧才复制自己：`models.py:339-342,376-388`；M0 使用原始上一帧形成同样短窗：`train.py:44-51`。
- 在线 `update_observation` 把每个环境的双缓存按 ID 拼批，再逐 ID 写回：`models.py:434-457`。`evaluate.py:308-311,338-357` 在初始观测后，以及每个执行动作后更新一次，包括 8 步动作块的中间观测；只在块边界采样动作，符合当前连续观察设计。`371-374` 检查观测计数和动作计数一致。

极小 CPU（中央处理器）核查使用当前源码中函数的 AST（抽象语法树）片段和小型代数核心，单线程且隐藏所有 GPU，没有实例化视觉主干或大 Unet。结果：

| 针对性核查 | 结果 |
|---|---:|
| Mamba 整段/分段输出最大误差，非零初始双缓存 | 1.49e-8 |
| Mamba 整段/逐帧输出最大误差 | 2.98e-8 |
| 卷积缓存最大误差 | 0 |
| SSM 缓存最大误差 | 5.96e-8 |
| 前四步输出对后续输入的梯度 | 0 |
| CUDA 初态校正公式与逐步递推的输出/末态误差 | 各 4.77e-7 |

这些核查支持递推公式和数值分段一致性；本次没有运行实际 CUDA kernel（计算内核）或完整模型评测。

## 4. 八槽与交叉注意力没有偷看未来或跨环境混用

- `OrderedEvents.read()` 只读取当前已有事件：`models.py:143-152`；每一时刻先 read 再 write，当前帧事件不会反向进入此前条件：`351-355`。
- 槽含视觉向量、历史向量、相对时间年龄和有效位，不含任务标签、未来完成帧或真值动作：`120-152,327-328`。
- 当前帧查询含当前视觉/本体和截至当前的 Mamba 历史：`365-371`。编码器先拼视觉再拼低维状态（`DP/model/vision/multi_image_obs_encoder.py:127-179`），所以 `features[..., :visual_dim]` 的视觉切片没有把本体当图像。
- 槽满时保护初始事件，并淘汰后续相邻冗余对的较早成员：`models.py:154-174`。事件数量上限是常量 8。没有按子任务完成真值触发写入。
- 写入选择由脱离梯度的余弦分数决定，选中后的视觉/历史张量仍保留段内梯度；交叉注意力投影可学习：`models.py:132-133,239-250,344-355`。极小 CPU 核查：空库输出精确 0，无效槽梯度精确 0，有效槽内容和 query（查询向量）梯度非零。
- `_online` 以环境 ID 为键，每个 state 有自己的缓存/槽/时间，组前和组后 reset：`models.py:308,330-337,434-457`；`evaluate.py:309-311,365-367`。未发现环境间共享 `OrderedEvents` 实例或跨回合复用旧状态。
- 评测 `task_diagnostics`（任务诊断）输出只进入日志；政策实际只接收相机 RGB 和 14 维关节状态：`evaluate.py:218-236,282-298`。没有将日志里的初态/阶段真值送给模型。

## 5. R3M 关键帧仅用作训练标签，推理不执行教师或预测头

离线：`ablation_targets.py:70-118` 从训练示范注释确定事件完成图 → 冻结 `R3MSpatial` → 保存目标特征；`spatial_goal_targets.py:29-55` 将教师设为 eval（评估模式）且禁止梯度，读取 layer4（第四层特征），池化为 3×3×512 并逐位置归一化。

训练：`GoalTargets.batch()` 返回标签及前半段有效掩码（`goal_targets.py:46-86`）→ `data.py:297-305` → `train.py:53-55,90-94` → `models.py:422-426`。条件只由当前/历史观测构成，未来特征只参与与预测头输出的余弦损失。有效训练区间不晚于目标帧，`goal_targets.py:65-76` 明确约束，辅助权重在 10 轮内增至 0.02。

推理：`evaluate.py:392-407` 从 EMA（指数移动平均）检查点加载策略并 `.eval()` → `models.py:468-481` 只把已缓存条件交给采样器。`goal_head` 虽然仍在 M5 的 checkpoint（模型检查点）中，但这个预测路径没有调用它；冻结教师也不被构造或载入。未发现未来特征/目标掩码/语言事件描述泄漏到在线预测。

RGB、动作和本体 normalizer 在训练构建并写入模型状态；评测直接载入检查点的 EMA 状态，而不重新用测试数据拟合：`train.py:99-104`，`evaluate.py:400-407`。视觉主干使用 GroupNorm（组归一化），无跨 batch（批次）统计；评测 `.eval()` 使官方随机裁剪器取中心裁剪：`DP/model/vision/crop_randomizer.py:80-105`。未发现评测仍保留训练随机裁剪或用测试数据更新归一化统计。

## 6. 需实验证实的效果风险：固定阈值与槽保留行为

`calibrate.py:35-85` 使用初始化的 M3，在 8 条训练示范相邻帧余弦分布的第 20 百分位固定阈值。其优点是没有测试数据参与；局限是阈值不随特征学习调整，校准时中心裁剪，而训练时随机裁剪。M3 与 M5 的事件视觉投影还因 M3 先构造 past_head（过去动作预测头）而采用不同初始化随机序列（`models.py:267-279`）。这些都可能改变写槽频率和语义，但不能仅凭源码量化影响。

当前已有日志可直接看到频繁替换、多个极早槽长留的现象。以下每项只读取 epoch_050 首条现成回合，属于实例证据，不能当全任务统计：

| 当前修正任务/种子 | 观测数 | 写入数/替换数 | 最后保留事件时刻 |
|---|---:|---:|---|
| swap_T / 100000 | 601 | 274 / 266 | 0,1,2,3,5,11,594,599 |
| swap_blocks / 100001 | 600 | 438 / 430 | 0,1,2,3,5,8,592,596 |
| battery_try / 100000 | 794 | 515 / 507 | 0,1,2,7,10,19,788,791 |
| press_button / 100000 | 1501 | 390 / 382 | 0,1,5,23,61,1362,1473,1474 |

证据路径为 `EXP/m5_seed0_rgb_corrected/<task>/seed0/evaluation/M5/epoch_050/episodes.jsonl` 首条。四条均 `initial_retained=true`，Mamba 观测数与日志总观测数相等。

这说明当前八槽不等于八个语义子任务完成事件；中段事件可能被大量替换。它是记忆效果/叙事风险，没有把失败算成功的作用。建议先结合已有槽日志和关键动作时刻分析，再决定是否要改阈值或写入策略；本审查不改代码、不自动增开实验。

## 已覆盖文件

完整或按关键路径阅读：
- `SRC/AGENTS.md`。
- `SRC/implementation/rmbench/models.py`、`data.py`、`train.py`、`goal_targets.py`、`ablation_targets.py`、`calibrate.py`、`prepare_m5_rgb_corrected.py`。
- `SRC/implementation/rmbench/evaluate.py` 的模型载入、输入处理、在线动作块与 reset 路径（1-407 行）。
- `SRC/implementation/rmbench/spatial_goal_targets.py` 的教师、目标生成与有效区间路径。
- `SRC/implementation/coffee_models.py`（区分有限窗口旧实现与当前连续实现）。
- `SRC/implementation/rmbench/test_models.py`（只读既有检查定义，未运行其大模型检查）。
- `DP/model/vision/multi_image_obs_encoder.py`、`model/vision/crop_randomizer.py`、`policy/diffusion_unet_image_policy.py`、`common/normalize_util.py` 相关路径。
- 当前六任务 `configs/protocol.json` 及 `configs/M5.json`；旧九任务 seed0 的协议及存在的模型配置；`EXP/m5_seed0_rgb_corrected/result_sources.json`；上述四条现有回合日志。

未覆盖完整 checkpoint 张量/训练重跑/完整模型 CUDA 前向；本次没有为只读代码审查执行无关冒烟测试、哈希或版本核验。

## 用户最新安排

RGB版本比较问题本轮仅保留记录，暂不处理、不重配对照、不重跑；其余模型与评测审查照常完成。
