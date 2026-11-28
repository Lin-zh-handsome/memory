# RMBench 官方评测与环境补丁审查

日期：2026-11-28。范围：只读代码、文档和依赖源码；未启动仿真、训练或 GPU（图形处理器）作业，未改协议、环境、模型代码。

## 结论与分类

**没有发现项目通过修改九任务成功条件、放宽步数、跳过官方物理执行或额外调用成功函数来抬高 M5 成功率。** qpos（关节位置目标）补丁跳过的是本来不参与关节回放的 CuRobo 初始化，实际关节目标仍走原来的 CPU TOPP（中央处理器上的轨迹时间参数化）和物理子步循环。

**明确差异但合理适配：** 单头部相机、96×96 图像、自定义模型、16/2/8 的预测时域/观测帧数/动作窗口、50 回合和五检查点取最好，支持项目内部消融；不能称为当前官方 DP（扩散策略）默认配置的逐项复现或论文正式统计口径。

**确认实现 bug（程序缺陷）：** 官方 press_button 的确认按钮复位错误读取第一个按钮的关节位置。项目未改这段代码，是否影响特定回合尚需轨迹或仿真验证，不能用它解释其他任务高分。

**尚需实测：** 显式渲染设备与自动设备路径的像素差异；press_button 缺陷是否实际触发；与论文或官方基线数值直接比较时的匹配实验。本审查未补跑这些实验。

## 路径与官方来源

- `R/` = `/home/linzhenghong/project/rmbench_m0_m3/source/RMBench/`。
- `L/` = `/home/linzhenghong/project/rmbench_m0_m3/source/m5-goal-spatial-r3m/implementation/rmbench/`。
- 实际解释器由根审查确认：`/home/linzhenghong/miniconda3/envs/coffee-dp/bin/python`；TOPP 依赖核查使用此环境源码。
- 当前检出版本提交标题：`7d1cdfc Merge pull request #24 from FlowsMple/main`。只读 `git diff` 显示修改仅为 `envs/_base_task.py`、`envs/robot/planner.py`、`envs/robot/robot.py`；九任务文件、步数表和官方评测脚本无修改。
- 已通过 GitHub 连接器读取当前官方 README 评测部分，及固定版本 `envs/press_button.py:187–201`、`envs/_base_task.py:1586–1630`、`envs/robot/robot.py:326–358`。
- [当前官方 README](https://github.com/RoboTwin-Platform/RMBench/blob/main/README.md#4-run-policies) 同时保留原适配器入口和推荐的新 XPolicyLab 入口；[固定官方版本](https://github.com/RoboTwin-Platform/RMBench/tree/7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7)。

## 三处补丁

### 与原流程一致：关节控制和成功检查

`R/envs/_base_task.py:1596–1639` 用当前关节位置和目标位置调用左右 Mplib TOPP，参数为 `1/250`；1700–1735 逐子步设关节及夹爪、`scene.step()`、`check_success()`，成功才置 `eval_success=True`。这些代码在补丁中均未改。TOPP 失败时手臂维持、夹爪执行 50 子步的回退同样来自官方原代码。原 qpos 路径本来就不调用 CuRobo 碰撞规划，因此该补丁没有关闭原有的关节回放碰撞检查。

项目每个 14 维目标调用 `env.take_action(..., action_type="qpos")`（`L/evaluate.py:345`）。官方 XPolicyLab 把 `joint` 映射为 `qpos`（`R/scripts/eval_policy_xpolicylab.py:1242–1247`），左臂、左夹爪、右臂、右夹爪拼接顺序见 1087–1124、1153–1161，与项目一致。

### 明确不同但合理适配：qpos_only 初始化

`R/envs/robot/robot.py:281–287` 在策略回放跳过 CuRobo 工作进程，调用同参数的 `set_topp_planner()`（339–359）。专家筛选强制 `qpos_only=False`（`L/evaluate.py:473–475`）。`R/envs/robot/planner.py:247–257` 仅把原 200 步夹爪插值改成静态方法，公式保持；`robot.py:379–392` 直接调用同一插值。

复位时 `robot.py:136–150` 确实复用初次构建的 MplibPlanner；不能仅凭对象持有旧场景就认定控制有 bug。其 TOPP 是 `R/envs/robot/planner.py:315` 绑定的方法。实际 coffee-dp 环境的 `lib/python3.10/site-packages/mplib/planner.py:351–379` 只用传入 path（关节路径）及固定 joint_vel_limits / joint_acc_limits（关节速度/加速度限制），不读取旧场景或机械臂实体。当前纯 qpos 控制复用该对象未发现改变执行约束的机制；此结论不扩展至末端位姿规划。

### 明确不同但合理适配：渲染设备

`R/envs/_base_task.py:91–94,210–246` 加入设备及 shader（着色器）参数；显式分支使用 `PhysxCpuSystem()` 与指定 `RenderSystem(device)`，无界面运行。保留原 scene_config（场景配置）、1/250 秒物理步长、摩擦/弹性参数。默认仍为 `rt`（光线追踪），32 采样、深度 8、OIDN 降噪。未改变成功函数或物理动作循环。像素是否逐值相同未实测，应披露设备适配，不能据此断言性能升降。

## 官方入口、初态、动作时序

### 与原流程一致

1. 原入口 `R/scripts/eval_policy.py:241–275`、XPolicyLab 入口 `eval_policy_xpolicylab.py:808–844` 先 `play_once()`，要求 `plan_success and check_success()`，再用同一 seed（随机种子）重新初始化。项目 `L/evaluate.py:157–175` 使用同一筛选。专家成功不是策略成功；专家那次成功检查在重新初始化前，任务计数由 `load_actors()` 重置。
2. 项目只以官方 `env.eval_success` 停止/计分（`L/evaluate.py:304–306,345–347,379`），未以中间 stage（阶段）、第一次揭盖或 max_reward（最大奖励）替代终局成功。
3. 官方步数：observe_and_pickup 250；rearrange_blocks 700；cover_blocks 1500；swap_T 600；press_button 1500；swap_blocks 1000；blocks_ranking_try 3500；put_back_block 500；battery_try 1000。项目读取核对同一 `_eval_step_limit.yml`（`L/evaluate.py:88–101`），循环使用同一 `take_action_cnt` 上限。这是目标调用次数，不是物理子步数。
4. 官方 `get_obs()`（`R/envs/_base_task.py:457–529`）仅刷新渲染和读取图像/关节，不 `scene.step()`、不 `check_success()`。项目目标执行后取新观测，成功时复用官方已捕获的终态观测（`L/evaluate.py:345–348`）。XPolicyLab 同样在动作窗口内逐目标更新观测（脚本 884–921；`R/XPolicyLab/policy/DP/deploy.py:5–17`），不存在项目额外拍照就多推进任务状态。
5. 官方 `frequency/ctrl_freq` 默认 30（XPolicyLab 脚本 789）只存入观测附加字段（1055）；循环没有墙钟等待强制 30 Hz（每秒30次）执行，物理时长由 TOPP 决定。DP 历史队列按每次 `update_obs()` 更新（`dp_runner.py:57–68`）。不能把元数据频率误当成固定目标执行节拍。

### 明确不同但合理适配

官方原入口固定 100 回合（`R/scripts/eval_policy.py:164–169`）；XPolicyLab 默认 100（306–308），起点 `100000*(1+seed)`。项目共同 50 个专家有效初态、五个完整节点取最好，是自定义统计和选点协议；会引入对同一评测集的选点偏差，应与论文口径分开。共享初态、分片汇总及实际结果由根审查独立核对。

官方 `demo_clean.yml:10–32` 关闭背景、光照、桌高、相机位移随机化，开启头部/腕部相机。项目沿用干净场景，关闭未使用腕部图像，策略用 head_camera（头部相机）96×96；第三视角用于视频。腕相机创建不采样随机数（`R/envs/camera/camera.py:133–151`），头部/静态相机随机数路径保留（92–100、174–215）。

当前官方 `R/XPolicyLab/policy/DP/model.py:129–140` 使用头部和左右腕 3 路 320×240 图像；`diffusion_policy/config/robot_dp.yaml:13–15` 默认 horizon（预测时域）8、n_obs_steps（观测帧数）3、n_action_steps（动作窗口）6。项目 16/2/8、单视角和共同预算内部基线是自定义配置，不能叫逐项官方 DP 复现。

## 隐藏变量与子任务规划

项目策略观测只返回配置相机 RGB（红绿蓝图像）及14维关节状态（`L/evaluate.py:218–227`）。诊断中的目标编号、正确电池组合、揭盖顺序在 `initial_state` / `report_state` 输出，仅记录到日志，未加入 `stack_batch()` 的策略输入（270–300）。这核查了回放输入；训练标签由根审查负责。

官方 Mem_0 的 M(n) 规划属于策略适配器：`R/XPolicyLab/policy/Mem_0/model.py:320–321` 用初始图像启动高级规划；348–355 根据模型子任务结束信号触发切换，297–308 调 `update_high_observation()`。通用环境没有自动给项目提供真实子任务编号，项目也没调用专家子任务接口替策略操作。端到端策略可用自身隐式规划完成同一完整任务，但与 Mem_0 Planning+Execution（规划与执行）架构不同，不能称为复现该架构。

## 成功函数副作用和九任务条件

所有任务文件未改。成功函数每个1/250秒物理子步调用，不宜在回放中额外调用：按钮计数/复位、遮挡和阶段迁移本身包含在这些函数中。

| 任务 | 官方完整判据及必要细节 | 副作用/代码 |
|---|---|---|
| put_back_block | 中央放块并首次按键成stage1；释放按钮后送回初始目标，XY各<3cm、Z<0.77、右夹爪打开，成stage2 | `put_back_block.py:150–168`：按钮计数/复位、阶段推进；stage2保持成功。 |
| swap_blocks | 互换至对侧原篮，XY各阈值4/6cm、Z<0.765、右夹爪开、flag有效、恰按1次且按钮标志正受压 | `swap_blocks.py:187–191,208–225`：按钮复位/计数；两块同x且近似同高可永久使flag失败。 |
| observe_and_pickup | 前20次成功检查任一末端位移不得>3cm，第20次加墙，之后目标物Z>0.8且未fail | `observe_and_pickup.py:170–187`：get_obs_cnt在成功检查中加1，实际计物理子步，不是20帧。官方仅检查目标高度，不要求两指接触。 |
| rearrange_blocks | 首次按键时两块在指定mat成stage1；释放后第一块留目标mat，第二块至空区，右夹爪开、计数1，成stage2 | `rearrange_blocks.py:163–190`：按钮复位/计数、阶段推进；stage2保持。 |
| swap_T | 两T块到对方初始XY，均<2.5cm；各自姿态误差<15度；双夹爪开；两Z<0.77 | `swap_T.py:91–124`：除最大奖励更新，无阶段计数副作用。 |
| cover_blocks | 按左中右盖三块，再按红绿蓝揭盖，完整通过6次状态转移，无不合法稳定状态 | `cover_blocks.py:68–78,156–192`：推进current_state_pointer或永久fail；不是只揭第一个盖就成功。 |
| battery_try | 两电池各在对应槽XY 2cm内、朝向组合正确、仪表开 | `battery_try.py:169–177`：组合正确时开仪表；官方位置判据只查XY，未额外查Z/释放。 |
| blocks_ranking_try | 三块相邻XY在[13cm,4cm]内、x1<x2<x3、右夹爪开、按钮当前按压标志为真 | `blocks_ranking_try.py:179–193`：更新计数/按钮复位/释放标志；官方不要求恰按1次，未额外查Z。 |
| press_button | 两计数等于对应卡片目标，确认按钮标志为真 | `press_button.py:187–201`：三按钮计数/复位，有下述官方缺陷。 |

**诊断辅助调用为纯读取。** `L/task_diagnostics.py:64–117` 对 Cover 调 `check_if_cover`、对 Battery 调 `check_battery_in_slot`、`get_curr_combination`、`check_dashboard_on`。对应 `R/envs/cover_blocks.py:80–84`、`battery_try.py:74–101` 只读位姿/关节，不开仪表、不推进阶段、不改计数。诊断未调有副作用的 `check_correct_combination()` 或 `check_success()`。`track_controller_fallbacks`（171–194）仅包装计数，原样返回 TOPP 结果或重新抛出原异常。

## 确认的官方 press_button 缺陷

`R/envs/press_button.py:199`：
```python
self.set_button_unpressed(self.check_button,
    target=min(0.0, self.get_current_button_value("button", self.button1)+0.002))
```

它给 check_button（确认按钮）写入 button1（第一个计数按钮）的位置，而非自身位置；实际读取由参数决定（81–91）。当第一个按钮持续受压时，可能把受压位置复制到确认按钮，下个子步 `update_press_success()` 可据此置确认标志。最终判据只看两计数吻合及此标志（201）。同一错误已在官方固定版本 GitHub 文件核对，项目未修改。

已确认错误对象读取和可能影响确认键状态的机制；不能仅凭代码判断具体回合已受影响，不能据此改动既有成功数量。尚需接触/按钮关节轨迹或针对性仿真；本次未改环境、未补跑。

## 交接

本报告核查执行机制及官方源码，没有把协议不同自动判成虚假，也没有把高成功率当正确性证据。实际逐回合计分、分片汇总、完整度、五节点选择和高分日志能独立证实的物理条件由根审查负责；日志未记录的夹爪开度/四元数等不能由stage反推为独立证实。
