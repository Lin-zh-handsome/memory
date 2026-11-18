# Findings
- 原实验历史为 d9ff5eb（训练）及 4577eda（闭环和续跑）。保留服务器原始快照。
- Swap Blocks 官方示范 50 条，共 30,017 帧，长度 585–643，相机键 ['cam_head', 'cam_left_wrist', 'cam_right_wrist', 'cam_third_view']。所有动作均匹配下一时刻的 14 维关节/夹爪状态。仅用训练数据校准双阈值 {'visual': 0.9966006994247436, 'history': 0.9952274084091186}。官方步数上限 1,000。
