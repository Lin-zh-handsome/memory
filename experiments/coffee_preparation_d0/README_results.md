# CoffeePreparation_D0：官方 CNN-DP M0/M2

本实验尚在执行，未产生完整闭环结果。未完成的数值保持空值，不能作为零成功率。

服务器目录：`/home/linzhenghong/project/coffee_dp_official/experiments/coffee_preparation_d0/`。
源代码目录：`/home/linzhenghong/project/coffee_dp_official/source/`；适配代码目录：`/home/linzhenghong/project/coffee_dp_official/implementation/`。

## 来源与方法

全新克隆官方仓库，未使用 Lin-zh-handsome/depth 代码：

- Diffusion Policy：<https://github.com/real-stanford/diffusion_policy>，提交标识 `5ba07ac6661db573af695b419a7947ecb704690f`。
- MimicGen：<https://github.com/NVlabs/mimicgen>，提交标识 `c0fbff22ba40060f05a917a80817f3f5c20731e7`。
- 数据：官方核心集 `amandlek/mimicgen_datasets/core/coffee_preparation_d0.hdf5`。优先尝试官方脚本，因缺少 gdown 导入失败，随后从同一官方数据的 hf-mirror.com 镜像下载；镜像降速后通过现有代理连接官方 Hugging Face 源，四路 HTTP 字节范围续传剩余部分。
- 指定补丁 ZIP：本地工作目录递归检查及远端用户目录四层检查均未找到，按照任务指定架构实现。

M0 直接实例化官方 `DiffusionUnetImagePolicy`。M2 继承该类，保留同一观测编码器及 U-Net 参数，新增指定 Mamba（选择性状态空间模型）历史分支、条件维度和过去动作重构头。原始仓库代码不作修改，适配代码位于独立目录。

M2 只把最近16帧观测送入共享编码器。最后两帧编码直接来自该窗口；过去8步专家动作仅为训练标签。辅助损失采用动作有效掩码，并直接回传至 Mamba 与可训练的视觉编码器。推理不调用辅助头，不使用跨整段任务的流式状态。

## 固定协议

完整配置见 `configs/protocol.json`。两模型均使用种子0、1、2，各250个完整训练轮次；第50、100、150、200、250轮各评测50回合，每回合最多800环境步。每种子取五个检查点成功率的最大值，再对三个种子求均值及样本标准差（ddof=1）。这是训练期间 Best-of-5（五个检查点择优）结果，不能解释为独立测试集的无偏估计。

沿用官方图像策略训练配置：ResNet18（18层残差网络），两路相机分别编码；GroupNorm（组归一化）；84×84图像，训练随机裁剪76×76、评测中心裁剪；AdamW（解耦权重衰减优化器），学习率0.0001，余弦调度与500步预热；批量64；预测16步，执行8步，动作执行从索引1开始；100步 DDPM（去噪扩散概率模型）；EMA（指数移动平均）策略评测。未使用预训练视觉权重。

使用原数据的控制器动作，不转成绝对位姿或其他任务的动作语义。实际数据包含1,000条示范、689,273个时间步、两路84×84原生RGB图像和7维增量动作，控制频率20 Hz。直接使用官方存储的图像、本体状态和同一步动作；不使用object或相对物体位姿等特权观测。已在第0、336、672步将原生观测与状态回放对照：图像最大像素差为1，末端位置最大差约1.98×10⁻⁸，方向和时间索引一致。详细记录见 `logs/stored_vs_state_replay_alignment.json`。采样保持官方序列规则：前填充1步、后填充7步；历史观测在各回合内左侧重复首帧，过去动作的填充标签掩码为0。

评测初始条件由官方 D0 reset（环境重置）分布生成种子100000—100049，保存完整环境模型 XML 与仿真状态，所有模型、种子与检查点恢复同一批状态。成功判定调用官方 `env.is_success()['task']`；首次完整任务成功即计成功。

## 运行与恢复

服务器执行：

```bash
ROOT=$HOME/project/coffee_dp_official
nohup bash "$ROOT/implementation/run_pipeline.sh" > "$ROOT/experiments/coffee_preparation_d0/logs/pipeline.log" 2>&1 < /dev/null &
```

恢复使用同一命令。准备阶段按已完成回合继续；训练从各次运行最新完整轮次恢复模型、优化器、调度器与随机数状态；闭环评测保留已记录回合，继续缺失回合。仅在没有其他计算进程且显存占用低的 GPU（图形处理器）上启动作业，动态分配，不固定卡号。遵守用户全局偏好：默认仅用1张卡依次训练，确有需要时总计最多2张；评测阶段在同一张卡上运行10个并行环境并批量推理。

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/project/coffee_dp_official/source/diffusion_policy:$HOME/project/coffee_dp_official/implementation"
$HOME/miniconda3/envs/coffee-dp/bin/python "$HOME/project/coffee_dp_official/implementation/summarize.py" --root "$HOME/project/coffee_dp_official/experiments/coffee_preparation_d0"
```

单次运行恢复可直接执行 `train.py --root <实验目录> --model M0 --seed 0`，并设置所选空闲 GPU 的 `CUDA_VISIBLE_DEVICES` 和 `MUJOCO_EGL_DEVICE_ID`，勿与调度器重复运行。

## 实际结果与待完成项

最终状态见 `status.json`；逐轮训练见 `logs/*_epochs.csv`；全部回合见 `eval/episodes.csv`；全部检查点见 `eval/checkpoint_results.csv`；三种子统计见 `results_summary.csv`。未完成全部五个检查点的种子不计算最终最佳成绩。

参数量与辅助梯度证据保存至 `logs/interface_checks.json`。接口检查计时包含初始化开销，不作为稳定吞吐基准；正式训练每轮耗时及 GPU 峰值显存单独记录。辅助头不参与推理，推理参数量应扣除该头。完整成功与失败视频分别保存至 `videos/`；若实际没有成功回合，不伪造成功视频。

失败阶段根据官方 `_get_partial_task_metrics()` 逐步记录的子任务完成情况分析，在闭环数据形成后汇总。研究方向判断也必须等待实际结果。

环境兼容修正：独立 Conda 环境使用 robosuite 1.4.1、官方推荐的 MuJoCo 2.3.2、robomimic 0.3.0；仅将 robomimic 中未用于该版本环境的旧 `mujoco_py` 导入改为可选，修正脚本完整保留。Mamba 2.2.2 使用与 PyTorch 2.4、Python3.10 对应的官方 CUDA 预编译扩展。完整实际版本由运行脚本写入 `configs/code_versions.json`。



2026-10-30 用户收缩资源后，仅保留 M2 种子0继续训练，另外4个尚未完成第一轮的训练前缀停止并重新排队；它们不计入完成轮次或闭环成绩。已保存 logs/resource_policy_change.json。

<!-- ACTUAL_RESULTS -->

## 实际执行状态

已完成训练：2/6；已完成检查点评测：17/30；已记录闭环回合：850/1500。

| 模型 | 种子0最佳 | 种子1最佳 | 种子2最佳 | 均值 ± 样本标准差 |
|---|---|---|---|---|
| M0 | 88.00% | 未完成 | 未完成 | 未完成 |
| M2 | 90.00% | 未完成 | 未完成 | 未完成 |

结果尚不完整，不能判断 M2 的收益或研究方向。

参数与计算开销（正式轮次耗时包含数据加载及共享算力竞争；种子0两模型同卡并行，耗时比不能直接解释为模块纯计算开销）：

```json
{
  "M0_parameters": 329334535,
  "M2_training_parameters": 337457727,
  "M2_inference_parameters": 337377543,
  "added_training_parameters": 8123192,
  "added_inference_parameters": 8043008,
  "median_formal_epoch_seconds": {
    "M0": 1170.7155741387978,
    "M2": 1629.1611281021032
  },
  "timing_interpretation": "Observed wall time includes data loading and GPU sharing. Seed-0 models are authorized to run concurrently on one GPU; timing ratios are not isolated module compute overhead.",
  "observed_epoch_time_ratio_M2_to_M0": 1.3915943070122283
}
```

失败阶段计数见 `eval/failure_stages.csv`，记录失败回合中从未达到抓取胶囊、插入胶囊、抓杯、放杯等官方子条件的比例。多个子条件可能同时缺失，不能视为互斥归因。
