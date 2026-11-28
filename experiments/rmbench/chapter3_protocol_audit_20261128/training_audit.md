# Training/data/target audit

Scope: read-only review of training/data/auxiliary target code and actual protocols. No experiment source or running jobs changed.

## Initial verified facts
- `data.py:273-309` shuffles full episodes deterministically by `(seed, epoch)` and traverses chronological contiguous anchor blocks; terminal RGB observations are included for recurrent processing. It uses all 50 official demonstrations, no offline validation split (`data.py:147-175`).
- `train.py:135-178` restores model, EMA, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, sampler epoch/batch position and detached recurrent carry. Need inspect recurrent state implementation and actual logs before judging exact resume.
- M5 labels are provided as `goal_feature`/`goal_valid_mask`, separate from `obs` (`train.py:41-59`; `goal_targets.py:46-86`). Endpoint images are training demonstration future labels, not automatically evidence of evaluation leakage.
- Corrected M5 preparer deliberately reuses completed range-normalized tasks and retrains unfinished identity-normalized tasks (`prepare_m5_rgb_corrected.py:43-50,69-73`). Actual roots and fair comparison impact remain to inspect.

## Confirmed comparison/configuration problem
- Current M5 roots for `swap_blocks,battery_try,rearrange_blocks,swap_T,blocks_ranking_try,press_button` have `vision.rgb_normalization=identity`, while the corresponding four-model ablation protocols and historical controls default to `range`. `put_back_block,cover_blocks,observe_and_pickup` M5 retain `range` by result_sources reuse. Corrected preparer also removes/recomputes event thresholds (`prepare_m5_rgb_corrected.py:69-90`); actual swap thresholds change from visual/history 0.996600699/0.995227408 to 0.997718096/0.996720076. Thus six-task future M5-vs-old-control scores confound RGB preprocessing and gate calibration with auxiliary supervision. No evidence this creates false successful rollouts; it prevents attribution to the auxiliary mechanism and a homogeneous nine-task protocol claim.
- `data.py:239-248` range maps [0,1] to [-1,1], then official `MultiImageObsEncoder.py:22-24,105-108` applies ImageNet normalization assuming [0,1]. For old range models the transform is `(2*x-1-mean)/std`; corrected identity models use `(x-mean)/std`. This is a real earlier preprocessing mismatch with the official encoder contract, but training/evaluation are both affected consistently, so it is not by itself train/test leakage or mathematical invalidity.
- Actual training budget is task-specific from all eligible anchors: put_back 300 updates/epoch (75000 total), cover 818 (204500), observe 150 (37500), rearrange 350 (87500), swap_T 300 (75000); other active jobs not yet complete. Budget equality is within a task/model protocol, not across tasks.

## Verified recovery/loss details
- `models.py:195-210` detaches and moves all carried conv/SSM/event/feature/history/condition tensors. `train.py:199-214` resets only at episode start, detaches only between segments, and clears at episode end. EMA optimization_step is restored from global_step (`train.py:145-146`), matching official EMAModel increment (`ema_model.py:58,88`). AdamW and scheduler restored, with deterministic independent NumPy episode order (`data.py:275-276`). No missing optimizer/scheduler/RNG state observed.
- Current action DDPM target path is epsilon matching protocol; M5 auxiliary loss is masked per valid anchor, 1-cosine averaged across nine positions (`models.py:396-426`), and weight ramps 0 to 0.02 over epochs1-10 (`train.py:62-63`). mamba_only and memory_only have no past-action auxiliary loss; model agent covers full architecture.

## Numeric data/label check completed
Used CPU NumPy memory maps for state/action and RGB headers only (no JPEG decoding). For eight ablation tasks, all prepared datasets have exactly episode IDs 0..49 with no duplicates; shape/length checks pass for all 400 episodes; maximum `abs(action[:-1]-state[1:])` is exactly 0. Totals:

| task | paired steps | action anchors/epoch | updates/epoch | label events | valid label anchors |
|---|---:|---:|---:|---:|---:|
| put_back_block |17562|17212|300|150|8745|
| swap_blocks |30017|29667|500|200|14948|
| battery_try |33653|33303|553|151|16790|
| observe_and_pickup |7468|7118|150|100|3698|
| rearrange_blocks |20085|19735|350|150|10005|
| swap_T |17149|16799|300|200|8522|
| blocks_ranking_try |73739|73389|1165|346|36775|
| press_button |26302|25952|430|448|13004|

All target goal indices are present for these eight tasks. Official task source matches specialized grouping: swap 3 placements+press; wait endpoint includes first wall-present RGB; press targets chosen at end of down motion before release. Cover uses visual completion verifier, not group-only endpoints.

Training/evaluation normalizers are not mixed within a checkpoint: `evaluate.py:400-406` requires exact checkpoint config equality and loads EMA state including normalizer.

## Additional scheduler/queue and baseline scan
- `run_ablation.py:87-103` launches train.py with root/data/model and M5-only goal-target path; it never overwrites protocols, moves checkpoints across models, or changes training data. `prepare_ablation.py:97-108` clones per-task config changing only training seed, attaches that task's data and annotations path, and queues all four variants. Current six corrected jobs have correct per-task target path.
- Scanning every `experiments/**/protocol.json` found only the six corrected M5 protocols with `rgb_normalization=identity`; no corrected M0, mamba_only or memory_only protocol currently exists. All identity protocols explicitly list only M5. Main corrected queue has six M5 jobs; old ablation queue has 103 jobs spanning four models but uses range protocols. Thus an identity M0 control is not merely pending in an existing configured queue.
- Queue checkpoint eligibility is file existence; actual evaluator validates model/epoch/config before loading. Completion markers do not carry full protocol identity (`run_ablation.py:15-20,31-38,43`), so manual mutation/reuse of a root could create stale skip risk. No actual stale marker/config mismatch has been demonstrated; corrected jobs use new roots and train.py refuses resume against differing config.

## Initial-state separation
Archive seed.txt was compared to actual old ablation reset manifests for all eight non-cover tasks. Training ranges are put_back0..49, swap0..94, battery1..146, observe0..53, rearrange0..49, swap_T0..54, ranking0..49, press0..49 (50 accepted demonstrations each). Evaluation seeds start at100000; intersections are empty. Old ranking/press manifests currently have only one seed, so their isolation evidence covers that prefix; corrected scorer uses copied/completed separate manifest. Merely reading training future keyframes into loss labels does not feed future/evaluation information into observations.

## Batch grouping and image augmentation: design adaptations, not confirmed unfairness
- Every variant uses the identical chronological 64-anchor episode blocks from `EpisodeDataset.batches`, including identical short final blocks and optimizer update counts. M0 assembles [previous,current] observation pairs (`train.py:44-50`); recurrent variants encode each chronological frame once and reuse its feature for adjacent pair/history (`models.py:310-325,339-342`).
- M0 official policy flattens its 2-frame windows and samples a separate random crop for each occurrence. Recurrent policy uses one crop for each frame at its time of arrival and caches that feature across the segment boundary. Crop size84x84 from96x96, marginal sampling rule, GroupNorm encoder, and evaluation center crop are shared. They therefore do not have bit-identical augmentation sequences or identical number of encoder calls. This follows the online recurrent representation and is not itself a privileged-information or extra-action-budget advantage. It should be documented as training adaptation when calling M0 an official DP reproduction.
- M5 and memory_only share the same recurrent batching/augmentation implementation; the RGB revision and threshold recalibration are the concrete current comparison confounds.
- Additional terminal observations in the final recurrent segment (`data.py:285-287`) do not become anchors, and the recurrent outputs are causal with per-sample GroupNorm. Those observations cannot influence the earlier conditions being trained, and final carry is discarded. No future-input leak found.
- Historical reused M0 protocols differ from the new ablation templates only in model/auxiliary fields unused by M0, reset-manifest location and resource metadata. Core optimizer, horizon, action chunk, crop, encoder, training epochs and loss settings match. Old range baselines are internally matched to old range M5, not current corrected M5.

## Cover completion of the numerical audit
Cover prepared data also has 50 episodes, 51077 paired steps, no shape/length mismatch, maximum next-state error exactly0, 50727 anchors and818 updates/epoch. Its target file has300 events, all with valid goals, shape(300,9,512),25478 valid anchors. Existing construction statistics report300 default completion frames verified,0 adjusted and0 invalid. Archive training seeds0..49 and evaluation100000..100049 are disjoint. Thus the numerical alignment audit covers all450 demonstrations in nine tasks.
Official RMBench `envs/utils/pkl2hdf5.py:137-142,155-165` explicitly writes state from raw[:-1], action from raw[1:], RGB from raw[:-1], matching local training alignment and the final-RGB omission assumed by event indexing.

## Classification and interpretation
**Confirmed error (preprocessing):** range-normalized RGB followed by ImageNet transform violates the official encoder's declared[0,1] input contract. Affects historical M0 and old range M5/memory models; it does not invalidate their measured simulator outcomes or produce success by label leakage.

**Confirmed comparison confound (P1):** only six M5 tasks switched RGB normalization and recalibrated gates; no matching identity controls are configured. Historical range-vs-range comparisons remain internally comparable; current corrected-vs-range comparisons cannot isolate the M5 auxiliary contribution. Nine-task combined source map is heterogeneous.

**Necessary/design adaptations:** chronological episode TBPTT, frame feature caching and augmentation reuse,50 accepted demonstrations, all eligible L-7 anchors, common per-task updates across variants, training-only future goal labels, and frozen R3M spatial target extraction. Their presence alone is not an error.

**Potential risks without observed failure:** completion marker skip checks do not bind protocol identity; random-initialization calibration may drift after training (scientific mechanism property, not demonstrated bug); source data manifest records train/eval isolation as prose, while archive seed.txt provides the actual evidence; recurrent carry values span optimizer updates by standard truncated-BPTT design, so mathematically exact full-episode gradient training is not claimed. No actual missing optimizer/scheduler/RNG resume state, episode leakage, wrong-task targets, loss/EMA update bug, action offset error or evaluation-config mismatch was found in this audit.

## Coverage
Read source AGENTS.md; training.py/data.py; goal_targets.py; spatial_goal_targets.py; ablation_targets.py; calibrate.py; prepare_ablation.py; prepare_m5_rgb_corrected.py; run_ablation.py; report_ablation.py source mapping; relevant models.py loss/state/batching sections; evaluate.py checkpoint loading; coffee_models.py/coffee_data.py (not imported by current rmbench entrypoints); official Stanford DP sampler, encoder, crop randomizer, policy loss and EMA; official RMBench DP data processing and HDF converter; eight auxiliary target metadata sets plus cover target/statistics; all current/reused protocols, queues, numeric prepared arrays and archive seed lists.

Review finished. Only this independent ledger was written. No experiment source/configuration, dataset, checkpoint, GPU job or shared planning file changed; no project artifact was downloaded to the local machine.

## 用户最新安排

RGB版本比较问题本轮仅保留记录，暂不处理、不重配对照、不重跑；其余模型与评测审查照常完成。
