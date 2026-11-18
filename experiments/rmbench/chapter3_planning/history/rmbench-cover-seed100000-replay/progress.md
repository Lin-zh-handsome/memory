Initialized an isolated diagnostic directory. Formal outputs and checkpoint are read-only inputs.
The server lacks numactl; use existing simulator thread limits, one BLAS thread and lower process priority.
Launched replay PID 4098625 on GPU 5, with original GPU 7 jobs left alone.
Analysis queued after replay on the same spare GPU. No live code or formal artifacts edited.
Replay stage trace matched exactly, 15 MB capture saved. Location readout was 24/24 held-out and center at every replay snapshot. Synthetic color-channel swaps did not move the readout; reject and use genuine layout donors.
Retry uses cached real-seed readout and a mask restricted to the actual block row.
Completed isolated replay and action diagnosis. Artifacts: pre_reveal_features.pt, replay_alignment.json, location_readout.json, intervention_inputs.json, action_counterfactuals.json and action_counterfactuals.pt.
Started follow-up projection diagnosis from the saved 29 feature snapshots and 24 natural-layout controls. No simulator replay required.
Projected slot alone held-out 24/24; cross-time complete projected conditions 196/528 matched contrasts and 4/44 factual reads. Native-scale donor substitution reproduces baseline exactly and gives at most 1.23e-4 rad arm change. Checking same-time held-out readout and amplified perturbation as a sensitivity test.
Completed follow-up offline projection and frozen-head diagnostics. Saved projection_readout.json, projection_features.npz, projected_donor_actions.json/.pt, projection_sensitivity.json.
