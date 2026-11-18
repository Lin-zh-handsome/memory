# Cover Blocks M3 epoch 100 seed 100000 diagnostic
- [x] Locate original run, seed pair and isolated spare GPU.
- [x] Replay original batch through the first reveal and save pre-decision representations and actions.
- [x] Fit a location readout from other seeds, validate on held-out seeds, and evaluate the replay.
- [x] If decodable, test matched memory counterfactuals at fixed observations and diffusion noise.
- [x] Summarize verified results and limitations. Preserve formal run and checkpoint.

## Projection versus denoiser
- [x] Decode red location after projection, holding out natural layouts and late replay timestamps.
- [x] Query the frozen diffusion head with those same projected location substitutions at fixed noise.
- [x] Report the localization and an incremental repair plan.
