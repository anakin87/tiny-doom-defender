# tiny-doom-defender

![cover](./assets/cover.png)

A ~1M-param conv-stem ModernBERT agent for VizDoom's *defend_the_center*: behavior-cloned from a scripted oracle, then PPO-refined. Transformers-native — every checkpoint is a standard HF model dir.

## Files

`src/tiny_doom_defender/`

| File | What it does |
|---|---|
| `configuration_doom.py` | HF config: stem geometry + nested ModernBERT encoder config |
| `constants.py` | Pipeline constants: input geometry, action encoding, VizDoom settings, seed pools |
| `data.py` | Frame-stacking dataset over the recorded oracle data |
| `env.py` | Gymnasium env producing the flat uint8 observation the policy consumes |
| `evaluation.py` | Checkpoint loading, episode loop and metrics shared by the eval scripts |
| `game.py` | Raw VizDoom interface: scenario setup and screen downsampling |
| `modeling_doom.py` | The models: conv stem + ModernBERT trunk, with SFT classifier and RL policy heads |
| `utils.py` | Action/observation encoding helpers, device picker, pipeline-geometry guard |

`src/tiny_doom_defender/scripts/` — console commands

| Command | What it does |
|---|---|
| `create-model` | Write a fresh untrained model dir for a non-default architecture (optional) |
| `record-oracle` | Scripted oracle plays and records the SFT dataset |
| `train-sft` | Behavior-clone the classifier on oracle demonstrations |
| `train-ppo` | PPO-refine the SFT policy; writes per-iteration snapshots |
| `select-ppo-snapshots` | Rank a run's snapshots, keep the best as `policy_best` |
| `quantize-int8` | Quantize a checkpoint to int8 so policy + code fit on a floppy |
| `eval-model` | Score a checkpoint on held-out test seeds |
| `play-doom` | Watch a checkpoint play in a live DOOM window |
