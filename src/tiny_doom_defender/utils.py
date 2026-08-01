import json
import os

import numpy as np

from tiny_doom_defender import config


def combine_action(turn, shoot):
    """(turn 0..2, shoot 0..1) -> flat index 0..5."""
    return int(turn) * 2 + int(shoot)


def stack_channels(frames):
    """List of `N_FRAMES` (H, W, 3) uint8 frames (oldest first) -> (9, H, W) uint8.

    Channel order is [f0_RGB, f1_RGB, f2_RGB] with f2 = current frame. The SFT
    dataset and the live env MUST build the stack the same way (they both call
    this) so there is zero train/serve skew.
    """
    hwc = np.concatenate(list(frames), axis=2)  # (H, W, 9)
    return np.ascontiguousarray(hwc.transpose(2, 0, 1))  # (9, H, W)


def pack_obs(stack_u8, prev_actions):
    """(9, H, W) uint8 stack + length-N_PREV prev-action list -> (OBS_LEN,) uint8."""
    obs = np.empty(config.OBS_LEN, dtype=np.uint8)
    obs[: config.STACK_PIXELS] = stack_u8.reshape(-1)
    obs[config.STACK_PIXELS :] = np.asarray(prev_actions, dtype=np.uint8)
    return obs


def stem_config():
    """The conv-stem geometry, from config.py — the one runtime source.

    create_model.py stamps this into <model_dir>/stem_config.json and check_stem_config
    compares a model dir's stamp back against it, so both sides always speak of the
    same keys.
    """
    return {
        "input_resolution": [config.RES_H, config.RES_W],
        "n_frames": config.N_FRAMES,
        "in_channels": config.IN_CH,
        "n_prev_actions": config.N_PREV,
        "n_action_states": config.N_ACTION_STATES,
        "token_grid": [config.GRID_H, config.GRID_W],
        "n_tokens": config.N_TOKENS,
    }


def check_stem_config(model_dir):
    """Raise if <model_dir>/stem_config.json disagrees with the config.py geometry.

    config.py is what the recorder, the dataset and the env actually produce, so a model
    stamped with a different geometry has nothing that can feed it. Compares every key
    of stem_config() present in the stamp — including the ones ConvStem never reads
    (it is fully convolutional, so a resolution mismatch would otherwise load cleanly
    and merely score worse). Unstamped dirs pass unchecked.
    """
    path = os.path.join(model_dir, "stem_config.json")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        stamped = json.load(f)
    diff = {k: (v, stamped[k]) for k, v in stem_config().items() if k in stamped and stamped[k] != v}
    if diff:
        rows = "\n".join(f"  {k}: model {got}, config.py {want}" for k, (want, got) in diff.items())
        raise ValueError(
            f"{path} does not match the config.py geometry:\n{rows}\n"
            "Use a model built with the current geometry, or change config.py to match this "
            "one — but every recorded dataset and every other checkpoint is tied to config.py, "
            "so they have to be re-made too."
        )
