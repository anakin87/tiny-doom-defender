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


def load_stem_config(model_dir):
    """Conv-stem geometry the encoder was built with, from <model_dir>/stem_config.json.

    Falls back to the config.py constants when the sidecar is absent (older model
    dirs), so a checkpoint stays loadable without it.
    """
    path = os.path.join(model_dir, "stem_config.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {
        "input_resolution": [config.RES_H, config.RES_W],
        "n_frames": config.N_FRAMES,
        "in_channels": config.IN_CH,
        "n_prev_actions": config.N_PREV,
        "n_action_states": config.N_ACTION_STATES,
        "token_grid": [config.GRID_H, config.GRID_W],
        "n_tokens": config.N_TOKENS,
    }
