import numpy as np
import torch

from tiny_doom_defender import constants


def pick_device(device_arg="auto", parallel=False):
    """'auto' -> cuda/mps/cpu, but cpu when parallel: worker processes would contend on the
    single accelerator. Pinning the device also keeps scores comparable across runs (MPS
    and CPU differ in float and RNG)."""
    if device_arg != "auto":
        return torch.device(device_arg)
    if parallel:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    obs = np.empty(constants.OBS_LEN, dtype=np.uint8)
    obs[: constants.STACK_PIXELS] = stack_u8.reshape(-1)
    obs[constants.STACK_PIXELS :] = np.asarray(prev_actions, dtype=np.uint8)
    return obs


def check_pipeline_config(model_config):
    """Raise if a model's input geometry disagrees with constants.py — what the env,
    recorder and dataset actually produce. The stem is fully convolutional, so a
    mismatch would otherwise load cleanly and merely score worse."""
    expected = {
        "res_h": constants.RES_H,
        "res_w": constants.RES_W,
        "n_frames": constants.N_FRAMES,
        "n_prev_actions": constants.N_PREV,
        "n_action_states": constants.N_ACTION_STATES,
    }
    diff = {k: (want, getattr(model_config, k)) for k, want in expected.items() if getattr(model_config, k) != want}
    if diff:
        rows = "\n".join(f"  {k}: model {got}, constants.py {want}" for k, (want, got) in diff.items())
        raise ValueError(
            f"Model geometry does not match constants.py:\n{rows}\n"
            "Use a model built with the current geometry, or change constants.py."
        )
