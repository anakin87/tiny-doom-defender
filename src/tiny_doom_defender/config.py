"""Shared constants for the conv-stem DOOM pipeline."""

# --- Conv-stem input geometry -------------------------------------------------
RES_W, RES_H = 160, 100  # conv-stem input resolution (pixels)
N_FRAMES = 3  # previous 2 frames + current  -> motion
IN_CH = N_FRAMES * 3  # 9 input channels (3 frames x RGB)

# Two stride-2 3x3 convs take 160x100 -> 40x25 exactly (each dim / 4).
GRID_H, GRID_W = RES_H // 4, RES_W // 4  # 25 x 40
N_TOKENS = GRID_H * GRID_W  # 1000  (== ModernBERT sequence length)

FRAME_HWC = (RES_H, RES_W, 3)  # single-frame storage shape, uint8
FRAME_PIXELS = RES_H * RES_W * 3  # 48,000  bytes per stored frame
STACK_PIXELS = IN_CH * RES_H * RES_W  # 144,000 bytes per 9-channel stack

# --- Actions ------------------------------------------------------------------
# MultiDiscrete([3, 2]) = turn(L/none/R) x shoot(no/yes).  No forward action —
# the player never moves, so the action space drops it.
# We flatten the 6 combos to a single index for the prev-action embedding, plus a
# START token used for episode-start gaps (and the repeated-frame gap at step 1).
N_ACTION_COMBOS = 3 * 2  # 6
START_ACTION = N_ACTION_COMBOS  # 6  ("no previous action")
N_ACTION_STATES = N_ACTION_COMBOS + 1  # 7

# One prev-action per inter-frame gap: N_FRAMES frames -> N_FRAMES-1 transitions.
# Each transition's ego-motion is explained by the action that caused it, so the
# model gets the last (N_FRAMES-1) actions, oldest-first: [a_{t-2}, a_{t-1}].
N_PREV = N_FRAMES - 1  # 2

# Flat observation for the PPO vector env: the raveled 9-ch stack followed by the
# N_PREV prev-action bytes. Kept as a single uint8 vector so the PPO rollout buffer
# stays one tensor (and small: uint8, not int64).
OBS_LEN = STACK_PIXELS + N_PREV

# --- VizDoom / eval ------------------------------------------------------------
FRAME_SKIP = 4  # action repeat (matched train+eval; VizDoom convention)
EPISODE_TIMEOUT = 2100
TICS_PER_SECOND = 35

# Seed pools (one role each, all disjoint).
SEED_ROLLOUT = 42  # PPO training rollouts
SEED_INLOOP_EVAL = 30000  # in-loop sanity eval (diagnostics only, never selection)
SEED_SELECTION = 50000  # offline snapshot rerank (acknowledged-biased)
SEED_TEST = 10000  # held-out test — the one comparable number
