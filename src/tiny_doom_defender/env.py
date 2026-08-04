"""How the model sees DOOM: game.py's raw interface wrapped as a Gymnasium env that
produces the flat uint8 observation the policy consumes."""

from collections import deque

import gymnasium as gym
import numpy as np
import vizdoom

from tiny_doom_defender.constants import FRAME_SKIP, N_FRAMES, N_PREV, OBS_LEN, START_ACTION
from tiny_doom_defender.game import screen_to_frame, setup_game
from tiny_doom_defender.utils import combine_action, pack_obs, stack_channels


class DefendCenterConvStemEnv(gym.Env):
    """defend_the_center with MultiDiscrete([3, 2]) actions and a vision-only obs.

    Holds the rolling buffers behind the (OBS_LEN,) observation: the last N_FRAMES frames
    and the last N_PREV actions. The game comes from game.setup_game, so resolution,
    screen format and button order match the recorder. `info` at episode end carries
    killcount plus the bullets/hits behind the accuracy diagnostic.

    visible=True opens a real DOOM window — for watching a policy play, not for training.
    """

    metadata = {"render_modes": []}

    def __init__(self, visible=False):
        super().__init__()
        self._game = None  # lazy init in reset() so subprocess spawn stays cheap
        self._visible = visible
        self._start_ammo = float("nan")  # AMMO2 at reset, for bullets-fired accounting
        self._frame_buf = deque(maxlen=N_FRAMES)
        self._act_hist = deque(maxlen=N_PREV)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(OBS_LEN,), dtype=np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([3, 2])

    @property
    def game(self):
        """The live VizDoom game, None before the first reset. For spectating tools that
        need HUD variables mid-episode."""
        return self._game

    def _ensure_game(self):
        if self._game is None:
            self._game = setup_game(visible=self._visible)

    def _prev_list(self):
        # [older ... newer], left-padded with START when history is short.
        h = list(self._act_hist)
        return [START_ACTION] * (N_PREV - len(h)) + h

    def _obs(self):
        return pack_obs(stack_channels(self._frame_buf), self._prev_list())

    def _push_frame(self, frame):
        # Keep the buffer at exactly N_FRAMES: when short (fresh episode) fill it by
        # repeating this frame — the same rule the recorded dataset uses at episode start.
        if len(self._frame_buf) < N_FRAMES:
            self._frame_buf.clear()
            for _ in range(N_FRAMES):
                self._frame_buf.append(frame)
        else:
            self._frame_buf.append(frame)

    def _terminal_obs(self):
        return np.zeros(OBS_LEN, dtype=np.uint8)

    def _episode_info(self):
        info = {}
        try:
            info["killcount"] = float(self._game.get_game_variable(vizdoom.GameVariable.KILLCOUNT))
        except Exception:
            info["killcount"] = float("nan")
        # Accuracy accounting: bullets fired = start AMMO2 - final AMMO2, hits = HITCOUNT.
        try:
            info["bullets"] = self._start_ammo - float(self._game.get_game_variable(vizdoom.GameVariable.AMMO2))
            info["hits"] = float(self._game.get_game_variable(vizdoom.GameVariable.HITCOUNT))
        except Exception:
            info["bullets"] = float("nan")
            info["hits"] = float("nan")
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_game()
        if seed is not None:
            self._game.set_seed(int(seed) % (2**31 - 1))
        self._game.new_episode()
        self._act_hist.clear()
        self._frame_buf.clear()  # fresh episode: no carry-over frames
        try:
            self._start_ammo = float(self._game.get_game_variable(vizdoom.GameVariable.AMMO2))
        except Exception:
            self._start_ammo = float("nan")
        state = self._game.get_state()
        if state is None:
            return self._terminal_obs(), {}
        self._push_frame(screen_to_frame(state.screen_buffer))  # -> N_FRAMES copies
        return self._obs(), {}

    def step(self, action):
        turn_a, shoot_a = int(action[0]), int(action[1])
        buttons = [shoot_a == 1, turn_a == 0, turn_a == 2]  # [ATTACK, TURN_LEFT, TURN_RIGHT]
        reward = float(self._game.make_action(buttons, FRAME_SKIP))
        self._act_hist.append(combine_action(turn_a, shoot_a))
        if self._game.is_episode_finished():
            return self._terminal_obs(), reward, True, False, self._episode_info()
        state = self._game.get_state()
        if state is None:
            return self._terminal_obs(), reward, True, False, self._episode_info()
        self._push_frame(screen_to_frame(state.screen_buffer))
        return self._obs(), reward, False, False, {}

    def close(self):
        if self._game is not None:
            try:
                self._game.close()
            except Exception:
                pass
            self._game = None


def make_env():
    """Thunk for gymnasium.vector.AsyncVectorEnv (PPO rollout collection)."""

    def _thunk():
        return DefendCenterConvStemEnv()

    return _thunk
