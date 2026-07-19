"""VizDoom interface: open a scenario and downsample its screen to the model input."""

import numpy as np
import vizdoom
from PIL import Image

from tiny_doom_defender.config import EPISODE_TIMEOUT, RES_H, RES_W


def setup_game(scenario="defend_the_center", skill=None, episode_timeout=EPISODE_TIMEOUT, objects_info=False):
    """Headless PLAYER-mode game rendering full-res RGB, labels buffer on."""
    game = vizdoom.DoomGame()
    game.load_config(vizdoom.scenarios_path + f"/{scenario}.cfg")
    if skill is not None:
        game.set_doom_skill(skill)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)  # oracle decision input ONLY (never recorded)
    if objects_info:
        game.set_objects_info_enabled(True)  # privileged side-channel, never an observation
    game.set_window_visible(False)
    game.set_render_hud(True)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.clear_available_buttons()
    game.add_available_button(vizdoom.Button.ATTACK)  # 0
    game.add_available_button(vizdoom.Button.TURN_LEFT)  # 1
    game.add_available_button(vizdoom.Button.TURN_RIGHT)  # 2
    game.set_episode_timeout(episode_timeout)
    game.init()
    return game


def screen_to_frame(screen_buffer):
    """(480, 640, 3) uint8 RGB screen -> (RES_H, RES_W, 3) uint8, area-averaged with PIL BOX resampling."""
    img = Image.fromarray(screen_buffer)  # (H, W, 3)
    img = img.resize((RES_W, RES_H), Image.Resampling.BOX)  # (RES_W, RES_H)
    return np.asarray(img, dtype=np.uint8)  # (RES_H, RES_W, 3)
