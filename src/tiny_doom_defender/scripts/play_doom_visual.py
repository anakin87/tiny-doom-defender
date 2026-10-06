"""
Watch a trained policy play defend_the_center in a LIVE DOOM window.

What you see is fed to the policy through the same observation pipeline eval-model
scores. Playback is paced to ~real time; the terminal prints HP / kills / ammo, the
chosen action, and the per-axis probabilities behind it.

Usage:
  play-doom --ckpt output/cnn-ppo/policy_best
  play-doom --ckpt tiny-doom-defender --overlay --episodes 1 --hold-final
  play-doom --ckpt output/cnn-ppo/policy_best --seed 10000 --episodes 1 --fps 20 --mode sampled
"""

import argparse
import math
import time
from collections import Counter

import numpy as np
import torch
import vizdoom

from tiny_doom_defender.constants import FRAME_SKIP, TICS_PER_SECOND
from tiny_doom_defender.env import DefendCenterConvStemEnv
from tiny_doom_defender.evaluation import build_policy

TURN_NAMES = ("left", "none", "right")


@torch.no_grad()
def decide(policy, obs, device, mode):
    """One observation -> (turn, shoot, turn_probs, shoot_probs)."""
    turn_p, shoot_p = policy.action_probs(torch.from_numpy(obs[None]).to(device))
    turn_p, shoot_p = turn_p[0].cpu(), shoot_p[0].cpu()
    if mode == "argmax":
        turn, shoot = int(turn_p.argmax()), int(shoot_p.argmax())
    else:
        turn, shoot = int(torch.multinomial(turn_p, 1)), int(torch.multinomial(shoot_p, 1))
    return turn, shoot, turn_p.numpy(), shoot_p.numpy()


class OverlayViewer:
    """Render a separate copy of the RGB game frame, preserving its aspect ratio."""

    def __init__(self):
        import pygame

        self.pg = pygame
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((960, 720), pygame.RESIZABLE)
        pygame.display.set_caption("Tiny DOOM Defender — Esc / Q to exit")
        self.font = pygame.font.Font(None, 44)
        self.small_font = pygame.font.Font(None, 24)
        self.frame = None
        self.kills = 0
        self.episode = 1
        self.finished = False

    def draw(self, frame, kills, episode, *, finished=False):
        if frame is not None:
            # frombytes owns its pixel data; drawing cannot mutate the observation.
            height, width = frame.shape[:2]
            self.frame = self.pg.image.frombytes(frame.tobytes(), (width, height), "RGB")
        self.kills, self.episode, self.finished = kills, episode, finished
        self._paint()

    def _paint(self):
        if self.frame is None:
            return
        pg = self.pg
        scene = self.frame.copy()
        label = self.font.render(f"KILLS: {self.kills:.0f}", True, (255, 226, 120))
        panel = pg.Rect(scene.get_width() - label.get_width() - 32, 12, label.get_width() + 20, 54)
        pg.draw.rect(scene, (18, 18, 18), panel, border_radius=5)
        scene.blit(label, (panel.x + 10, panel.y + (panel.height - label.get_height()) // 2))
        if self.finished:
            text = self.small_font.render(f"EPISODE {self.episode} COMPLETE", True, (255, 255, 255))
            box = text.get_rect(center=scene.get_rect().center).inflate(24, 20)
            pg.draw.rect(scene, (18, 18, 18), box, border_radius=5)
            scene.blit(text, text.get_rect(center=box.center))
        width, height = self.window.get_size()
        scale = min(width / scene.get_width(), height / scene.get_height())
        size = (max(1, round(scene.get_width() * scale)), max(1, round(scene.get_height() * scale)))
        scaled = pg.transform.scale(scene, size)
        self.window.fill((0, 0, 0))
        self.window.blit(scaled, scaled.get_rect(center=self.window.get_rect().center))
        pg.display.flip()

    def poll(self):
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT or (
                event.type == self.pg.KEYDOWN and event.key in (self.pg.K_ESCAPE, self.pg.K_q)
            ):
                return False
            if event.type == self.pg.VIDEORESIZE:
                self._paint()
        return True

    def wait_until(self, deadline):
        while self.poll():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return True
            time.sleep(min(remaining, 0.01))
        return False

    def close(self):
        self.pg.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="output/cnn-ppo/policy_best", help="SFT dir (best/) or a PPO snapshot dir.")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="First episode seed, then seed+1, ... (held-out TEST seeds are 10000+). Omit for random.",
    )
    ap.add_argument("--mode", default="argmax", choices=["argmax", "sampled"])
    ap.add_argument("--fps", type=float, default=35.0, help="Playback pace (35=real time, 20=slow-mo, 60+=fast).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--print-every", type=int, default=10, help="HUD print cadence (steps).")
    ap.add_argument(
        "--overlay", action="store_true", help="Show the classic HUD with a large spectator-only kill counter."
    )
    ap.add_argument(
        "--hold-final", action="store_true", help="With --overlay, keep the final frame until you close it."
    )
    args = ap.parse_args()
    if args.hold_final and not args.overlay:
        ap.error("--hold-final requires --overlay")
    if args.episodes < 1 or args.print_every < 1:
        ap.error("--episodes and --print-every must be at least 1")
    if not math.isfinite(args.fps) or args.fps <= 0:
        ap.error("--fps must be a finite number greater than zero")

    device = torch.device(args.device)
    print(f"Loading policy {args.ckpt}...")
    policy = build_policy(args.ckpt, device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {n_params:,} params  |  mode={args.mode}")

    print("\nStarting DOOM (defend_the_center) — a window should open...")
    env = DefendCenterConvStemEnv(visible=not args.overlay)
    step_interval = FRAME_SKIP / max(args.fps, 1e-6)  # wall-seconds per decision

    frags_hist = []
    viewer = None
    try:
        viewer = OverlayViewer() if args.overlay else None
        for ep in range(args.episodes):
            if viewer and not viewer.poll():
                break
            seed = args.seed + ep if args.seed is not None else None
            obs, _ = env.reset(seed=seed)
            if viewer:
                viewer.draw(env.game.get_state().screen_buffer, 0, ep + 1)
            seed_str = f"seed {seed}" if seed is not None else "random seed"
            print(f"\n{'=' * 64}\nEpisode {ep + 1}/{args.episodes}  ({seed_str})\n{'=' * 64}")

            step, done, info = 0, False, {}
            counter = Counter()
            while not done:
                if viewer and not viewer.poll():
                    return
                t0 = time.perf_counter()
                turn, shoot, turn_p, shoot_p = decide(policy, obs, device, args.mode)
                act_name = f"{TURN_NAMES[turn]}{'+shoot' if shoot else ''}"
                counter[act_name] += 1

                if step % args.print_every == 0:
                    g = env.game
                    print(
                        f"  step {step:4d} | "
                        f"HP={g.get_game_variable(vizdoom.GameVariable.HEALTH):3.0f} "
                        f"K={g.get_game_variable(vizdoom.GameVariable.KILLCOUNT):2.0f} "
                        f"ammo={g.get_game_variable(vizdoom.GameVariable.AMMO2):2.0f} | "
                        f"{act_name:12s} | turn L/N/R={turn_p[0]:.2f}/{turn_p[1]:.2f}/{turn_p[2]:.2f} "
                        f"shoot={shoot_p[1]:.2f}"
                    )

                obs, _, term, trunc, info = env.step(np.array([turn, shoot], dtype=np.int32))
                done = term or trunc
                step += 1

                if viewer:
                    state = env.game.get_state()
                    kills = info["killcount"] if done else env.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)
                    viewer.draw(None if state is None else state.screen_buffer, kills, ep + 1, finished=done)
                    if not done and not viewer.wait_until(t0 + step_interval):
                        return
                else:
                    dt = time.perf_counter() - t0  # pace to ~real time
                    if dt < step_interval:
                        time.sleep(step_interval - dt)

            frags = info.get("killcount", float("nan"))
            bullets = info.get("bullets", float("nan"))
            frags_hist.append(frags)
            acc = frags / bullets if bullets else float("nan")
            surv = env.game.get_episode_time() / TICS_PER_SECOND
            print(
                f"\n  --- Episode {ep + 1}: frags={frags:.0f}  bullets={bullets:.0f}  "
                f"acc={acc:.3f}  survived={surv:.1f}s ---"
            )
            for name, c in counter.most_common():
                pct = 100 * c / max(step, 1)
                print(f"    {name:12s}: {c:4d} ({pct:4.1f}%)  {'#' * int(pct / 3)}")
            if viewer:
                hold = args.hold_final and ep == args.episodes - 1
                if hold:
                    print("Episode complete. Press Esc / Q or close the window to exit.", flush=True)
                if not viewer.wait_until(math.inf if hold else time.perf_counter() + 2.0):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        if viewer:
            viewer.close()

    if frags_hist:
        arr = np.array(frags_hist, dtype=float)
        print(f"\n{'=' * 64}\n{len(arr)} episodes: frags mean={arr.mean():.2f} [{arr.min():.0f}, {arr.max():.0f}]")


if __name__ == "__main__":
    main()
