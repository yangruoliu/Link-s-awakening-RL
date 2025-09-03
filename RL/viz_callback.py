import os
import json
import imageio
import numpy as np
from typing import List, Optional

from stable_baselines3.common.callbacks import BaseCallback


class StepRenderCallback(BaseCallback):
    """
    Collect rgb frames step-by-step during rollouts and save GIF/MP4 per episode.

    - Saves frames to output_dir/episode_{n}.gif
    - Annotates frames with step, action, reward if provided.
    """

    def __init__(
        self,
        output_dir: str = "videos",
        save_every_n_episodes: int = 1,
        max_frames_per_episode: Optional[int] = 2000,
        fps: int = 30,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.output_dir = output_dir
        self.save_every_n_episodes = save_every_n_episodes
        self.max_frames_per_episode = max_frames_per_episode
        self.fps = fps
        self.episode_idx = 0
        self.frames: List[np.ndarray] = []
        self.step_logs: List[dict] = []

        os.makedirs(self.output_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        self.episode_idx = 0
        self.frames = []
        self.step_logs = []

    def _on_rollout_start(self) -> None:
        # Clear frames at the start of each rollout segment
        self.frames = []

    def _on_step(self) -> bool:
        # Access the underlying env
        env = self.model.get_env()
        try:
            # VecEnv -> capture first sub-env's render
            if hasattr(env, "envs") and len(env.envs) > 0:
                frame = env.envs[0].render(mode="rgb_array")
            else:
                frame = env.render(mode="rgb_array")
        except Exception:
            # If env does not support rgb_array, skip
            frame = None

        if frame is not None:
            # Ensure uint8 HxWx3
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            elif frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            self.frames.append(frame)
            if self.max_frames_per_episode and len(self.frames) >= self.max_frames_per_episode:
                # Stop collecting too many frames within an episode
                pass

        # Try to capture step-level metadata if available
        actions = self.locals.get("actions", None)
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        step_log = {}
        try:
            if actions is not None:
                step_log["action"] = (actions[0].item() if hasattr(actions[0], "item") else int(actions[0]))
            if rewards is not None:
                step_log["reward"] = (rewards[0].item() if hasattr(rewards[0], "item") else float(rewards[0]))
            if infos is not None and len(infos) > 0 and isinstance(infos[0], dict):
                # include selected info keys if present
                for key in ["room", "goal"]:
                    if key in infos[0]:
                        step_log[key] = infos[0][key]
        except Exception:
            # be robust to structure changes
            pass
        if step_log:
            self.step_logs.append(step_log)

        # If the first env is done, flush current episode to disk
        try:
            done_flag = bool(dones[0]) if dones is not None else False
        except Exception:
            done_flag = False
        if done_flag:
            self._save_episode()
        return True

    def _on_rollout_end(self) -> None:
        # Episodes may span multiple rollouts; do nothing here unless user wants periodic dumps
        if (self.episode_idx % self.save_every_n_episodes) == 0 and len(self.frames) > 0 and self.save_every_n_episodes < 0:
            self._save_episode()

    def _on_training_end(self) -> None:
        # Flush any remaining frames as a final video
        if len(self.frames) > 0:
            # Save any partial episode at training end
            self._save_episode(suffix="_final")

    def _save_episode(self, suffix: str = "") -> None:
        if len(self.frames) == 0:
            return
        if (self.episode_idx % self.save_every_n_episodes) != 0:
            # Skip saving based on interval
            self.frames = []
            self.step_logs = []
            self.episode_idx += 1
            return
        outfile = os.path.join(self.output_dir, f"episode_{self.episode_idx:05d}{suffix}.gif")
        try:
            imageio.mimsave(outfile, self.frames, fps=self.fps)
            # Also write step logs alongside
            logs_path = os.path.join(self.output_dir, f"episode_{self.episode_idx:05d}{suffix}_steps.jsonl")
            with open(logs_path, "w", encoding="utf-8") as f:
                for entry in self.step_logs:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if self.verbose:
                print(f"Saved episode visualization to {outfile} ({len(self.frames)} frames)")
                print(f"Saved step logs to {logs_path} ({len(self.step_logs)} steps)")
        except Exception as e:
            if self.verbose:
                print(f"Failed to save visualization/logs: {e}")
        self.frames = []
        self.step_logs = []
        self.episode_idx += 1

