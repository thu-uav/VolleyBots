from typing import Any, Callable, Dict

import numpy as np
import torch
import wandb
from tensordict import TensorDict
from torchrl.envs.transforms import TransformedEnv

from .psro import PSROPolicy


class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


@torch.no_grad()
def evaluate(env: TransformedEnv, policy: PSROPolicy):
    frames = []

    def record_frame(*args, **kwargs):
        frame = env.base_env.render(mode="rgb_array")
        frames.append(frame)

    _should_render = env.base_env._should_render
    env.base_env.enable_render(True)
    env.reset()
    env.eval()

    env.rollout(
        max_steps=env.base_env.max_episode_length,
        policy=policy,
        callback=Every(record_frame, 2),
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )
    env.base_env.enable_render(_should_render)
    env.reset()
    env.train()

    info = {}

    if len(frames):
        # video_array = torch.stack(frames)
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        info["recording"] = wandb.Video(
            video_array, fps=0.5 / env.base_env.dt, format="mp4"
        )
    frames.clear()

    return info
