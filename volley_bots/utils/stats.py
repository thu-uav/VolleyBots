import io

import numpy as np
import torch
from PIL import Image



class RangeIntegerTracker:
    def __init__(self, low: int, high: int) -> None:
        assert high - low > 0
        self.low = low
        self.high = high
        self.buffer = torch.zeros(high - low)

    def reset(self):
        self.buffer = 0

    def update(self, t: torch.Tensor):
        t = t.long().flatten().to(self.buffer.device)
        output, counts = torch.unique(t, return_counts=True)
        self.buffer[output - self.low] = self.buffer[output - self.low] + counts

    def _tight(self):
        nonzero_indices = self.buffer.nonzero().squeeze()
        self.low = torch.min(nonzero_indices).item()
        self.high = torch.max(nonzero_indices).item()
        self.buffer = self.buffer[self.low : self.high + 1]

    def save(self, path: str):
        nonzero_indices = self.buffer.nonzero().squeeze()
        low = torch.min(nonzero_indices)
        high = torch.max(nonzero_indices)
        d = {"buffer": self.buffer[low : high + 1], "low": low, "high": high}
        torch.save(d, path)

    @classmethod
    def load(self, path: str):
        d = torch.load(path)
        t = RangeIntegerTracker(d["low"], d["high"])
        t.buffer = d["buffer"]
        return t

    def numpy_histogram(self):
        self._tight()
        hist = self.buffer.numpy()
        bin_edges = np.arange(self.low, self.high + 2)
        return hist, bin_edges

    def _mean(self):
        score = np.arange(self.low, self.high + 1)
        cnt = self.buffer.numpy()
        return (score * cnt).sum() / cnt.sum()

    def matplotlib_bar(self):
        self._tight()
        hist, bin_edges = self.numpy_histogram()
        buf = io.BytesIO()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.bar(bin_edges[:-1], hist, width=1, align="edge")
        ax.set_title(f"MEAN:{self._mean():.2f}")
        fig.savefig(buf)

        return Image.open(buf)


def get_rangeIntegerTracker_process_and_save_func(low: int, high: int):
    tracker = RangeIntegerTracker(low, high)

    def process_func(x: torch.Tensor) -> torch.Tensor:
        tracker.update(x)
        return torch.mean(x.float()).item()

    def save_func(path: str):
        tracker.save(path)

    return process_func, save_func


def _get_first_episode_process_and_save_func(low: int, high: int):
    tracker = RangeIntegerTracker(low, high)

    def process_func(x: torch.Tensor) -> torch.Tensor:
        tmp = x[:, 0]
        is_first_episode = x[:, 1].bool()
        first_episode_x = tmp[is_first_episode]
        if len(first_episode_x) > 0:
            tracker.update(first_episode_x)

        # used only in eval.py for debugging. log_func is None. No need to return

    def save_func(path: str):
        tracker.save(path)

    return process_func, save_func





def episode_average_process_func(x: torch.Tensor) -> torch.Tensor:
    a = torch.sum(x[..., 0])
    b = torch.sum(x[..., 1])
    tmp = (a / b).item()
    return tmp


def episode_average_except_0_process_func(x: torch.Tensor) -> torch.Tensor:
    x = x.view(-1, 2)
    mask = x[:, 1] != 0
    x = x[mask]
    a = torch.sum(x[..., 0])
    b = torch.sum(x[..., 1])
    tmp = (a / b).item()
    return tmp


def mask_process_func(x: torch.Tensor) -> torch.Tensor:
    input = x[..., 0]
    mask = x[..., 1].bool()
    tmp = input[mask]
    return tmp.mean().item()


PROCESS_FUNC = {
    ("stats", "angular_deviation"): episode_average_process_func,
    ("stats", "dist_anchor"): episode_average_process_func,
    ("stats", "ball_vxy"): episode_average_process_func,
    ("stats", "ball_vz"): episode_average_process_func,
    ("stats", "ball_height"): episode_average_process_func,
    ("stats", "abs_x"): episode_average_process_func,
    ("stats", "frame_per_hit"): episode_average_except_0_process_func,
    ("stats", "len_no_wrong"): mask_process_func,
    ("stats", "step_reward"): episode_average_process_func,
    ("stats", "yaw"): episode_average_process_func,
    ("stats", "ball_vz"): episode_average_except_0_process_func,
    ("stats", "drone_z_on_hit"): episode_average_except_0_process_func,
}
