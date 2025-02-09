import logging

import numpy as np


class ConvergedIndicator:
    def __init__(
        self,
        max_size: int = 10,
        mean_threshold: float = 0.7,
        std_threshold: float = 0.005,
        min_iter_steps: int = 50,
        max_iter_steps: int = 2000,
        player_id: int = 0,
    ) -> None:
        self.win_rates = []
        self.max_size = max_size
        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold
        self.min_iter_steps = min_iter_steps
        self.max_iter_steps = max_iter_steps
        self._step_cnt = 0  # count
        self.player_id = player_id

    def player_id_to_str(self):
        return f"Player {self.player_id}"

    def update(self, value: float):
        self.win_rates.append(value)
        self._step_cnt += 1
        if len(self.win_rates) > self.max_size:
            self.win_rates.pop(0)

    def reset(self, player_id: int):
        self.win_rates = []
        self._step_cnt = 0
        self.player_id = player_id

    def converged(self) -> bool:
        mean = np.mean(self.win_rates)
        std = np.std(self.win_rates)

        if self._step_cnt < 10 or self._step_cnt % 10 == 0:
            logging.info(
                f"{self.player_id_to_str()} Recent Win Rate: mean={mean:.2f}, std={std:.4f}, len={len(self.win_rates)}, step={self._step_cnt}"
            )

        if len(self.win_rates) < self.max_size:
            return False
        if self._step_cnt < self.min_iter_steps:
            return False
        if self._step_cnt > self.max_iter_steps:
            logging.info("ConvergedIndicator: Max Iter Steps Reached")
            logging.info(
                f"{self.player_id_to_str()} Recent Win Rate: mean={mean:.2f}, std={std:.4f}, len={len(self.win_rates)}, step={self._step_cnt}"
            )
            return True

        if mean >= self.mean_threshold and std <= self.std_threshold:
            logging.info("ConvergedIndicator: Converged")
            logging.info(
                f"{self.player_id_to_str()} Recent Win Rate: mean={mean:.2f}, std={std:.4f}, len={len(self.win_rates)}, step={self._step_cnt}"
            )
            return True
        else:
            return False
