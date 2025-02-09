import abc
from typing import List

import numpy as np
from scipy.optimize import linprog


class MetaSolver(abc.ABC):

    @abc.abstractmethod
    def solve(self, payoffs: List[np.ndarray]) -> List[np.ndarray]:
        """Given a payoff tensor, calculate the meta-strategy for each player.
        Args:
            payoffs: List[np.ndarray]:
                payoffs[i]: np.ndarry, i is the index of player, payoffs[i][j][k], j is the index of player 0's policy, k is the index of player 1's policy.
        Returns:
            List[np.ndarray]: list of meta-strategies, i.e. policy sample probabilities.
        """
        raise NotImplementedError()


class NaiveSPSolver(MetaSolver):
    """
    Use the lastest policy.
    """

    def solve(self, payoffs):
        meta_strategy_list = []
        for payoff in payoffs:
            oppo_population_size = payoff.shape[1]
            prob = np.zeros(oppo_population_size)
            prob[-1] = 1.0
            meta_strategy_list.append(prob)
        return meta_strategy_list


class FSPSolver(MetaSolver):
    """
    Use uniform sample probability as the meta-strategy for each player.
    """

    def solve(self, payoffs):
        meta_strategy_list = []
        for payoff in payoffs:
            oppo_population_size = payoff.shape[1]
            meta_strategy_list.append(
                np.ones(oppo_population_size) / oppo_population_size
            )
        return meta_strategy_list


class DeltaUniformSPSolver(MetaSolver):

    def __init__(self, delta: float = 0.5):
        self.delta = delta
        if delta < 0 or delta > 1:
            raise ValueError(f"Delta should be in [0, 1], but got {delta}")

    def solve(self, payoffs):
        meta_strategy_list = []
        for payoff in payoffs:
            oppo_population_size = payoff.shape[1]
            temp_size = int(oppo_population_size * self.delta)
            n = oppo_population_size - temp_size
            prob = np.zeros(oppo_population_size)
            if n == 0:
                prob[-1] = 1.0
            else:
                prob[-n:] = 1.0 / n
            meta_strategy_list.append(prob)
        return meta_strategy_list

    def set_delta(self, delta: float):
        self.delta = delta
        if delta < 0 or delta > 1:
            raise ValueError(f"Delta should be in [0, 1], but got {delta}")


class NashSolver(MetaSolver):
    """Use Nash equilibrium as the meta-strategy for each player. A meta-game is built using the empirical
    payoff tensor, and each policy in the population is a strategy for the player. Then Nash equilibrium
    gives the mixed strategy that no player can benefit from changing his meta-strategy unilaterally.

    NOTE: Currently Nash solver only supports 2-player zero-sum game. For other settings like multi-player
    or general-sum game, the complexity to compute Nash equilibrium is too high, use approximate solvers
    instead.
    """

    def solve(self, payoffs):
        assert (
            len(payoffs) == 2
        ), f"Currently Nash solver only supports 2-player zero-sum game, but got {payoffs.shape[0]} players."
        assert (payoffs[0] + payoffs[1].T == 0).all(), "The game is not zero-sum."

        num_player = len(payoffs)
        payoff_matrix = payoffs[0]
        num_strategies_player1, num_strategies_player2 = payoff_matrix.shape

        # Objective function: maximize v (the value of the game)
        # Since linprog does minimization, we minimize -v
        c = np.zeros(num_strategies_player1 + 1)
        c[-1] = -1  # We want to maximize v, so we minimize -v
        # Constraints for the linear program
        # -payoff_matrix.T @ x + v <= 0
        A_ub = np.hstack((-payoff_matrix.T, np.ones((num_strategies_player2, 1))))
        b_ub = np.zeros(num_strategies_player2)
        # Constraint that probabilities sum to 1
        A_eq = np.ones((1, num_strategies_player1 + 1))
        A_eq[0, -1] = 0  # We do not want to sum v into this constraint
        b_eq = np.array([1])
        # Bounds for the probabilities and game value
        bounds = [(0, 1) for _ in range(num_strategies_player1)] + [(None, None)]
        # Solve the linear program
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method="highs")
        # Extract the mixed strategy and the value of the game
        mixed_strategy_player1 = res.x[:-1]
        game_value = -res.x[-1]

        # Solve for player 2's strategy by using the negative transpose of the payoff matrix
        c2 = np.zeros(num_strategies_player2 + 1)
        c2[-1] = -1
        A_ub2 = np.hstack((payoff_matrix, np.ones((num_strategies_player1, 1))))
        b_ub2 = np.zeros(num_strategies_player1)
        A_eq2 = np.ones((1, num_strategies_player2 + 1))
        A_eq2[0, -1] = 0
        b_eq2 = np.array([1])
        bounds2 = [(0, 1) for _ in range(num_strategies_player2)] + [(None, None)]
        res2 = linprog(c2, A_ub2, b_ub2, A_eq2, b_eq2, bounds2, method="highs")
        mixed_strategy_player2 = res2.x[:-1]

        meta_strategy_list = [mixed_strategy_player1, mixed_strategy_player2]

        return meta_strategy_list


def get_meta_solver(name: str) -> MetaSolver:
    d = {
        "fsp": FSPSolver,
        "nash": NashSolver,
        "naive": NaiveSPSolver,
        "delta_uniform": DeltaUniformSPSolver,
    }
    return d[name]()
