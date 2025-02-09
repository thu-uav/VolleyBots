import datetime
import logging
import os
import sys

import hydra
import numpy as np
import torch

np.set_printoptions(threshold=sys.maxsize)
import traceback
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from setproctitle import setproctitle
from tensordict import TensorDict
from torch import vmap
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from tqdm import tqdm

import omni_drones.utils.volleyball.elopy as elopy
from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.learning import PSROPolicy
from omni_drones.utils.psro.convergence import ConvergedIndicator
from omni_drones.utils.psro.meta_solver import get_meta_solver
from omni_drones.utils.torchrl import AgentSpec, SyncDataCollector
from omni_drones.utils.wandb import init_wandb


class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


def get_policy(cfg: DictConfig, agent_spec: AgentSpec):
    algos = {
        "psro": PSROPolicy,
    }
    algo_name = cfg.algo.name.lower()
    if algo_name not in algos:
        raise RuntimeError(f"{algo_name} not supported.")
    policy = algos[algo_name](cfg.algo, agent_spec=agent_spec, device="cuda")
    return policy


def get_transforms(
    cfg: DictConfig, base_env, logger_func: Callable[[Dict], None] = None
):
    from omni_drones.utils.torchrl.transforms import LogOnEpisode

    stats_keys = [
        k
        for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    logger = LogOnEpisode(
        cfg.env.num_envs,
        in_keys=stats_keys,
        log_keys=stats_keys,
        logger_func=logger_func,
        process_func=None,
    )
    transforms = [InitTracker(), logger]

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is None:
        pass
    elif action_transform == "rate":
        from omni_drones.controllers import RateController as _RateController
        from omni_drones.utils.torchrl.transforms import RateController

        controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
        transform = RateController(controller)
        transforms.append(transform)
    elif not action_transform.lower() == "none":
        raise NotImplementedError(f"Unknown action transform: {action_transform}")

    return transforms


@torch.no_grad()
def eval_win_rate_once(env: TransformedEnv, policy: PSROPolicy):
    env.reset()
    env.eval()
    policy.eval_payoff = True
    td = env.rollout(
        max_steps=env.base_env.max_episode_length,
        policy=policy,
        auto_reset=True,
        break_when_any_done=True,
        return_contiguous=True,
    )
    env.reset()
    env.train()
    policy.eval_payoff = False

    done: torch.Tensor = td["next", "done"].squeeze(-1).bool()  # (E, max_episode_len,)
    actor_0_wins: torch.Tensor = td["next", "stats", "actor_0_wins"].squeeze(
        -1
    )  # (E,max_episode_len,)
    actor_1_wins: torch.Tensor = td["next", "stats", "actor_1_wins"].squeeze(
        -1
    )  # (E,max_episode_len,)

    win = actor_0_wins[done].sum().item()
    lose = actor_1_wins[done].sum().item()
    draw = 1 - win - lose
    assert win + lose + draw == 1
    assert win == 0 or win == 1
    assert lose == 0 or lose == 1
    assert draw == 0 or draw == 1

    return win, draw, lose


def train(
    cfg: DictConfig, simulation_app: SimulationApp, env: TransformedEnv, wandb_run
):
    agent_spec: AgentSpec = env.agent_spec["drone"]

    num_games = 1000
    logging.info(f"num_games:{num_games}")
    num_wins = 0
    num_draws = 0
    num_loses = 0

    # initiate elo rating
    team_0_init_elo = 1000
    team_1_init_elo = 1000
    logging.info(f"team_0_init_elo:{team_0_init_elo}")
    logging.info(f"team_1_init_elo:{team_1_init_elo}")

    elo_system = elopy.Implementation()
    elo_system.addPlayer("team_0", team_0_init_elo)
    elo_system.addPlayer("team_1", team_1_init_elo)

    # initiate PSRO policy
    policy: PSROPolicy = get_policy(
        cfg, agent_spec
    )  # each population has a random initial policy

    append_actors_0_from_path = cfg.get("append_actors_0_from_path")
    append_actors_1_from_path = cfg.get("append_actors_1_from_path")
    assert append_actors_0_from_path is not None
    assert append_actors_1_from_path is not None

    if append_actors_0_from_path and append_actors_1_from_path:
        file_names = sorted(
            os.listdir(append_actors_0_from_path), key=lambda x: int(x.split(".")[0])
        )
        # file_names = os.listdir(append_actors_0_from_path)
        file_paths = [
            os.path.join(append_actors_0_from_path, f)
            for f in file_names
            if os.path.isfile(os.path.join(append_actors_0_from_path, f))
        ]
        for file_path in file_paths:
            actor_dict_0 = torch.load(file_path)
            policy.population_0.add_actor(actor_dict_0)
            logging.info(f"Append actor_dict_0 from {file_path}")

        file_names = sorted(
            os.listdir(append_actors_1_from_path), key=lambda x: int(x.split(".")[0])
        )
        # file_names = os.listdir(append_actors_1_from_path)
        file_paths = [
            os.path.join(append_actors_1_from_path, f)
            for f in file_names
            if os.path.isfile(os.path.join(append_actors_1_from_path, f))
        ]
        for file_path in file_paths:
            actor_dict_1 = torch.load(file_path)
            policy.population_1.add_actor(actor_dict_1)
            logging.info(f"Append actor_dict_1 from {file_path}")

    if cfg.get("load_meta_policy_0_path") is not None:
        meta_policy_0_dict = np.load(cfg.get("load_meta_policy_0_path"))
        meta_policy_0 = meta_policy_0_dict["meta_policy_0"]
        assert meta_policy_0.shape[0] == len(policy.population_0)
        logging.info(f"Load meta-policy 0 from {cfg.get('load_meta_policy_0_path')}")
    else:
        meta_policy_0 = np.ones(len(policy.population_0)) / len(policy.population_0)
        logging.info(f"meta-policy 0 is initialized as uniform distribution")

    if cfg.get("load_meta_policy_1_path") is not None:
        meta_policy_1_dict = np.load(cfg.get("load_meta_policy_1_path"))
        meta_policy_1 = meta_policy_1_dict["meta_policy_1"]
        assert meta_policy_1.shape[0] == len(policy.population_1)
        logging.info(f"Load meta-policy 1 from {cfg.get('load_meta_policy_1_path')}")
    else:
        meta_policy_1 = np.ones(len(policy.population_1)) / len(policy.population_1)
        logging.info(f"meta-policy 1 is initialized as uniform distribution")

    progress_bar = tqdm(range(num_games), desc="Simulating games")
    for i in progress_bar:
        policy.both_sample_pure_strategy(meta_policy_0, meta_policy_1)
        win, draw, lose = eval_win_rate_once(env=env, policy=policy)
        if win:
            elo_system.recordMatch("team_0", "team_1", winner="team_0")
            num_wins += 1
            current_winner = "team_0"
        elif lose:
            elo_system.recordMatch("team_0", "team_1", winner="team_1")
            num_loses += 1
            current_winner = "team_1"
        else:
            elo_system.recordMatch("team_0", "team_1", draw=True)
            num_draws += 1
            current_winner = "Draw"
        team_0_elo = elo_system.getPlayerRating("team_0")
        team_1_elo = elo_system.getPlayerRating("team_1")

        progress_bar.set_postfix(
            {
                "Current Winner": current_winner,
                "team_0_Elo": round(team_0_elo, 2),
                "team_1_Elo": round(team_1_elo, 2),
            }
        )

    win_rate = num_wins / (num_wins + num_loses)
    logging.info(f"team_0_win_rate:{win_rate}")
    logging.info(f"team_0_final_elo: {elo_system.getPlayerRating('team_0')}")
    logging.info(f"team_1_final_elo: {elo_system.getPlayerRating('team_1')}")

    logging.info(f"Run URL:{wandb_run.get_url()}")


@hydra.main(
    version_base=None, config_path=CONFIG_PATH, config_name="train_psro_cross_play"
)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    run_name_suffix = cfg.get("run_name_suffix")
    run = init_wandb(cfg, run_name_suffix)
    setproctitle(run.name)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    def log(info):
        if cfg.wandb.get("mode", "disabled") != "online":
            tmp = {
                k: v
                for k, v in info.items()
                if k in ("train/stats.actor_0_wins", "train/stats.actor_1_wins")
            }
            print(OmegaConf.to_yaml(tmp))
        run.log(info)

    transforms = get_transforms(cfg, base_env, log)

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    train(cfg=cfg, simulation_app=simulation_app, env=env, wandb_run=run)

    wandb.save(os.path.join(run.dir, "*"))
    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
