import datetime
import logging
import os
import traceback
from typing import Callable, Dict, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from setproctitle import setproctitle
from tensordict import TensorDict
from torch import vmap
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from tqdm import tqdm

from volley_bots import CONFIG_PATH, init_simulation_app
from volley_bots.learning import PSROPolicy
from volley_bots.utils.psro.convergence import ConvergedIndicator
from volley_bots.utils.psro.meta_solver import get_meta_solver
from volley_bots.utils.torchrl import AgentSpec, SyncDataCollector
from volley_bots.utils.wandb import init_wandb


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
    """
    Use the latest strategies of both teams to evaluate the policy
    """
    logging.info("Evaluating the latest policy v.s. the latest policy.")
    frames = []
    info = {}

    def record_frame(*args, **kwargs):
        frame = env.base_env.render(mode="rgb_array")
        frames.append(frame)

    _should_render = env.base_env._should_render
    env.base_env.enable_render(True)
    env.reset()
    env.eval()
    policy.eval_payoff = True
    policy.set_latest_strategy()

    env.rollout(
        max_steps=env.base_env.max_episode_length,
        policy=policy,
        callback=Every(record_frame, 2),
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )

    if len(frames):
        # video_array = torch.stack(frames)
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        info["recording"] = wandb.Video(
            video_array, fps=0.5 / env.base_env.dt, format="mp4"
        )

    env.base_env.enable_render(_should_render)
    env.reset()
    env.train()
    policy.eval_payoff = False
    frames.clear()

    return info


@torch.no_grad()
def evaluate_share_population(env: TransformedEnv, policy: PSROPolicy):
    """
    Use the latest strategy and second latest strategy to evaluate the policy (Only used in share_population mode)
    """

    logging.info(
        "Evaluating the latest policy (player 1) v.s. the second latest policy (player 2)."
    )

    info = {}
    frames = []

    def record_frame(*args, **kwargs):
        frame = env.base_env.render(mode="rgb_array")
        frames.append(frame)

    _should_render = env.base_env._should_render
    env.base_env.enable_render(True)
    env.reset()
    env.eval()
    policy.eval_payoff = True

    policy.population_0.set_latest_policy()
    policy.population_1.set_second_latest_policy()

    env.rollout(
        max_steps=env.base_env.max_episode_length,
        policy=policy,
        callback=Every(record_frame, 2),
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )

    if len(frames):
        # video_array = torch.stack(frames)
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        info["recording_team_0_latest_vs_team_1_second_latest"] = wandb.Video(
            video_array, fps=0.5 / env.base_env.dt, format="mp4"
        )

    logging.info(
        "Evaluating the second latest policy (player 1) v.s. the latest policy (player 2)."
    )
    frames.clear()
    frames = []
    env.reset()

    policy.population_0.set_second_latest_policy()
    policy.population_1.set_latest_policy()

    env.rollout(
        max_steps=env.base_env.max_episode_length,
        policy=policy,
        callback=Every(record_frame, 2),
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )

    if len(frames):
        # video_array = torch.stack(frames)
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        info["recording_team_0_second_latest_vs_team_1_latest"] = wandb.Video(
            video_array, fps=0.5 / env.base_env.dt, format="mp4"
        )

    env.base_env.enable_render(_should_render)
    env.reset()
    env.train()
    policy.eval_payoff = False
    frames.clear()

    return info


def calculate_jpc(payoffs: np.ndarray):
    """
    Calculate JPC (Joint Payoff Convexity) of the game
    """
    assert len(payoffs.shape) == 2 and payoffs.shape[0] == payoffs.shape[1]
    n = payoffs.shape[0]
    assert n > 0
    d = np.trace(payoffs) / n
    o = (np.sum(payoffs) - n * d) / (n * (n - 1))
    r = (d - o) / d
    return r


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
    from volley_bots.utils.torchrl.transforms import LogOnEpisode

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
        from volley_bots.controllers import RateController as _RateController
        from volley_bots.utils.torchrl.transforms import RateController

        controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
        transform = RateController(controller)
        transforms.append(transform)
    elif not action_transform.lower() == "none":
        raise NotImplementedError(f"Unknown action transform: {action_transform}")

    return transforms


@torch.no_grad()
def eval_win_rate(env: TransformedEnv, policy: PSROPolicy):
    env.reset()
    env.eval()
    policy.eval_payoff = True
    td = env.rollout(
        max_steps=env.base_env.max_episode_length,
        policy=policy,
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=True,
    )
    env.reset()
    env.train()
    policy.eval_payoff = False

    done: torch.Tensor = td["stats", "done"].squeeze(-1).bool()  # (E, max_episode_len,)
    actor_0_wins: torch.Tensor = td["stats", "actor_0_wins"].squeeze(
        -1
    )  # (E,max_episode_len,)
    actor_1_wins: torch.Tensor = td["stats", "actor_1_wins"].squeeze(
        -1
    )  # (E,max_episode_len,)

    num_wins = actor_0_wins[done].sum().item()
    num_loses = actor_1_wins[done].sum().item()

    if num_wins + num_loses == 0:
        return 0.5

    return num_wins / (num_wins + num_loses)


def get_new_payoffs(
    env: TransformedEnv,
    policy: PSROPolicy,
    old_payoffs: Optional[np.ndarray],
):
    """
    compute missing payoff tensor entries via game simulations
    """
    assert len(policy.population_0) == len(policy.population_1)
    n = len(policy.population_0)
    new_payoffs = np.zeros(shape=(n, n))

    if old_payoffs is not None:
        assert (
            len(old_payoffs.shape) == 2
            and old_payoffs.shape[0] == old_payoffs.shape[1]
            and old_payoffs.shape[0] + 1 == n
        )
        new_payoffs[:-1, :-1] = old_payoffs

    for i in range(n):
        policy.set_pure_strategy(idx_0=n - 1, idx_1=i)
        wr = eval_win_rate(env=env, policy=policy)
        new_payoffs[-1, i] = wr - (1 - wr)

    for i in range(n - 1):
        policy.set_pure_strategy(idx_0=i, idx_1=n - 1)
        wr = eval_win_rate(env=env, policy=policy)
        new_payoffs[i, -1] = wr - (1 - wr)

    return new_payoffs


def payoffs_to_win_rate(payoffs: np.ndarray) -> np.ndarray:
    """
    Convert payoffs to win rate
    """
    assert len(payoffs.shape) == 2 and payoffs.shape[0] == payoffs.shape[1]
    win_rates = (payoffs + 1) / 2
    return win_rates


def log_heatmap(win_rate: np.ndarray):
    """
    Get heatmap of win rate
    """
    plt.figure(figsize=(12, 10))
    if win_rate.shape[0] < 5:
        hm = sns.heatmap(
            win_rate,
            annot=True,
            fmt=".3f",
            center=0.5,
            cmap="coolwarm",
            xticklabels=range(win_rate.shape[1]),
            yticklabels=range(win_rate.shape[0]),
        )
    else:
        hm = sns.heatmap(
            win_rate,
            annot=False,
            center=0.5,
            cmap="coolwarm",
            xticklabels=range(win_rate.shape[1]),
            yticklabels=range(win_rate.shape[0]),
        )
    hm.xaxis.tick_top()
    plt.title("Player 1 Win Rate")
    plt.xlabel("Player 2 Strategy Population")
    plt.ylabel("Player 1 Strategy Population")
    plt.savefig("heatmap.png")
    plt.close()
    wandb.log({"heatmap": wandb.Image("heatmap.png")})
    return None


def get_emprical_win_rate(data: TensorDict, win_key: str, lose_key: str) -> float:

    win = data["stats", win_key]
    lose = data["stats", lose_key]
    done = data["stats", "done"].bool()

    if done.sum().item() == 0:
        return None

    num_wins = win[done].sum().item()
    num_loses = lose[done].sum().item()

    if num_wins + num_loses == 0:
        return None

    return num_wins / (num_wins + num_loses)


def train(
    cfg: DictConfig, simulation_app: SimulationApp, env: TransformedEnv, wandb_run
):
    agent_spec: AgentSpec = env.agent_spec["drone"]

    init_by_latest_strategy = cfg.get("init_by_latest_strategy", False)
    share_population = cfg.get("share_population", False)
    max_population_size = cfg.get("max_population_size", False)

    # initiate meta-solver
    meta_solver = get_meta_solver(cfg.get("solver_type").lower())

    # initiate PSRO policy
    policy: PSROPolicy = get_policy(
        cfg, agent_spec
    )  # each population has a random initial policy
    use_whole_policy = cfg.get("policy_checkpoint_path")
    use_actor_params = cfg.get("actor_0_checkpoint_path") and cfg.get(
        "actor_1_checkpoint_path"
    )
    if use_whole_policy and use_actor_params:
        raise ValueError(
            "Cannot use both policy_checkpoint_path and actor_checkpoint_path."
        )
    if use_whole_policy:
        policy.load_state_dict(torch.load(cfg.policy_checkpoint_path))
        print(f"Load policy from {cfg.policy_checkpoint_path}")
    if use_actor_params:
        actor_dict_0 = torch.load(cfg.actor_0_checkpoint_path)
        actor_dict_1 = torch.load(cfg.actor_1_checkpoint_path)
        policy.load_actor_dict(actor_dict_0, actor_dict_1)
        print(f"Load actor_dict_0 from {cfg.actor_0_checkpoint_path}")
        print(f"Load actor_dict_1 from {cfg.actor_1_checkpoint_path}")
    if cfg.save_meta_policy:
        meta_policy_save_path = os.path.join(wandb.run.dir, "meta_policy")
        os.makedirs(meta_policy_save_path, exist_ok=True)
    policy.current_player_id = cfg.get("first_id", 0)  # 0 or 1

    # initiate converged indicator
    converged_indicator = ConvergedIndicator(
        mean_threshold=cfg.mean_threshold,
        std_threshold=cfg.std_threshold,
        min_iter_steps=cfg.min_iter_steps,
        max_iter_steps=cfg.max_iter_steps,
        player_id=policy.current_player_id,
    )

    # other initializations
    cnt_update: int = 0  # count of policy updates
    meta_policy_0, meta_policy_1 = np.array([1.0]), np.array([1.0])
    payoffs = get_new_payoffs(
        env=env, policy=policy, old_payoffs=None
    )  # (a) Complete: compute missing payoff tensor entries via game simulations
    win_rate = payoffs_to_win_rate(payoffs)
    log_heatmap(win_rate)

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )
    pbar = tqdm(collector, total=total_frames // frames_per_batch)

    env.train()

    logging.info(f"Run URL:{wandb_run.get_url()}")

    for i, data in enumerate(pbar):

        if max_population_size and payoffs.shape[0] >= max_population_size:
            logging.info(
                f"Population size reaches the maximum size {max_population_size}."
            )
            break

        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        if i == 0:  # first evaluation
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate(env, policy))
            env.train()

        info.update(
            policy.train_op(data.to_tensordict())
        )  # update policy of the current playe

        empirical_win_rate = get_emprical_win_rate(
            data,
            win_key=f"actor_{policy.current_player_id}_wins",
            lose_key=f"actor_{1-policy.current_player_id}_wins",
        )
        if empirical_win_rate is not None:
            converged_indicator.update(empirical_win_rate)

        if converged_indicator.converged():

            policy.append_actor(
                policy.current_player_id, share_population=share_population
            )  # (c) Expand: add new policies to the population (If share_population == false and cnt_update % 2 == 1, the new policy is only used in the evaluation)
            policy.switch_current_player_id()  # switch current player id

            if init_by_latest_strategy:
                policy.set_actor_params_with_latest_policy()
            else:
                policy.set_actor_params_with_initial_random_policy()

            converged_indicator.reset(policy.current_player_id)
            cnt_update += 1

            if share_population:
                if (
                    len(policy.population_0) == 1 and len(policy.population_1) == 1
                ):  # initial policy is random policy and is replaced by trained policy
                    payoffs = get_new_payoffs(
                        env, policy, None
                    )  # (a) Complete: compute missing payoff tensor entries via game simulations
                else:
                    payoffs = get_new_payoffs(
                        env, policy, payoffs
                    )  # (a) Complete: compute missing payoff tensor entries via game simulations
                win_rate = payoffs_to_win_rate(payoffs)
                log_heatmap(win_rate)
                meta_policy_0, meta_policy_1 = meta_solver.solve(
                    [payoffs, -payoffs.T]
                )  # (b) Solve: calculate meta-strategy via meta-solver

                logging.info(f"cnt_update:{cnt_update}")
                logging.info(f"payoffs:{payoffs}")
                logging.info(f"win_rate:{win_rate}")
                # logging.log(f"JPC:{calculate_jpc(payoffs)}")
                logging.info(
                    f"Meta-policy_0:{meta_policy_0}, Meta-policy_1:{meta_policy_1}"
                )

                if cfg.save_meta_policy:
                    meta_policy_dict = {
                        "meta_policy_0": meta_policy_0,
                        "meta_policy_1": meta_policy_1,
                    }
                    np.savez(
                        os.path.join(
                            meta_policy_save_path, f"meta_policy_iter_{cnt_update}.npz"
                        ),
                        **meta_policy_dict,
                    )

            else:
                if (
                    cnt_update % 2 == 0
                ):
                    if (
                        len(policy.population_0) == 1 and len(policy.population_1) == 1
                    ):  # initial policy is random policy and is replaced by trained policy
                        payoffs = get_new_payoffs(
                            env, policy, None
                        )  # (a) Complete: compute missing payoff tensor entries via game simulations
                    else:
                        payoffs = get_new_payoffs(
                            env, policy, payoffs
                        )  # (a) Complete: compute missing payoff tensor entries via game simulations
                    win_rate = payoffs_to_win_rate(payoffs)
                    log_heatmap(win_rate)
                    meta_policy_0, meta_policy_1 = meta_solver.solve(
                        [payoffs, -payoffs.T]
                    )  # (b) Solve: calculate meta-strategy via meta-solver

                    logging.info(f"cnt_update:{cnt_update}")
                    logging.info(f"payoffs:{payoffs}")
                    logging.info(f"win_rate:{win_rate}")
                    # logging.log(f"JPC:{calculate_jpc(payoffs)}")
                    logging.info(
                        f"Meta-policy_0:{meta_policy_0}, Meta-policy_1:{meta_policy_1}"
                    )

                    if cfg.save_meta_policy:
                        meta_policy_dict = {
                            "meta_policy_0": meta_policy_0,
                            "meta_policy_1": meta_policy_1,
                        }
                        np.savez(
                            os.path.join(
                                meta_policy_save_path,
                                f"meta_policy_iter_{cnt_update/2}.npz",
                            ),
                            **meta_policy_dict,
                        )

            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            logging.info(f"Eval at {collector._frames} steps.")
            if share_population and len(policy.population_0) > 1:
                info.update(evaluate_share_population(env, policy))
            info.update(evaluate(env, policy))
            wandb_run.log(info)

            env.reset()

            if max_population_size and payoffs.shape[0] >= max_population_size:
                logging.info(
                    f"Population size reaches the maximum size {max_population_size}."
                )
                break

        policy.sample_pure_strategy(meta_policy_0, meta_policy_1)

        wandb_run.log(info)

        pbar.set_postfix(
            {
                "rollout_fps": collector._fps,
                "frames": collector._frames,
            }
        )

    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate(env, policy))
    wandb_run.log(info)

    if hasattr(policy, "state_dict"):
        ckpt_path = os.path.join(wandb_run.dir, "policy_final.pt")
        logging.info(f"Save checkpoint to {str(ckpt_path)}")
        torch.save(policy.state_dict(), ckpt_path)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_psro")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    run_name_suffix = cfg.get("run_name_suffix")
    if cfg.get("share_population", False):
        if run_name_suffix is None:
            run = init_wandb(cfg, cfg.solver_type, "share_population")
        else:
            run = init_wandb(cfg, cfg.solver_type, "share_population", run_name_suffix)
    else:
        if run_name_suffix is None:
            run = init_wandb(cfg, cfg.solver_type)
        else:
            run = init_wandb(cfg, cfg.solver_type, run_name_suffix)
    setproctitle(run.name)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from volley_bots.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    def log(info):
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
