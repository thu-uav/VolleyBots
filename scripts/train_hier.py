import logging
import os
import time

from typing import Dict, Any

import hydra
import torch
import numpy as np
import wandb
from torch import vmap
from omegaconf import OmegaConf, DictConfig

from volley_bots import CONFIG_PATH, init_simulation_app
from volley_bots.utils.torchrl import SyncDataCollector, AgentSpec
from volley_bots.utils.torchrl.transforms import (
    LogOnEpisode,
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    History,
)
from volley_bots.utils.wandb import init_wandb
from volley_bots.learning.hier import (
    MAPPOPolicy_hier,
)

from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv,
    InitTracker,
    Compose,
)

from tqdm import tqdm

import copy

import matplotlib.pyplot as plt

from torch import nn

from tensordict import TensorDict

import pandas as pd

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


from volley_bots.utils.stats import PROCESS_FUNC


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_hier")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    
    run_name_suffix: str = cfg.get("run_name_suffix")
    if run_name_suffix is None:
        run = init_wandb(cfg)
    else:
        run = init_wandb(cfg, run_name_suffix)
    
    simulation_app = init_simulation_app(cfg)

    setproctitle(run.name)

    from volley_bots.envs.isaac_env import IsaacEnv

    algos = {
        "mappo_hier": MAPPOPolicy_hier,
    }

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)


    def log(info: Dict[str, Any]):
        run.log(info)

        interested_keys = [
            "train/stats.return",
            "train/stats.episode_len",
        ]
        info = {k: v for k, v in info.items() if k in interested_keys}
        if len(info) > 0:
            print(OmegaConf.to_yaml(info))


    stats_keys = [
        k
        for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]

    log_stats_keys = [
        k[1]
        for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    modified_log_stats_keys = [
        "reward/" + s if (s.startswith("reward_") or s.startswith("penalty_") or s.startswith("return"))else s
        for s in log_stats_keys
    ]

    process_func = copy.copy(PROCESS_FUNC)

    logger = LogOnEpisode(
        cfg.env.num_envs,
        in_keys=stats_keys,
        log_keys=modified_log_stats_keys,
        logger_func=log,
        process_func=process_func,
    )
    transforms = [InitTracker(), logger]

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    Opp_agent_spec: AgentSpec = env.agent_spec["Opp"]

    FirstPass_serve_agent_spec: AgentSpec = env.agent_spec["FirstPass_serve"]
    FirstPass_serve_hover_agent_spec: AgentSpec = env.agent_spec["FirstPass_serve_hover"]
    FirstPass_goto_agent_spec: AgentSpec = env.agent_spec["FirstPass_goto"]
    FirstPass_agent_spec: AgentSpec = env.agent_spec["FirstPass"]
    FirstPass_hover_agent_spec: AgentSpec = env.agent_spec["FirstPass_hover"]

    SecPass_goto_agent_spec: AgentSpec = env.agent_spec["SecPass_goto"]
    SecPass_agent_spec: AgentSpec = env.agent_spec["SecPass"]
    SecPass_hover_agent_spec: AgentSpec = env.agent_spec["SecPass_hover"]

    Att_goto_agent_spec: AgentSpec = env.agent_spec["Att_goto"]
    Att_agent_spec: AgentSpec = env.agent_spec["Att"]
    Att_hover_agent_spec: AgentSpec = env.agent_spec["Att_hover"]

    agent_spec_dict = {}

    agent_spec_dict["Opp"] = Opp_agent_spec

    agent_spec_dict["FirstPass_serve"] = FirstPass_serve_agent_spec
    agent_spec_dict["FirstPass_serve_hover"] = FirstPass_serve_hover_agent_spec
    agent_spec_dict["FirstPass_goto"] = FirstPass_goto_agent_spec
    agent_spec_dict["FirstPass"] = FirstPass_agent_spec
    agent_spec_dict["FirstPass_hover"] = FirstPass_hover_agent_spec

    agent_spec_dict["SecPass_goto"] = SecPass_goto_agent_spec
    agent_spec_dict["SecPass"] = SecPass_agent_spec
    agent_spec_dict["SecPass_hover"] = SecPass_hover_agent_spec

    agent_spec_dict["Att_goto"] = Att_goto_agent_spec
    agent_spec_dict["Att"] = Att_agent_spec
    agent_spec_dict["Att_hover"] = Att_hover_agent_spec

    policy = algos[cfg.algo.name.lower()](
        cfg.algo, agent_spec_dict=agent_spec_dict, device="cuda"
    )

    if cfg.get("Opp_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_policy_checkpoint_path), player="Opp")
        print(f"Load Opp policy from {cfg.Opp_policy_checkpoint_path}")
    
    if cfg.get("FirstPass_serve_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_serve_policy_checkpoint_path), player="FirstPass_serve")
        print(f"Load FirstPass_serve policy from {cfg.FirstPass_serve_policy_checkpoint_path}")
    if cfg.get("FirstPass_serve_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_serve_hover_policy_checkpoint_path), player="FirstPass_serve_hover")
        print(f"Load FirstPass_serve_hover policy from {cfg.FirstPass_serve_hover_policy_checkpoint_path}")
    if cfg.get("FirstPass_goto_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_goto_policy_checkpoint_path), player="FirstPass_goto")
        print(f"Load FirstPass_goto policy from {cfg.FirstPass_goto_policy_checkpoint_path}")
    if cfg.get("FirstPass_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_policy_checkpoint_path), player="FirstPass")
        print(f"Load FirstPass policy from {cfg.FirstPass_policy_checkpoint_path}")
    if cfg.get("FirstPass_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_hover_policy_checkpoint_path), player="FirstPass_hover")
        print(f"Load FirstPass_hover policy from {cfg.FirstPass_hover_policy_checkpoint_path}")
    
    if cfg.get("SecPass_goto_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.SecPass_goto_policy_checkpoint_path), player="SecPass_goto")
        print(f"Load SecPass_goto policy from {cfg.SecPass_goto_policy_checkpoint_path}")
    if cfg.get("SecPass_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.SecPass_policy_checkpoint_path), player="SecPass")
        print(f"Load SecPass policy from {cfg.SecPass_policy_checkpoint_path}")
    if cfg.get("SecPass_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.SecPass_hover_policy_checkpoint_path), player="SecPass_hover")
        print(f"Load SecPass_hover policy from {cfg.SecPass_hover_policy_checkpoint_path}")
    
    if cfg.get("Att_goto_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Att_goto_policy_checkpoint_path), player="Att_goto")
        print(f"Load Att_goto policy from {cfg.Att_goto_policy_checkpoint_path}")
    if cfg.get("Att_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Att_policy_checkpoint_path), player="Att")
        print(f"Load Att policy from {cfg.Att_policy_checkpoint_path}")
    if cfg.get("Att_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Att_hover_policy_checkpoint_path), player="Att_hover")
        print(f"Load Att_hover policy from {cfg.Att_hover_policy_checkpoint_path}")

    only_eval = cfg.get("only_eval", False)
    only_eval_one_traj = cfg.get("only_eval_one_traj", False)
    if only_eval_one_traj and only_eval == False:
        raise ValueError("only_eval is required for only_eval_one_traj")
    if only_eval_one_traj and cfg.task.env.num_envs != 1:
        raise ValueError("only_eval_one_traj is only supported for single env")

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )


    @torch.no_grad()
    def evaluate(policy = None, only_eval_one_traj: bool = False):
        """
        Evaluate the policy.
        input:
            only_eval_one_traj: bool, if True, only evaluate one trajectory.
        """
        policy = policy
        frames = []

        def record_frame(*args, **kwargs):
            frame = base_env.render(mode="rgb_array")
            frames.append(frame)

        base_env.enable_render(True)
        env.reset()
        env.eval()
        
        if only_eval_one_traj:
            rollout = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=Every(record_frame, 2),
                auto_reset=True,
                break_when_any_done=True,
                return_contiguous=False,
            )
        else:
            rollout = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=Every(record_frame, 2),
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )

            done = rollout.get(("next", "done"))
            first_done = torch.argmax(done.long(), dim=1).cpu()
            first_done += 1

            def take_first_episode(tensor: torch.Tensor):
                indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
                return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)
            
            traj_stats = {
                k: take_first_episode(v)
                for k, v in rollout[("stats")].cpu().items()
            }

            info = {
                "eval/stats." + k: torch.mean(v.float()).item()
                for k, v in traj_stats.items()
            }

        base_env.enable_render(not cfg.headless)
        env.reset()
        env.train()

        if len(frames):
            # video_array = torch.stack(frames)
            video_array = np.stack(frames).transpose(0,3,1,2)
            # print(video_array.shape) # (max_steps/2,H,W,C)
            info["recording"] = wandb.Video(
                video_array, fps=0.5 / cfg.sim.dt, format="mp4"
            )
        frames.clear()

        return info


    if only_eval == False: 
        pbar = tqdm(collector, total = total_frames // frames_per_batch)
        env.train()
        for i, data in enumerate(pbar):

            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            info.update(policy.train_op(data.to_tensordict()))

            if eval_interval > 0 and i % eval_interval == 0:
                logging.info(f"Eval at {collector._frames} steps.")
                info.update(evaluate(policy=policy))
                env.train()

            if save_interval > 0 and i % save_interval == 0:
                pass
                # if hasattr(policy, "state_dict"):
                #     ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}_att_hover.pt")
                #     logging.info(f"Save checkpoint to {str(ckpt_path)}")
                #     torch.save(policy.state_dict(player="Att_hover"), ckpt_path)

            run.log(info)
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

            pbar.set_postfix(
                {
                    "rollout_fps": collector._fps,
                    "frames": collector._frames,
                }
            )

            if max_iters > 0 and i >= max_iters - 1:
                break

        if hasattr(policy, "state_dict"):
            pass
            # ckpt_path = os.path.join(run.dir, f"checkpoint_final_att_hover.pt")
            # logging.info(f"Save checkpoint to {str(ckpt_path)}")
            # torch.save(policy.state_dict(player="Att_hover"), ckpt_path)

    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate(policy=policy, only_eval_one_traj=only_eval_one_traj))

    run.log(info)

    wandb.save(os.path.join(run.dir, "checkpoint*"))
    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()