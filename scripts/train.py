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

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import SyncDataCollector, AgentSpec
from omni_drones.utils.torchrl.transforms import (
    LogOnEpisode,
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    History,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import (
    MAPPOPolicy,
    HAPPOPolicy,
    QMIXPolicy,
    MADDPGPolicy,
    MATPolicy,
    TD3Policy,
    SACPolicy,
    DQNPolicy,
    VDNPolicy,
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


class debug_policy(nn.Module):

    def __init__(self, agent_spec, actions=None):
        super(debug_policy, self).__init__()
        self.agent_spec = agent_spec
        print(self.agent_spec.observation_spec) # [E, 2, 33]
        print(self.agent_spec.action_spec) # [E, 2, 4]
        self.obs_name = ("agents", "observation")
        self.state_name = ("agents", "state")
        self.act_name = ("agents", "action")
        self.reward_name = ("agents", "reward")
        self.actor_in_keys = [self.obs_name, self.act_name]
        self.actions = actions
        self.i = 0
    
    def forward(self, tensordict: TensorDict):

        if self.actions is not None:

            action = self.actions[self.i]
            action = action.expand(1, 1, 4)  # [1, 1, 4]

            self.i += 1
            self.i %= self.actions.shape[0]
            tensordict.set(("agents", "action"), action)

        else:

            observation: torch.Tensor = tensordict['agents', 'observation'] # [E, 1, 33]
            
            action_dim = 4
            action_shape = observation.shape[:-2] + (1, action_dim) # [1, 1, 4]
            action = 2 * torch.rand(action_shape, device=observation.device) - 1  # [1, 1, 4]
            # action = action.expand(-1, 2, -1)  # [1, 2, 4]

            tensordict.set(("agents", "action"), action)

        return tensordict


from omni_drones.utils.stats import PROCESS_FUNC


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    run = init_wandb(cfg)
    
    simulation_app = init_simulation_app(cfg)

    setproctitle(run.name)

    from omni_drones.envs.isaac_env import IsaacEnv

    algos = {
        "mappo": MAPPOPolicy,
        "happo": HAPPOPolicy,
        "qmix": QMIXPolicy,
        "maddpg": MADDPGPolicy,
        "mat": MATPolicy,
        "td3": TD3Policy,
        "sac": SACPolicy,
        "dqn": DQNPolicy,
        "vdn":VDNPolicy,
    }

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    def log(info: Dict[str, Any]):
        run.log(info)

        interested_keys = [
            "train/stats.return",
            "train/stats.score",
            "train/stats.episode_len",
            "train/stats.delta_pos",
            "train/stats.delta_vel",
            "train/stats.theta",
            "train/stats.yaw",
            "train/stats.drone_z_on_hit"
        ]
        info = {k: v for k, v in info.items() if k in interested_keys}
        if len(info) > 0:
            print(OmegaConf.to_yaml(info))

    stats_keys = [
        k
        for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]

    process_func = copy.copy(PROCESS_FUNC)

    logger = LogOnEpisode(
        cfg.env.num_envs,
        in_keys=stats_keys,
        log_keys=stats_keys,
        logger_func=log,
        process_func=process_func,
    )
    transforms = [InitTracker(), logger]

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.algo.get("action_transform", None)
    print("action_transform", action_transform)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            print("nbins", nbins)
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")
    print("transforms", transforms)

    action_transform: str = cfg.task.get("action_transform", None)
    print("action_transform", action_transform)
    if action_transform == "rate":
        from omni_drones.controllers import RateController as _RateController
        from omni_drones.utils.torchrl.transforms import RateController
        controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
        transform = RateController(controller)
        transforms.append(transform)
    elif action_transform == "PIDrate": # CTBR
        from omni_drones.controllers import PIDRateController as _PIDRateController
        from omni_drones.utils.torchrl.transforms import PIDRateController
        controller = _PIDRateController(cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
        transform = PIDRateController(controller)
        # transforms.append(TanhTransform)
        transforms.append(transform)

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](
        cfg.algo, agent_spec=agent_spec, device="cuda"
    )

    if cfg.get("policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.policy_checkpoint_path))
        print(f"Load policy from {cfg.policy_checkpoint_path}")
    
    only_eval_once = cfg.get("only_eval_once", False)
    use_debug_policy = cfg.get("use_debug_policy", False)
    eval_type = cfg.get("eval_type", None)

    if use_debug_policy:
        if only_eval_once == False:
            raise ValueError("only_eval_once is required for use_debug_policy")
    if only_eval_once and use_debug_policy==False:
        if cfg.get("policy_checkpoint_path") is None:
            raise ValueError("policy_checkpoint_path is required for only_eval_once without use_debug_policy")

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
    def evaluate(policy = None, only_eval_once: bool = False, eval_type: str = None):
        """
        Evaluate the policy.
        input:
            only_eval_once: bool, if True, only evaluate once and one environment.
            eval_type: str, the type of evaluation used if only_eval_once=true, "ctbr" or "log_actions"
        """
        policy = policy
        frames = []

        def record_frame(*args, **kwargs):
            frame = base_env.render(mode="rgb_array")
            frames.append(frame)

        base_env.enable_render(True)
        env.reset()
        env.eval()
        
        if only_eval_once:

            if use_debug_policy:

                df = pd.read_csv('log_actions/action_hover.csv', delimiter=',', header=None)
                np_array = df.to_numpy()
                action = torch.tensor(np_array) # [500, 4]
                policy = debug_policy(agent_spec, action)

            rollout = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=Every(record_frame, 2),
                auto_reset=True,
                break_when_any_done=True,
                return_contiguous=False,
            )

            if eval_type == "ctbr":
            
                # CT: [0, 2^16]
                # BR: [-180, 180] degree/s
                target_ctbr = rollout["info"]["target_ctbr"][:, :, 0, :] # only one uav
                
                # CT: [0, 2^16]
                # BR: [-2^16, 2^16]
                real_ctbr = rollout["info"]["real_ctbr"][:, :, 0, :] # only one uav
                
                # degree
                roll = rollout["info"]["roll"]/torch.pi*180
                pitch = rollout["info"]["pitch"]/torch.pi*180
                yaw = rollout["info"]["yaw"]/torch.pi*180
                

                output = torch.cat((target_ctbr, real_ctbr, roll, pitch, yaw), axis=-1)
                output = output[0, ...] # only one env, size: ()
                
                output = output.cpu().numpy()

                os.makedirs('ctbr', exist_ok=True)

                np.savetxt(
                    'ctbr/output.csv', 
                    output, 
                    delimiter=',', 
                    fmt='%.4f', 
                    header='Target Roll Rate, Target Pitch Rate, Target Yaw Rate, Target Thrust, Real Roll Rate, Real Pitch Rate, Real Yaw Rate, Real Thrust, Roll, Pitch, Yaw', comments=''
                    )
                wandb.save('ctbr/output.csv')


                # Plotting the target consistency check
                plt.figure(figsize=(12, 10))
                time = np.arange(0, output.shape[0]*cfg.sim.dt, cfg.sim.dt)
                # Roll Rate Consistency Check
                ax1 = plt.subplot(3, 1, 1)
                ax1.plot(time, output[:, 0], label='Roll Rate (deg/s)', color='orange')
                ax1.set_ylabel('Rate (deg/s)', color='orange')
                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()
                ax2.plot(time, output[:, 8], label='Rate of Roll Angle (deg)', color='red')
                ax2.set_ylabel('Angle (deg)', color='red')
                ax2.tick_params(axis='y')
                plt.title('Roll Rate Consistency Check')
                plt.xlabel('Time (s)')
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                # Pitch Rate Consistency Check
                ax1 = plt.subplot(3, 1, 2)
                ax1.plot(time, output[:, 1], label='Pitch Rate (deg/s)', color='orange')
                ax1.set_ylabel('Rate (deg/s)', color='orange')
                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()
                ax2.plot(time, output[:, 9], label='Rate of Pitch Angle (deg)', color='red')
                ax2.set_ylabel('Angle (deg)', color='red')
                ax2.tick_params(axis='y')
                plt.title('Pitch Rate Consistency Check')
                plt.xlabel('Time (s)')
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                # Yaw Rate Consistency Check
                ax1 = plt.subplot(3, 1, 3)
                ax1.plot(time, output[:, 2], label='Yaw Rate (deg/s)', color='orange')
                ax1.set_ylabel('Rate (deg/s)', color='orange')
                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()
                ax2.plot(time, output[:, 10], label='Rate of Yaw Angle (deg)', color='red')
                ax2.set_ylabel('Angle (deg)', color='red')
                ax2.tick_params(axis='y')
                plt.title('Yaw Rate Consistency Check')
                plt.xlabel('Time (s)')
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig("ctbr/target_rate_consistency.png")
                wandb.log({"target_rate_consistency": wandb.Image("ctbr/target_rate_consistency.png")})

                
                # Plotting the real consistency check
                plt.figure(figsize=(12, 10))
                time = np.arange(0, output.shape[0]*cfg.sim.dt, cfg.sim.dt)
                # Roll Rate Consistency Check
                ax1 = plt.subplot(3, 1, 1)
                ax1.plot(time, output[:, 4], label='Roll Rate', color='orange')
                ax1.set_ylabel('Rate', color='orange')
                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()
                ax2.plot(time, output[:, 8], label='Rate of Roll Angle (deg)', color='red')
                ax2.set_ylabel('Angle (deg)', color='red')
                ax2.tick_params(axis='y')
                plt.title('Roll Rate Consistency Check')
                plt.xlabel('Time (s)')
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                # Pitch Rate Consistency Check
                ax1 = plt.subplot(3, 1, 2)
                ax1.plot(time, output[:, 5], label='Pitch Rate', color='orange')
                ax1.set_ylabel('Rate', color='orange')
                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()
                ax2.plot(time, output[:, 9], label='Rate of Pitch Angle (deg)', color='red')
                ax2.set_ylabel('Angle (deg)', color='red')
                ax2.tick_params(axis='y')
                plt.title('Pitch Rate Consistency Check')
                plt.xlabel('Time (s)')
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                # Yaw Rate Consistency Check
                ax1 = plt.subplot(3, 1, 3)
                ax1.plot(time, output[:, 6], label='Yaw Rate', color='orange')
                ax1.set_ylabel('Rate', color='orange')
                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()
                ax2.plot(time, output[:, 10], label='Rate of Yaw Angle (deg)', color='red')
                ax2.set_ylabel('Angle (deg)', color='red')
                ax2.tick_params(axis='y')
                plt.title('Yaw Rate Consistency Check')
                plt.xlabel('Time (s)')
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig("ctbr/real_rate_consistency.png")
                wandb.log({"real_rate_consistency": wandb.Image("ctbr/real_rate_consistency.png")})
            
            elif eval_type == "log_actions":

                action = rollout["agents"]["action"][0, :, 0, :].cpu().numpy()

                os.makedirs('log_actions', exist_ok=True)

                np.savetxt(
                    'log_actions/action.csv', 
                    action, 
                    delimiter=',', 
                    fmt='%.4f',
                    )
                
                wandb.save('log_actions/action.csv')

        else:
            rollout = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=Every(record_frame, 2),
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )

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

    if only_eval_once == False: 
        pbar = tqdm(collector, total = total_frames // frames_per_batch)
        env.train()
        for i, data in enumerate(pbar):
            print(i)
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            info.update(policy.train_op(data.to_tensordict()))

            if eval_interval > 0 and i % eval_interval == 0:
                logging.info(f"Eval at {collector._frames} steps.")
                info.update(evaluate(policy=policy))
                env.train()

            if save_interval > 0 and i % save_interval == 0:
                if hasattr(policy, "state_dict"):
                    ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                    logging.info(f"Save checkpoint to {str(ckpt_path)}")
                    torch.save(policy.state_dict(), ckpt_path)

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
            ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
            logging.info(f"Save checkpoint to {str(ckpt_path)}")
            torch.save(policy.state_dict(), ckpt_path)

    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate(policy=policy, only_eval_once=only_eval_once, eval_type=eval_type))

    run.log(info)

    wandb.save(os.path.join(run.dir, "checkpoint*"))
    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()