import os
from typing import List, Tuple

import functorch
import omni.isaac.core.materials as materials
import omni.isaac.core.objects as objects
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.torch as torch_utils
import omni.kit.commands
import torch
import torch.distributions as D
import torch.nn.functional as NNF
from carb import Float3
from omegaconf import DictConfig
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.orbit.sensors import ContactSensor, ContactSensorCfg
from pxr import PhysxSchema, UsdShade
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

import omni_drones.utils.kit as kit_utils
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.envs.volleyball.common import (
    _carb_float3_add,
    rectangular_cuboid_edges,
)
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import euler_to_quaternion, normalize, quat_axis
from omni_drones.views import ArticulationView, RigidPrimView

_COLOR_T = Tuple[float, float, float, float]
x = 0


def my_function():
    global x
    x += 1


def _draw_net(
    W: float,
    H_NET: float,
    W_NET: float,
    color_mesh: _COLOR_T = (1.0, 1.0, 1.0, 1.0),
    color_post: _COLOR_T = (1.0, 0.729, 0, 1.0),
    size_mesh_line: float = 3.0,
    size_post: float = 10.0,
):
    n = 30

    point_list_1 = [Float3(0, -W / 2, i * W_NET / n + H_NET - W_NET) for i in range(n)]
    point_list_2 = [Float3(0, W / 2, i * W_NET / n + H_NET - W_NET) for i in range(n)]

    point_list_1.append(Float3(0, W / 2, 0))
    point_list_1.append(Float3(0, -W / 2, 0))

    point_list_2.append(Float3(0, W / 2, H_NET))
    point_list_2.append(Float3(0, -W / 2, H_NET))

    colors = [color_mesh for _ in range(n)]
    sizes = [size_mesh_line for _ in range(n)]
    colors.append(color_post)
    colors.append(color_post)
    sizes.append(size_post)
    sizes.append(size_post)

    return point_list_1, point_list_2, colors, sizes


def _draw_board(
    W: float, L: float, color: _COLOR_T = (1.0, 1.0, 1.0, 1.0), line_size: float = 10.0
):
    point_list_1 = [
        Float3(-L / 2, -W / 2, 0),
        Float3(-L / 2, W / 2, 0),
        Float3(-L / 2, -W / 2, 0),
        Float3(L / 2, -W / 2, 0),
        Float3(-L / 6, -W / 2, 0),
        Float3(L / 6, -W / 2, 0),
        Float3(0, -W / 2, 0),
    ]
    point_list_2 = [
        Float3(L / 2, -W / 2, 0),
        Float3(L / 2, W / 2, 0),
        Float3(-L / 2, W / 2, 0),
        Float3(L / 2, W / 2, 0),
        Float3(-L / 6, W / 2, 0),
        Float3(L / 6, W / 2, 0),
        Float3(0, W / 2, 0),
    ]

    colors = [color for _ in range(len(point_list_1))]
    sizes = [line_size for _ in range(len(point_list_1))]

    return point_list_1, point_list_2, colors, sizes


def _draw_lines_args_merger(*args):

    buf = [[] for _ in range(4)]
    for arg in args:
        buf[0].extend(arg[0])
        buf[1].extend(arg[1])
        buf[2].extend(arg[2])
        buf[3].extend(arg[3])

    return (
        buf[0],
        buf[1],
        buf[2],
        buf[3],
    )


def draw_court(W: float, L: float, H_NET: float, W_NET: float):

    return _draw_lines_args_merger(_draw_net(W, H_NET, W_NET), _draw_board(W, L))


def calculate_penalty_drone_abs_z(drone_rpos: torch.Tensor) -> torch.Tensor:

    # 获取无人机在z轴上的绝对值
    drone_rpos_z = drone_rpos[:, :, 2].abs()
    # 计算惩罚值：根据z方向上相对偏移量
    tmp = 1 / (drone_rpos_z - 0.1).clamp(min=0.1, max=0.7) - 1 / 0.7

    return tmp * 0.15


def attach_payload(parent_path):
    import omni.physx.scripts.utils as script_utils
    from omni.isaac.core import objects
    from pxr import UsdPhysics

    payload_prim = objects.DynamicCuboid(
        prim_path=parent_path + "/payload",
        scale=torch.tensor([0.1, 0.1, 0.15]),
        mass=0.0001,
    ).prim

    parent_prim = prim_utils.get_prim_at_path(parent_path + "/base_link")
    stage = prim_utils.get_current_stage()

    joint = script_utils.createJoint(stage, "Prismatic", payload_prim, parent_prim)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    joint.GetAttribute("physics:lowerLimit").Set(-0.15)
    joint.GetAttribute("physics:upperLimit").Set(0.15)
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("drive:linear:physics:damping").Set(10.0)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(10000.0)


class TestVelReturn(IsaacEnv):
    def __init__(self, cfg, headless):

        self.L: float = cfg.task.court.L
        self.W: float = cfg.task.court.W
        self.H_NET: float = cfg.task.court.H_NET
        self.W_NET: float = cfg.task.court.W_NET
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()
        self.throttles_in_obs = cfg.task.throttles_in_obs
        self.use_ctbr = True if cfg.task.action_transform == "PIDrate" else False
        super().__init__(cfg, headless)

        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.anchor_radius = cfg.task.anchor_radius
        self.time_encoding = cfg.task.time_encoding
        self.reward_velocity_factor = cfg.task.reward_velocity_factor

        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )
        self.draw = _debug_draw.acquire_debug_draw_interface()
        print("self.central_env_idx", self.central_env_idx)
        print("self.central_env_pos", self.central_env_pos)

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
        if "payload" in self.randomization:
            payload_cfg = self.randomization["payload"]

            self.payload_z_dist = D.Uniform(
                torch.tensor([payload_cfg["z"][0]], device=self.device),
                torch.tensor([payload_cfg["z"][1]], device=self.device),
            )
            self.payload_mass_dist = D.Uniform(
                torch.tensor([payload_cfg["mass"][0]], device=self.device),
                torch.tensor([payload_cfg["mass"][1]], device=self.device),
            )
            self.payload = RigidPrimView(
                f"/World/envs/env_*/{self.drone.name}_*/payload",
                reset_xform_properties=False,
                shape=(-1, self.drone.n),
            )
            self.payload.initialize()

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        # self.anchor = torch.tensor(cfg.task.anchor, device=self.device)
        # self.anchor_1 = torch.tensor(cfg.task.anchor_1, device=self.device)

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.5, -2.5, 1.0], device=self.device),
            torch.tensor([2.5, 2.5, 2.5], device=self.device),
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.1, -0.1, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 2.0], device=self.device) * torch.pi,
        )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.0, 0.0, 2.0], device=self.device) * torch.pi,
        )

        # self.target_heading = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.alpha = 0.8

        self.start_pos = torch.tensor(cfg.task.anchor, device=self.device)
        self.end_pos = torch.tensor(cfg.task.anchor_1, device=self.device)

        self.target_time_count = torch.zeros(self.num_envs, device=self.device)

        self.target_pos_all = self.end_pos.expand(self.num_envs, 1, 3)

        self.target_reached_nums = torch.zeros(self.num_envs, 1, device=self.device)

    def _design_scene(self):
        import omni.isaac.core.utils.prims as prim_utils

        import omni_drones.utils.kit as kit_utils

        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        if self.use_local_usd:
            # use local usd resources
            usd_path = os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "assets",
                "default_environment.usd",
            )
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                usd_path=usd_path,
            )
        else:
            # use online usd resources
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )

        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]
        if self.has_payload:
            attach_payload(drone_prim.GetPath().pathString)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3

        if not self.throttles_in_obs:
            observation_dim -= 4

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {
                            "observation": UnboundedContinuousTensorSpec(
                                (1, observation_dim), device=self.device
                            ),
                            "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(
                                self.device
                            ),
                        }
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        self.action_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {
                            "action": self.drone.action_spec.unsqueeze(0),
                        }
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        self.reward_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {"reward": UnboundedContinuousTensorSpec((1, 1))}
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        self.done_spec = (
            CompositeSpec(
                {
                    "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                    "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                    "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

        stats_spec = (
            CompositeSpec(
                {
                    "return": UnboundedContinuousTensorSpec(1),
                    "episode_len": UnboundedContinuousTensorSpec(1),
                    "dist_to_anchor": UnboundedContinuousTensorSpec(1),
                    # "heading_alignment": UnboundedContinuousTensorSpec(1),
                    "uprightness": UnboundedContinuousTensorSpec(1),
                    "action_smoothness": UnboundedContinuousTensorSpec(1),
                    "linear_velocity": UnboundedContinuousTensorSpec(1),
                    "max_linear_velocity": UnboundedContinuousTensorSpec(1),
                    "z_abs_linear_vel": UnboundedContinuousTensorSpec(1),
                    "drone_z": UnboundedContinuousTensorSpec(1),
                    # "reward_velocity": UnboundedContinuousTensorSpec(1),
                    # "penalty_drone_abs_z": UnboundedContinuousTensorSpec(1),
                    # "reward_effort":  UnboundedContinuousTensorSpec(1),
                    "reward_target_stop": UnboundedContinuousTensorSpec(1),
                    # "reward_spin" : UnboundedContinuousTensorSpec(1),
                    # "reward_up" : UnboundedContinuousTensorSpec(1),
                    # "reward_pose_all": UnboundedContinuousTensorSpec(1),
                    "reward_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                    "target_reached_nums": UnboundedContinuousTensorSpec(1),
                    "done_drone_misbehave": UnboundedContinuousTensorSpec(1),
                    "penalty_drone_misbehave": UnboundedContinuousTensorSpec(1),
                    "penalty_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                    "drone_too_low": UnboundedContinuousTensorSpec(1),
                    "drone_too_high": UnboundedContinuousTensorSpec(1),
                    "drone_too_remote": UnboundedContinuousTensorSpec(1),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        if self.use_ctbr:
            info_spec = (
                CompositeSpec(
                    {
                        "drone_state": UnboundedContinuousTensorSpec(
                            (self.drone.n, 13), device=self.device
                        ),
                        "prev_action": torch.stack(
                            [self.drone.action_spec] * self.drone.n, 0
                        ).to(self.device),
                        "prev_prev_action": torch.stack(
                            [self.drone.action_spec] * self.drone.n, 0
                        ).to(self.device),
                        "target_ctbr": UnboundedContinuousTensorSpec((self.drone.n, 4)),
                        "real_unnormalized_ctbr": UnboundedContinuousTensorSpec(
                            (self.drone.n, 4)
                        ),
                    }
                )
                .expand(self.num_envs)
                .to(self.device)
            )
        else:
            info_spec = (
                CompositeSpec(
                    {
                        "drone_state": UnboundedContinuousTensorSpec(
                            (self.drone.n, 13), device=self.device
                        ),
                    }
                )
                .expand(self.num_envs)
                .to(self.device)
            )

        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec

        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def update_mean_stats(self, key, value, cnt_key, idx=None):
        idx = (
            torch.ones_like(value, device=self.device, dtype=torch.bool)
            if idx is None
            else idx
        )
        n = self.stats[cnt_key][idx]
        new_value = value[idx]
        old_value = self.stats[key][idx]
        self.stats[key][idx] = ((n - 1) * old_value + new_value) / n

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.target_time_count[env_ids] = 0
        self.target_reached_nums[env_ids] = 0
        self.target_pos_all[env_ids] = self.end_pos
        pos = self.start_pos.expand(env_ids.shape[0], 1, -1)

        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)

        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )

        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        if self.has_payload:
            # TODO@btx0424: 找到更好的方法
            payload_z = self.payload_z_dist.sample(env_ids.shape)
            joint_indices = torch.tensor(
                [self.drone._view._dof_indices["PrismaticJoint"]], device=self.device
            )

            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices
            )
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices
            )
            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device),
                env_indices=env_ids,
                joint_indices=joint_indices,
            )

            payload_mass = (
                self.payload_mass_dist.sample(env_ids.shape + (1,))
                * self.drone.masses[env_ids]
            )
            self.payload.set_masses(payload_mass, env_indices=env_ids)

        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            self.draw.clear_lines()
            # self.debug_draw_region()
            self.debug_draw_turn()
            point_list_1, point_list_2, colors, sizes = draw_court(
                self.W, self.L, self.H_NET, self.W_NET
            )
            point_list_1 = [
                _carb_float3_add(p, self.central_env_pos) for p in point_list_1
            ]
            point_list_2 = [
                _carb_float3_add(p, self.central_env_pos) for p in point_list_2
            ]
            self.draw.draw_lines(point_list_1, point_list_2, colors, sizes)

        target_rpy = self.target_rpy_dist.sample((*env_ids.shape, 1))
        target_rot = euler_to_quaternion(target_rpy)
        # self.target_heading[env_ids] = quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)

        self.stats[env_ids] = 0.0

    def debug_draw_turn(self):
        self.draw.clear_points()
        ori = self.envs_positions[self.central_env_idx].detach()
        points = self.target_pos_all[self.central_env_idx].clone() + ori

        points[:, -1] = 0  # 将z坐标设为0，保持在地面
        points = points.tolist()

        colors = [(0, 1, 0, 1)]
        sizes = [25.0]

        self.draw.draw_points(points, colors, sizes)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[
            ("agents", "action")
        ]  # For rotor command actions, not the actions output by the policy.
        self.effort = self.drone.apply_action(actions)

        if self.use_ctbr:
            real_unnormalized_ctbr = tensordict["ctbr"]

            target_rate = tensordict["target_rate"]
            target_rate_roll = target_rate[..., 0]
            target_rate_pitch = target_rate[..., 1]
            target_rate_yaw = target_rate[..., 2]
            target_thrust = tensordict["target_thrust"]
            target_ctbr = torch.cat((target_rate, target_thrust), dim=-1)
            # target rate: [-180, 180] deg/s
            # target thrust: [0, 2**16]

            self.info["real_unnormalized_ctbr"] = real_unnormalized_ctbr
            self.info["target_ctbr"] = target_ctbr
            self.info["prev_action"] = tensordict[("info", "prev_action")]
            self.info["prev_prev_action"] = tensordict[("info", "prev_prev_action")]

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.rpos = self.target_pos_all - self.root_state[..., :3]

        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )
        if self.throttles_in_obs:
            obs = [
                self.root_state,
                self.rpos,
            ]
        else:
            obs = [
                pos,
                rot,
                vel,
                angular_vel,
                heading,
                up,
                self.rpos,
            ]
        # print("obs", [o.shape for o in obs])
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,  # 传感器内参
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        self.vel = self.drone.get_velocities(True)
        self.linear_velocities = self.vel[:, :, :3]
        self.rpos = self.target_pos_all - self.root_state[..., :3]
        dist_to_anchor = torch.norm(self.rpos, p=2, dim=-1)
        target_pos_before_update = self.target_pos_all.clone()

        # 到底目标点判定条件
        # reached_position = ((dist_to_anchor < 0.6)&(torch.norm(self.linear_velocities[..., :2], p=2, dim=-1)<0.4)).squeeze(-1)
        reached_position = (dist_to_anchor < 0.6).squeeze(-1)
        # 更新计数器：如果在目标位置范围内，则计数器加1；否则重置为0
        self.target_time_count_clone = self.target_time_count.clone()
        self.target_time_count_clone[
            reached_position
        ] += 1  # 满足位置条件的环境递增计数器
        self.target_time_count_clone[~reached_position] = (
            0  # 不满足位置条件的环境重置计数器
        )
        self.target_time_count = self.target_time_count_clone

        reached_target = self.target_time_count >= 5
        # 得到满足停留时间条件的索引
        self.reached_indices = torch.nonzero(reached_target, as_tuple=True)
        self.target_reached_nums[self.reached_indices] += 1

        # 仅在达到了目标的那些索引位置上更新 self.target_pos_all

        target_pos_all_clone = self.target_pos_all.clone()
        target_pos_all_clone[self.reached_indices] = torch.where(
            (self.target_pos_all[self.reached_indices] == self.end_pos).all(
                dim=-1, keepdim=True
            ),
            self.start_pos,
            self.end_pos,
        )
        self.target_pos_all = target_pos_all_clone

        # 目标点停留奖励
        reward_target_stop = 2.5 * (self.target_time_count).unsqueeze(-1)
        # 基于距离计算奖励
        reward_dist_to_anchor = 0.5 / (1.0 + torch.square(1.0 * dist_to_anchor))

        # uprightness
        # reward_up = 0.3 * torch.square((self.drone.up[..., 2] + 1) / 2 )

        # 旋转奖励
        # spinnage = torch.square(self.drone.vel[..., -1])
        # reward_spin = 1.0 / (1.0 + torch.square(spinnage))

        # reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        # reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        # reward_velocity
        # reward_velocity = self.reward_velocity_factor * torch.norm(self.linear_velocities[..., :2], p=2, dim=-1)

        # penalty_drone_abs_z = calculate_penalty_drone_abs_z(self.rpos)

        _dist_coeff = 0.01
        penalty_dist_to_anchor = _dist_coeff * (
            dist_to_anchor - self.anchor_radius
        ).clamp(
            min=0
        )  # individual, sparse, (E, 1)

        drone_too_low = self.drone.pos[..., 2] < 0.4
        drone_too_high = self.drone.pos[..., 2] > 5.0
        drone_too_remote = dist_to_anchor > 7.5
        drone_misbehave = drone_too_low | drone_too_high | drone_too_remote
        penalty_drone_misbehave = drone_misbehave * 10.0

        # reward_pose_all = (reward_dist_to_anchor + reward_dist_to_anchor * (reward_up + reward_spin) + reward_effort) * 0.015 * (70-self.target_time_count.view(-1, 1))

        reward_dist_to_anchor = (
            reward_dist_to_anchor * 0.02 * (50 - self.target_time_count.view(-1, 1))
        )

        reward = (
            reward_dist_to_anchor
            + reward_target_stop
            - penalty_drone_misbehave
            # - penalty_dist_to_anchor
            # + reward_effort
            # + reward_up
            # - penalty_drone_abs_z
            # + reward_action_smoothness  # 0
            # + reward_velocity  # 0
        )

        terminated = drone_misbehave
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        done = terminated | truncated

        if self._should_render(0) and reached_target[self.central_env_idx].any():
            self.debug_draw_turn()

        self.stats["done_drone_misbehave"][:] = drone_misbehave.float()
        self.stats["drone_too_low"][:] = drone_too_low.float()
        self.stats["drone_too_high"][:] = drone_too_high.float()
        self.stats["drone_too_remote"][:] = drone_too_remote.float()

        self.stats["reward_dist_to_anchor"].add_(
            reward_dist_to_anchor.mean(dim=-1, keepdim=True)
        )
        # self.stats["reward_up"].add_(reward_up.mean(dim=-1, keepdim=True))
        # self.stats["reward_spin"].add_(reward_spin)
        # self.stats["reward_pose_all"].add_(reward_pose_all)
        # self.stats["reward_effort"].add_(reward_effort.mean(dim=-1, keepdim=True))
        # self.stats["reward_velocity"].add_(reward_velocity.mean(dim=-1, keepdim=True))
        self.stats["reward_target_stop"].add_(
            reward_target_stop.mean(dim=-1, keepdim=True)
        )
        # self.stats["penalty_drone_abs_z"].add_(penalty_drone_abs_z.mean(dim=-1, keepdim=True))
        self.stats["penalty_drone_misbehave"].add_(
            penalty_drone_misbehave.mean(dim=-1, keepdim=True)
        )
        self.stats["penalty_dist_to_anchor"].add_(
            penalty_dist_to_anchor.mean(dim=-1, keepdim=True)
        )
        # self.stats["heading_alignment"].add_(heading_alignment)
        # self.stats["uprightness"].add_(self.root_state[..., 18])
        # self.stats["action_smoothness"].add_(-self.drone.throttle_difference)
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        speed_magnitude = torch.norm(self.linear_velocities, dim=-1)
        # self.stats["linear_velocity"][:] = speed_magnitude
        self.stats["max_linear_velocity"] = speed_magnitude.max(dim=0).values.expand(
            speed_magnitude.shape
        )
        self.update_mean_stats(
            "z_abs_linear_vel", self.vel[:, :, 2].abs(), "episode_len"
        )
        self.update_mean_stats("linear_velocity", speed_magnitude, "episode_len")
        self.update_mean_stats("dist_to_anchor", dist_to_anchor, "episode_len")
        self.update_mean_stats("drone_z", self.drone.pos[..., 2], "episode_len")
        # self.update_mean_stats("heading_alignment", heading_alignment, "episode_len")
        self.update_mean_stats("uprightness", self.root_state[..., 18], "episode_len")
        self.update_mean_stats(
            "action_smoothness", -self.drone.throttle_difference, "episode_len"
        )
        self.stats["target_reached_nums"][:] = self.target_reached_nums.float()

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
