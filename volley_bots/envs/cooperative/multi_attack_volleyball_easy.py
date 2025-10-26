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
from pxr import PhysxSchema, UsdShade
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

import volley_bots.utils.kit as kit_utils
from volley_bots.envs.isaac_env import AgentSpec, IsaacEnv
from volley_bots.envs.volleyball.common import (
    _carb_float3_add,
    rectangular_cuboid_edges,
)
from volley_bots.robots.drone import MultirotorBase
from volley_bots.utils.torch import euler_to_quaternion, normalize
from volley_bots.views import RigidPrimView

_COLOR_T = Tuple[float, float, float, float]

import os
import pdb

from omni.isaac.orbit.sensors import ContactSensor, ContactSensorCfg


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
    ]
    point_list_2 = [
        Float3(L / 2, -W / 2, 0),
        Float3(L / 2, W / 2, 0),
        Float3(-L / 2, W / 2, 0),
        Float3(L / 2, W / 2, 0),
    ]

    colors = [color for _ in range(4)]
    sizes = [line_size for _ in range(4)]

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


def quat_rotate(q, v):
    """
    Rotate vector v by quaternion q
    q: quaternion [w, x, y, z]
    v: vector [x, y, z]
    Returns the rotated vector
    """

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Quaternion rotation formula
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    # Construct the rotation matrix from the quaternion
    rot_matrix = torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)

    v_expanded = v.expand(*q.shape[:-1], 3)

    # Rotate the vector using the rotation matrix
    return torch.matmul(rot_matrix, v_expanded.unsqueeze(-1)).squeeze(-1)


def draw_court(W: float, L: float, H_NET: float, W_NET: float):
    return _draw_lines_args_merger(_draw_net(W, H_NET, W_NET), _draw_board(W, L))


def turn_to_mask(turn: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        turn (torch.Tensor): (*,)

    Returns:
        torch.Tensor: (*,2)
    """
    table = torch.tensor(
        [
            [True, False],
            [False, True],
            [False, False],
        ],
        device=turn.device,
    )
    return table[turn[..., 0]]


def turn_to_reward(turn: torch.Tensor):
    """convert representation of drone turn

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env,2)  res[:,i]=1.0 if t[:]==i else -1.0
    """
    table = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
        ],
        device=turn.device,
    )
    return table[turn[..., 0]]


def turn_to_obs(turn: torch.Tensor):
    """convert representation of drone turn to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env, 2, 2)
    """
    table = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        device=turn.device,
    )
    return table[turn[..., 0]]


class MultiAttackVolleyballEasy(IsaacEnv):
    def __init__(self, cfg, headless):
        self.L: float = cfg.task.court.L
        self.W: float = cfg.task.court.W
        self.H_NET: float = cfg.task.court.H_NET  # height of the net
        self.W_NET: float = (
            cfg.task.court.W_NET
        )  # not width of the net, but the width of the net's frame
        self.ball_mass: float = cfg.task.ball_mass
        self.ball_radius: float = cfg.task.ball_radius
        self.racket_radius = 0.2
        self.anchor_radius = cfg.task.anchor_radius
        self.target_radius = cfg.task.target_radius
        self.reward_shaping = cfg.task.reward_shaping

        super().__init__(cfg, headless)

        # x, y, z boundary for drone
        self.env_boundary_x = self.L / 2
        self.env_boundary_y = self.W / 2

        # env paras
        self.time_encoding = self.cfg.task.time_encoding
        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )

        # drone paras
        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization and "drone" in randomization:
            self.drone.setup_randomization(self.cfg.task.randomization["drone"])
        # contact sensor
        contact_sensor_cfg = ContactSensorCfg(prim_path="/World/envs/env_.*/ball")
        self.contact_sensor: ContactSensor = contact_sensor_cfg.class_type(
            contact_sensor_cfg
        )
        self.contact_sensor._initialize_impl()

        # ball paras
        self.ball = RigidPrimView(
            "/World/envs/env_*/ball",
            reset_xform_properties=False,
            track_contact_forces=False,
            shape=(-1, 1),
        )
        self.ball.initialize()

        # drone and ball init
        # (2,3) original positions of two drones without any offset
        self.anchor = torch.tensor(cfg.task.anchor, device=self.device)
        self.target = torch.tensor(cfg.task.target, device=self.device)
        # (2,3) 2 drones' initial positions with offsets
        self.init_drone_pos_dist = D.Uniform(
            torch.tensor(cfg.task.init_drone_pos_dist.low, device=self.device)
            + self.anchor,
            torch.tensor(cfg.task.init_drone_pos_dist.high, device=self.device)
            + self.anchor,
        )
        self.init_drone_rpy_dist = D.Uniform(
            torch.tensor([-0.1, -0.1, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 2.0], device=self.device) * torch.pi,
        )
        self.init_ball_offset = torch.tensor(cfg.task.ball_offset, device=self.device)

        # utils
        self.turn = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.int64)
        self.last_hit_step = torch.zeros(self.num_envs, 2, device=self.device)
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.ball_traj_vis = []
        # one-hot id [E,2,2]
        self.id = torch.zeros((cfg.task.env.num_envs, 2, 2), device=self.device)
        self.id[:, 0, 0] = 1
        self.id[:, 1, 1] = 1

        self.ball_last_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ball_last_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.ball_before_spike_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_after_spike_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_spike_time = torch.zeros(self.num_envs, 1, device=self.device)

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        material = materials.PhysicsMaterial(
            prim_path="/World/Physics_Materials/physics_material_0",
            restitution=0.8,
        )

        ball = objects.DynamicSphere(
            prim_path="/World/envs/env_0/ball",
            radius=self.ball_radius,
            mass=self.ball_mass,
            color=torch.tensor([1.0, 0.2, 0.2]),
            physics_material=material,
        )
        cr_api = PhysxSchema.PhysxContactReportAPI.Apply(ball.prim)
        cr_api.CreateThresholdAttr().Set(0.0)

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
        drone_prims = self.drone.spawn(translations=[(0.0, 0.0, 2.0), (0.0, 0.0, 2.0)])

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = (
            drone_state_dim + 3 + 3 + 3 + 3 + 3 + 2
        )  # specified in function _compute_state_and_obs

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {
                            "observation": UnboundedContinuousTensorSpec(
                                (2, observation_dim)  # 2 drones
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
                            "action": torch.stack([self.drone.action_spec] * 2, dim=0),
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
                        {"reward": UnboundedContinuousTensorSpec((2, 1))}
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
            2,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )

        _stats_spec = CompositeSpec(
            {
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "done": UnboundedContinuousTensorSpec(1),
                "truncated": UnboundedContinuousTensorSpec(1),
                "terminated": UnboundedContinuousTensorSpec(1),
                "ball_misbehave": UnboundedContinuousTensorSpec(1),
                "ball_too_low": UnboundedContinuousTensorSpec(1),
                "ball_too_high": UnboundedContinuousTensorSpec(1),
                "ball_hit_net": UnboundedContinuousTensorSpec(1),
                "ball_out_of_court": UnboundedContinuousTensorSpec(1),
                "drone_misbehave": UnboundedContinuousTensorSpec(1),
                "drone0_misbehave": UnboundedContinuousTensorSpec(1),
                "drone1_misbehave": UnboundedContinuousTensorSpec(1),
                "drone_too_low": UnboundedContinuousTensorSpec(1),
                "drone0_too_low": UnboundedContinuousTensorSpec(1),
                "drone1_too_low": UnboundedContinuousTensorSpec(1),
                "drone_cross_net": UnboundedContinuousTensorSpec(1),
                "drone0_cross_net": UnboundedContinuousTensorSpec(1),
                "drone1_cross_net": UnboundedContinuousTensorSpec(1),
                "wrong_hit": UnboundedContinuousTensorSpec(1),
                "drone0_wrong_hit": UnboundedContinuousTensorSpec(1),
                "drone1_wrong_hit": UnboundedContinuousTensorSpec(1),
                "wrong_hit_turn": UnboundedContinuousTensorSpec(1),
                "drone0_wrong_hit_turn": UnboundedContinuousTensorSpec(1),
                "drone1_wrong_hit_turn": UnboundedContinuousTensorSpec(1),
                "wrong_hit_racket": UnboundedContinuousTensorSpec(1),
                "drone0_wrong_hit_racket": UnboundedContinuousTensorSpec(1),
                "drone1_wrong_hit_racket": UnboundedContinuousTensorSpec(1),
                "misbehave_penalty": UnboundedContinuousTensorSpec(1),
                "drone0_misbehave_penalty": UnboundedContinuousTensorSpec(1),
                "drone1_misbehave_penalty": UnboundedContinuousTensorSpec(1),
                "penalty_ball_misbehave": UnboundedContinuousTensorSpec(1),
                "penalty_drone_misbehave": UnboundedContinuousTensorSpec(1),
                "penalty_wrong_hit": UnboundedContinuousTensorSpec(1),
                "drone0_penalty_wrong_hit": UnboundedContinuousTensorSpec(1),
                "drone1_penalty_wrong_hit": UnboundedContinuousTensorSpec(1),
                "task_reward": UnboundedContinuousTensorSpec(1),
                "reward_success_hit": UnboundedContinuousTensorSpec(1),
                "reward_downward_spike": UnboundedContinuousTensorSpec(1),
                "reward_success_cross": UnboundedContinuousTensorSpec(1),
                "reward_in_target": UnboundedContinuousTensorSpec(1),
                "penalty_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                "num_sim_hits": UnboundedContinuousTensorSpec(1),
                "drone0_num_sim_hits": UnboundedContinuousTensorSpec(1),
                "drone1_num_sim_hits": UnboundedContinuousTensorSpec(1),
                "num_true_hits": UnboundedContinuousTensorSpec(1),
                "drone0_num_true_hits": UnboundedContinuousTensorSpec(1),
                "drone1_num_true_hits": UnboundedContinuousTensorSpec(1),
                "num_success_hits": UnboundedContinuousTensorSpec(1),
                "drone0_num_success_hits": UnboundedContinuousTensorSpec(1),
                "drone1_num_success_hits": UnboundedContinuousTensorSpec(1),
                "wrong_hit_sim": UnboundedContinuousTensorSpec(1),
                "drone0_wrong_hit_sim": UnboundedContinuousTensorSpec(1),
                "drone1_wrong_hit_sim": UnboundedContinuousTensorSpec(1),
                "downward_spike": UnboundedContinuousTensorSpec(1),
                "num_ball_cross": UnboundedContinuousTensorSpec(1),
                "num_success_cross": UnboundedContinuousTensorSpec(1),
                "cross_height": UnboundedContinuousTensorSpec(1),
                "in_target": UnboundedContinuousTensorSpec(1),
                "drone0_x": UnboundedContinuousTensorSpec(1),
                "drone0_y": UnboundedContinuousTensorSpec(1),
                "drone0_z": UnboundedContinuousTensorSpec(1),
                "drone0_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                "drone1_x": UnboundedContinuousTensorSpec(1),
                "drone1_y": UnboundedContinuousTensorSpec(1),
                "drone1_z": UnboundedContinuousTensorSpec(1),
                "drone1_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                "drone0_hit_x": UnboundedContinuousTensorSpec(1),
                "drone0_hit_y": UnboundedContinuousTensorSpec(1),
                "drone0_hit_z": UnboundedContinuousTensorSpec(1),
                "drone0_hit_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                "drone1_hit_x": UnboundedContinuousTensorSpec(1),
                "drone1_hit_y": UnboundedContinuousTensorSpec(1),
                "drone1_hit_z": UnboundedContinuousTensorSpec(1),
                "drone1_hit_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                "ball_before_spike_vel": UnboundedContinuousTensorSpec(1),
                "ball_before_spike_vel_x": UnboundedContinuousTensorSpec(1),
                "ball_before_spike_vel_y": UnboundedContinuousTensorSpec(1),
                "ball_before_spike_vel_z": UnboundedContinuousTensorSpec(1),
                "ball_after_spike_vel": UnboundedContinuousTensorSpec(1),
                "ball_after_spike_vel_x": UnboundedContinuousTensorSpec(1),
                "ball_after_spike_vel_y": UnboundedContinuousTensorSpec(1),
                "ball_after_spike_vel_z": UnboundedContinuousTensorSpec(1),
                "ball_spike_to_done_time": UnboundedContinuousTensorSpec(1),
                "ball_done_x": UnboundedContinuousTensorSpec(1),
                "ball_done_y": UnboundedContinuousTensorSpec(1),
                "ball_done_z": UnboundedContinuousTensorSpec(1),
                "ball_done_dist_to_target_xy": UnboundedContinuousTensorSpec(1),
            }
        )
        if self.reward_shaping:
            _stats_spec.set("shaping_reward", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone0_shaping_reward", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone1_shaping_reward", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_dist_to_ball", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_hit_direction", UnboundedContinuousTensorSpec(1))
            _stats_spec.set(
                "drone0_reward_hit_direction", UnboundedContinuousTensorSpec(1)
            )
            _stats_spec.set(
                "drone1_reward_hit_direction", UnboundedContinuousTensorSpec(1)
            )
            _stats_spec.set("reward_spike_velocity", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_dist_to_target", UnboundedContinuousTensorSpec(1))
        stats_spec = _stats_spec.expand(self.num_envs).to(self.device)

        info_spec = (
            CompositeSpec(
                {
                    "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def check_ball_near_racket(self):
        z_direction_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        z_direction_world = quat_rotate(self.drone_rot, z_direction_local)  # (E,2,3)

        normal_vector_world = z_direction_world / torch.norm(
            z_direction_world, dim=-1
        ).unsqueeze(
            -1
        )  # (E,2,3)

        cylinder_bottom_center = self.drone.pos  # (E,2,3) cylinder bottom center
        cylinder_axis = 2.0 * self.ball_radius * normal_vector_world

        ball_to_bottom = self.ball_pos - cylinder_bottom_center  # (E,2,3)
        projection_ratio = torch.sum(
            ball_to_bottom * cylinder_axis, dim=-1
        ) / torch.sum(
            cylinder_axis * cylinder_axis, dim=-1
        )  # (E,2) projection of ball_to_bottom on cylinder_axis / cylinder_axis
        within_height = (projection_ratio >= 0) & (projection_ratio <= 1)  # (E,2)

        projection_point = (
            cylinder_bottom_center + projection_ratio.unsqueeze(-1) * cylinder_axis
        )  # (E,2,3)
        distance_to_axis = torch.norm(self.ball_pos - projection_point, dim=-1)  # (E,2)
        within_radius = distance_to_axis <= self.racket_radius  # (E,2)

        return within_height & within_radius  # (E,2)

    def check_out_of_court(self, pos):
        out_of_x = pos[..., 0].abs() > self.L / 2
        out_of_y = pos[..., 1].abs() > self.W / 2
        return out_of_x | out_of_y

    def check_hit_net(self, pos, radius):
        close_to_net = pos[..., 0].abs() < radius
        within_net_width = pos[..., 1].abs() < self.W / 2
        below_net_height = pos[..., 2] < self.H_NET
        return close_to_net & within_net_width & below_net_height

    def debug_draw_region(self):
        b_x = self.env_boundary_x
        b_y = self.env_boundary_y
        b_z_top = self.env_boundary_z_top
        b_z_bot = self.env_boundary_z_bot
        height = b_z_top - b_z_bot
        color = [(0.95, 0.43, 0.2, 1.0)]
        # [topleft, topright, botleft, botright]

        points_start, points_end = rectangular_cuboid_edges(
            2 * b_x, 2 * b_y, b_z_bot, height
        )
        points_start = [_carb_float3_add(p, self.central_env_pos) for p in points_start]
        points_end = [_carb_float3_add(p, self.central_env_pos) for p in points_end]

        colors_line = color * len(points_start)
        sizes_line = [1.0] * len(points_start)
        self.draw.draw_lines(points_start, points_end, colors_line, sizes_line)

    def debug_draw_turn(self):
        ori = self.envs_positions[self.central_env_idx].detach()
        points = self.anchor.clone()
        points[:, -1] = 0
        points = (ori + points).tolist()
        if self.turn[self.central_env_idx, 0] == 0:
            colors = [(0, 1, 0, 1), (1, 0, 0, 1)]  # green, red
        elif self.turn[self.central_env_idx, 0] == 1:
            colors = [(1, 0, 0, 1), (0, 1, 0, 1)]  # red, green
        elif self.turn[self.central_env_idx, 0] == 2:
            colors = [(1, 0, 0, 1), (1, 0, 0, 1)]  # red, red
        sizes = [30.0, 30.0]
        self.draw.draw_points(points, colors, sizes)

    def debug_draw_hit_racket(self, true_hit, ball_near_racket):
        ori = self.envs_positions[self.central_env_idx].detach()
        points = self.anchor.clone() + ori
        points = points.tolist()
        colors = []
        if ball_near_racket[0] == True:
            colors.append((0, 1, 0, 1))  # green
        elif true_hit[0] == True:
            colors.append((1, 0, 0, 1))  # red
        else:
            colors.append((1, 1, 0, 1))  # yellow
        if ball_near_racket[1] == True:
            colors.append((0, 1, 0, 1))  # green
        elif true_hit[1] == True:
            colors.append((1, 0, 0, 1))  # red
        else:
            colors.append((1, 1, 0, 1))  # yellow
        sizes = [30.0, 30.0]
        self.draw.draw_points(points, colors, sizes)

    def debug_draw_near_target(self, near_target):
        ori = self.envs_positions[self.central_env_idx].detach()
        points = [ori.tolist()]
        if self.turn[self.central_env_idx] != 2:
            colors = [(1, 0, 0, 1)]  # red
        elif near_target == True:  # turn == 2 and near_target == True
            colors = [(0, 1, 0, 1)]  # green
        else:  # turn == 2 and near_target == False
            colors = [(1, 1, 0, 1)]  # yellow
        sizes = [30.0]
        self.draw.draw_points(points, colors, sizes)

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
        # drone
        self.drone._reset_idx(env_ids, self.training)
        drone_pos = self.init_drone_pos_dist.sample(env_ids.shape)
        drone_rpy = self.init_drone_rpy_dist.sample((*env_ids.shape, 2))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(
            torch.zeros(len(env_ids), 2, 6, device=self.device), env_ids
        )

        # ball and turn
        turn = torch.zeros(len(env_ids), 1, device=self.device, dtype=torch.int64)
        self.turn[env_ids] = turn

        self.ball_last_pos[env_ids] = torch.zeros(
            len(env_ids), 1, 3, device=self.device
        )
        self.ball_last_vel[env_ids] = torch.zeros(
            len(env_ids), 1, 3, device=self.device
        )
        self.ball_before_spike_vel[env_ids] = torch.zeros(
            len(env_ids), 3, device=self.device
        )
        self.ball_after_spike_vel[env_ids] = torch.zeros(
            len(env_ids), 3, device=self.device
        )
        self.ball_spike_time[env_ids] = torch.zeros(len(env_ids), 1, device=self.device)

        ball_pos = (
            drone_pos[turn_to_mask(turn)] + self.init_ball_offset
        )  # ball initial position is on the top of the drone
        ball_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            len(env_ids), 1
        )
        self.ball.set_world_poses(
            ball_pos + self.envs_positions[env_ids], ball_rot, env_ids
        )
        self.ball.set_velocities(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids
        )
        # fix the mass now
        ball_masses = torch.ones_like(env_ids) * self.ball_mass
        self.ball.set_masses(ball_masses, env_ids)

        # env stats
        self.last_hit_step[env_ids] = -100.0
        self.stats[env_ids] = 0.0

        # draw
        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            self.ball_traj_vis.clear()
            self.draw.clear_lines()
            self.debug_draw_turn()
            self.debug_draw_hit_racket([False, False], [False, False])
            self.debug_draw_near_target([False])

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

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")].clone()
        if self.cfg.task.get("tanh_action", False):
            actions = torch.tanh(actions)
        self.effort = self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.contact_sensor.update(self.dt)

    def _compute_state_and_obs(self):
        # clone here
        self.root_state = self.drone.get_state()
        # pos, quat(4), vel, omega
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.ball_pos, _ = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()[..., :3]

        # relative position and heading
        self.rpos_ball = self.drone.pos - self.ball_pos

        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )
        rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot)
        self.drone_rot = rot

        self.rpos_drone = torch.stack(
            [
                self.drone.pos[..., 1, :] - self.drone.pos[..., 0, :],
                self.drone.pos[..., 0, :] - self.drone.pos[..., 1, :],
            ],
            dim=1,
        )  # (E,2,3)

        rpos_anchor = self.drone.pos - self.anchor  # (E,2,3)

        obs = [
            self.root_state,  # (E,2,23)
            rpos_anchor,  # (E,2,3)
            self.rpos_drone[..., :3],  # (E,2,3)
            self.rpos_ball,  # (E,2,3)
            self.ball_vel.expand(-1, 2, 3),  # (E,2,3)
            turn_to_obs(self.turn),  # (E,2,2)
            self.id,  # (E,2,2)
        ]
        # [drone_num(2),
        # each_obs_dim: root_state(rpos_anchor)+rpos_drone(3)+rpos_ball(3)+ball_vel(3)+turn(1)]

        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs.append(t.expand(-1, 2, self.time_encoding_dim))

        obs = torch.cat(obs, dim=-1)

        if self._should_render(0):
            central_env_pos = self.envs_positions[self.central_env_idx]
            ball_plot_pos = (
                self.ball_pos[self.central_env_idx] + central_env_pos
            ).tolist()  # [2, 3]
            if len(self.ball_traj_vis) > 1:
                point_list_0 = self.ball_traj_vis[-1]
                point_list_1 = ball_plot_pos
                colors = [(0.1, 1.0, 0.1, 1.0)]
                sizes = [1.5]
                self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
            self.ball_traj_vis.append(ball_plot_pos)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.num_envs,
        )

    def _compute_reward_and_done(self):
        # ball misbehave
        ball_too_low = self.ball_pos[..., 2] < 2 * self.ball_radius  # (E, 1)
        ball_too_high = self.ball_pos[..., 2] > 16  # (E, 1)
        ball_hit_net = self.check_hit_net(self.ball_pos, self.ball_radius)  # (E, 1)
        ball_out_of_court = self.check_out_of_court(self.ball_pos)  # (E, 1)
        ball_misbehave = (
            ball_too_low | ball_too_high | ball_hit_net | ball_out_of_court
        )  # (E, 1)

        # drone misbehave
        drone_too_low = self.drone.pos[..., 2] < self.racket_radius  # (E, 2)
        drone_cross_net = self.drone.pos[..., 0] < 0  # (E, 2)
        drone_misbehave = drone_too_low | drone_cross_net  # (E, 2)

        # drone hit ball
        ball_contact_forces = self.contact_sensor.data.net_forces_w  # (E, 1, 3)
        hit_drone = self.rpos_ball.norm(p=2, dim=-1).argmin(
            dim=1, keepdim=True
        )  # (E, 1) which drone is closer to the ball
        sim_hit = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.bool
        )  # (E, 2)
        sim_hit[turn_to_mask(hit_drone)] = ball_contact_forces.any(-1).squeeze(-1)

        true_hit_step_gap = 3
        true_hit = sim_hit & (
            (self.progress_buf.unsqueeze(-1) - self.last_hit_step) > true_hit_step_gap
        )
        wrong_hit_sim = sim_hit & (
            (self.progress_buf.unsqueeze(-1) - self.last_hit_step) <= true_hit_step_gap
        )
        self.last_hit_step[sim_hit] = self.progress_buf[sim_hit.any(-1)]

        wrong_hit_turn: torch.Tensor = true_hit & (
            self.turn != hit_drone
        )
        ball_near_racket = self.check_ball_near_racket()
        wrong_hit_racket = true_hit & torch.logical_not(ball_near_racket)
        wrong_hit = wrong_hit_turn | wrong_hit_racket
        success_hit = true_hit & torch.logical_not(wrong_hit)

        self.turn = self.turn + success_hit.any(dim=-1, keepdim=True)

        # ball cross net
        ball_cross = (self.ball_pos[..., 0] <= -self.ball_radius) & (
            self.ball_last_pos[..., 0] > -self.ball_radius
        )
        success_cross = (self.turn == 2) & ball_cross  # (E, 1)
        self.ball_last_pos = self.ball_pos

        # log info
        self.ball_spike_time += success_hit[..., 1].unsqueeze(
            -1
        ) * self.progress_buf.unsqueeze(-1)
        self.ball_before_spike_vel += (
            success_hit[..., 1].unsqueeze(-1) * self.ball_last_vel[:, 0]
        )
        self.ball_after_spike_vel += (
            success_hit[..., 1].unsqueeze(-1) * self.ball_vel[:, 0]
        )
        self.ball_last_vel = self.ball_vel

        # misbehave penalty
        _misbehave_penalty_coeff = 10.0
        penalty_ball_misbehave = (
            _misbehave_penalty_coeff * ball_misbehave
        )  # share, sparse, (E, 1)
        penalty_drone_misbehave = _misbehave_penalty_coeff * drone_misbehave.any(
            -1, keepdim=True
        )  # share, sparse, (E, 1)
        penalty_wrong_hit = (
            _misbehave_penalty_coeff * wrong_hit
        )  # individual, sparse, (E, 2)

        misbehave_penalty = (
            penalty_ball_misbehave + penalty_drone_misbehave + penalty_wrong_hit
        )  # (E, 2)

        # task reward
        _task_reward_coeff = 5.0
        reward_success_hit = _task_reward_coeff * success_hit.any(
            -1, keepdim=True
        )  # share, sparse, (E, 1)

        downward_spike = success_hit[..., 1].unsqueeze(-1) & (
            self.ball_vel[..., 2] < 0
        )  # (E, 1)
        reward_downward_spike = (
            _task_reward_coeff * downward_spike
        )  # share, sparse, (E, 1)

        reward_success_cross = (
            _task_reward_coeff * success_cross
        )  # share, sparse, (E, 1)

        dist_to_target_xy = torch.norm(
            self.ball_pos[..., :2] - self.target, p=2, dim=-1
        )  # (E, 1)
        in_target = (
            ball_too_low & (self.turn == 2) & (dist_to_target_xy < self.target_radius)
        )  # (E, 1)
        reward_in_target = _task_reward_coeff * in_target  # share, sparse, (E, 1)

        _dist_coeff = 0.05
        dist_to_anchor = torch.norm(self.drone.pos - self.anchor, p=2, dim=-1)  # (E, 2)
        penalty_dist_to_anchor = _dist_coeff * (
            dist_to_anchor - self.anchor_radius
        ).clamp(
            min=0
        )  # individual, sparse, (E, 2)
        penalty_dist_to_anchor = penalty_dist_to_anchor.mean(
            -1, keepdim=True
        )  # share, sparse, (E, 1)

        task_reward = (
            reward_success_hit
            + reward_downward_spike
            + reward_success_cross
            + reward_in_target
            - penalty_dist_to_anchor
        )  # (E, 1)

        # shaping reward
        dist_to_ball_xy = torch.norm(
            self.drone.pos[..., :2] - self.ball_pos[..., :2], p=2, dim=-1
        )  # (E, 2)
        reward_dist_to_ball = (
            _dist_coeff * turn_to_reward(self.turn) / (1 + dist_to_ball_xy)
        )  # individual, dense, (E, 2)
        reward_dist_to_ball = reward_dist_to_ball.mean(
            -1, keepdim=True
        )  # share, dense, (E, 1)

        _direction_reward_coeff = 1.0
        drone0_hit_dir_xy = self.drone.pos[:, 1, :2] - self.drone.pos[:, 0, :2]
        drone1_hit_dir_xy = self.target - self.drone.pos[:, 1, :2]
        hit_dir_xy = (
            success_hit[..., 0].unsqueeze(-1) * drone0_hit_dir_xy
            + success_hit[..., 1].unsqueeze(-1) * drone1_hit_dir_xy
        )
        ball_dir_xy = self.ball_vel[:, 0, :2]
        cosine_similarity = NNF.cosine_similarity(
            hit_dir_xy, ball_dir_xy, dim=-1
        ).unsqueeze(
            -1
        )  # (E, 1)
        reward_hit_direction = (
            _direction_reward_coeff * success_hit * cosine_similarity
        )  # individual, sparse, (E, 2)

        _spike_reward_coeff = 0.2
        reward_spike_velocity = (
            _spike_reward_coeff
            * success_hit[..., 1].unsqueeze(-1)
            * (5 - self.ball_vel[..., 2]).clamp(min=0)
        )  # share, sparse, (E, 1)

        _target_reward_coeff = 0.2
        reward_dist_to_target = (
            _target_reward_coeff
            * (self.turn == 2)
            * ball_too_low
            * (10 - dist_to_target_xy).clamp(min=0)
        )  # share, sparse, (E, 1)

        shaping_reward = (
            reward_dist_to_ball
            + reward_hit_direction
            + reward_spike_velocity
            + reward_dist_to_target
        )  # (E, 2)

        reward = -misbehave_penalty + task_reward + self.reward_shaping * shaping_reward

        # done
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(
            -1
        )  # [E, 1]
        terminated = (
            ball_misbehave
            | drone_misbehave.any(-1, keepdim=True)
            | wrong_hit.any(-1, keepdim=True)
        )  # [E, 1]
        done: torch.Tensor = truncated | terminated  # [E, 1]

        # render
        if self._should_render(0):
            self.debug_draw_hit_racket(
                true_hit[self.central_env_idx], ball_near_racket[self.central_env_idx]
            )
            if success_hit[self.central_env_idx].any():
                self.debug_draw_turn()
            near_target = dist_to_target_xy < self.target_radius
            if (
                success_hit[self.central_env_idx].any()
                or near_target[self.central_env_idx]
            ):
                self.debug_draw_near_target(near_target[self.central_env_idx])

        # log stats
        self.stats["return"].add_(reward.mean(dim=-1, keepdim=True))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        self.stats["done"].add_(done.float())
        self.stats["truncated"].add_(truncated.float())
        self.stats["terminated"].add_(terminated.float())

        self.stats["ball_misbehave"] = ball_misbehave.float()
        self.stats["ball_too_low"] = ball_too_low.float()
        self.stats["ball_too_high"] = ball_too_high.float()
        self.stats["ball_hit_net"] = ball_hit_net.float()
        self.stats["ball_out_of_court"] = ball_out_of_court.float()

        self.stats["drone_misbehave"] = drone_misbehave.any(-1, keepdim=True).float()
        self.stats["drone0_misbehave"] = drone_misbehave[..., 0].unsqueeze(-1).float()
        self.stats["drone1_misbehave"] = drone_misbehave[..., 1].unsqueeze(-1).float()
        self.stats["drone_too_low"] = drone_too_low.any(-1, keepdim=True).float()
        self.stats["drone0_too_low"] = drone_too_low[..., 0].unsqueeze(-1).float()
        self.stats["drone1_too_low"] = drone_too_low[..., 1].unsqueeze(-1).float()
        self.stats["drone_cross_net"] = drone_cross_net.any(-1, keepdim=True).float()
        self.stats["drone0_cross_net"] = drone_cross_net[..., 0].unsqueeze(-1).float()
        self.stats["drone1_cross_net"] = drone_cross_net[..., 1].unsqueeze(-1).float()

        self.stats["wrong_hit"] = wrong_hit.any(-1, keepdim=True).float()
        self.stats["drone0_wrong_hit"] = wrong_hit[..., 0].unsqueeze(-1).float()
        self.stats["drone1_wrong_hit"] = wrong_hit[..., 1].unsqueeze(-1).float()
        self.stats["wrong_hit_turn"] = wrong_hit_turn.any(-1, keepdim=True).float()
        self.stats["drone0_wrong_hit_turn"] = (
            wrong_hit_turn[..., 0].unsqueeze(-1).float()
        )
        self.stats["drone1_wrong_hit_turn"] = (
            wrong_hit_turn[..., 1].unsqueeze(-1).float()
        )
        self.stats["wrong_hit_racket"] = wrong_hit_racket.any(-1, keepdim=True).float()
        self.stats["drone0_wrong_hit_racket"] = (
            wrong_hit_racket[..., 0].unsqueeze(-1).float()
        )
        self.stats["drone1_wrong_hit_racket"] = (
            wrong_hit_racket[..., 1].unsqueeze(-1).float()
        )

        self.stats["misbehave_penalty"].add_(
            misbehave_penalty.mean(dim=-1, keepdim=True)
        )
        self.stats["drone0_misbehave_penalty"].add_(
            misbehave_penalty[..., 0].unsqueeze(-1)
        )
        self.stats["drone1_misbehave_penalty"].add_(
            misbehave_penalty[..., 1].unsqueeze(-1)
        )
        self.stats["penalty_ball_misbehave"].add_(penalty_ball_misbehave)
        self.stats["penalty_drone_misbehave"].add_(penalty_drone_misbehave)
        self.stats["penalty_wrong_hit"].add_(
            penalty_wrong_hit.mean(dim=-1, keepdim=True)
        )
        self.stats["drone0_penalty_wrong_hit"].add_(
            penalty_wrong_hit[..., 0].unsqueeze(-1)
        )
        self.stats["drone1_penalty_wrong_hit"].add_(
            penalty_wrong_hit[..., 1].unsqueeze(-1)
        )

        self.stats["task_reward"].add_(task_reward)
        self.stats["reward_success_hit"].add_(reward_success_hit)
        self.stats["reward_downward_spike"].add_(reward_downward_spike)
        self.stats["reward_success_cross"].add_(reward_success_cross)
        self.stats["reward_in_target"].add_(reward_in_target)
        self.stats["penalty_dist_to_anchor"].add_(penalty_dist_to_anchor)

        if self.reward_shaping:
            self.stats["shaping_reward"].add_(shaping_reward.mean(dim=-1, keepdim=True))
            self.stats["drone0_shaping_reward"].add_(
                shaping_reward[..., 0].unsqueeze(-1)
            )
            self.stats["drone1_shaping_reward"].add_(
                shaping_reward[..., 1].unsqueeze(-1)
            )
            self.stats["reward_dist_to_ball"].add_(reward_dist_to_ball)
            self.stats["reward_hit_direction"].add_(
                reward_hit_direction.mean(dim=-1, keepdim=True)
            )
            self.stats["drone0_reward_hit_direction"].add_(
                reward_hit_direction[..., 0].unsqueeze(-1)
            )
            self.stats["drone1_reward_hit_direction"].add_(
                reward_hit_direction[..., 1].unsqueeze(-1)
            )
            self.stats["reward_spike_velocity"].add_(reward_spike_velocity)
            self.stats["reward_dist_to_target"].add_(reward_dist_to_target)

        self.stats["num_sim_hits"].add_(sim_hit.any(-1, keepdim=True).float())
        self.stats["drone0_num_sim_hits"].add_(sim_hit[..., 0].unsqueeze(-1).float())
        self.stats["drone1_num_sim_hits"].add_(sim_hit[..., 1].unsqueeze(-1).float())
        self.stats["num_true_hits"].add_(true_hit.any(-1, keepdim=True).float())
        self.stats["drone0_num_true_hits"].add_(true_hit[..., 0].unsqueeze(-1).float())
        self.stats["drone1_num_true_hits"].add_(true_hit[..., 1].unsqueeze(-1).float())
        self.stats["num_success_hits"].add_(success_hit.any(-1, keepdim=True).float())
        self.stats["drone0_num_success_hits"].add_(
            success_hit[..., 0].unsqueeze(-1).float()
        )
        self.stats["drone1_num_success_hits"].add_(
            success_hit[..., 1].unsqueeze(-1).float()
        )
        self.stats["wrong_hit_sim"].add_(wrong_hit_sim.any(-1, keepdim=True).float())
        self.stats["drone0_wrong_hit_sim"].add_(
            wrong_hit_sim[..., 0].unsqueeze(-1).float()
        )
        self.stats["drone1_wrong_hit_sim"].add_(
            wrong_hit_sim[..., 1].unsqueeze(-1).float()
        )
        self.stats["downward_spike"].add_(downward_spike.float())

        self.stats["num_ball_cross"].add_(ball_cross.float())
        self.stats["num_success_cross"].add_(success_cross.float())
        if success_cross.any():
            self.update_mean_stats(
                "cross_height",
                self.ball_pos[..., 2],
                "num_success_cross",
                success_cross,
            )
        self.stats["in_target"] = in_target.float()

        self.update_mean_stats(
            "drone0_x", self.drone.pos[:, 0, 0].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats(
            "drone0_y", self.drone.pos[:, 0, 1].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats(
            "drone0_z", self.drone.pos[:, 0, 2].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats(
            "drone0_dist_to_anchor", dist_to_anchor[:, 0].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats(
            "drone1_x", self.drone.pos[:, 1, 0].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats(
            "drone1_y", self.drone.pos[:, 1, 1].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats(
            "drone1_z", self.drone.pos[:, 1, 2].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats(
            "drone1_dist_to_anchor", dist_to_anchor[:, 1].unsqueeze(-1), "episode_len"
        )

        if success_hit[..., 0].any():
            self.update_mean_stats(
                "drone0_hit_x",
                self.drone.pos[:, 0, 0].unsqueeze(-1),
                "drone0_num_success_hits",
                success_hit[..., 0].unsqueeze(-1),
            )
            self.update_mean_stats(
                "drone0_hit_y",
                self.drone.pos[:, 0, 1].unsqueeze(-1),
                "drone0_num_success_hits",
                success_hit[..., 0].unsqueeze(-1),
            )
            self.update_mean_stats(
                "drone0_hit_z",
                self.drone.pos[:, 0, 2].unsqueeze(-1),
                "drone0_num_success_hits",
                success_hit[..., 0].unsqueeze(-1),
            )
            self.update_mean_stats(
                "drone0_hit_dist_to_anchor",
                dist_to_anchor[:, 0].unsqueeze(-1),
                "drone0_num_success_hits",
                success_hit[..., 0].unsqueeze(-1),
            )
        if success_hit[..., 1].any():
            self.update_mean_stats(
                "drone1_hit_x",
                self.drone.pos[:, 1, 0].unsqueeze(-1),
                "drone1_num_success_hits",
                success_hit[..., 1].unsqueeze(-1),
            )
            self.update_mean_stats(
                "drone1_hit_y",
                self.drone.pos[:, 1, 1].unsqueeze(-1),
                "drone1_num_success_hits",
                success_hit[..., 1].unsqueeze(-1),
            )
            self.update_mean_stats(
                "drone1_hit_z",
                self.drone.pos[:, 1, 2].unsqueeze(-1),
                "drone1_num_success_hits",
                success_hit[..., 1].unsqueeze(-1),
            )
            self.update_mean_stats(
                "drone1_hit_dist_to_anchor",
                dist_to_anchor[:, 0].unsqueeze(-1),
                "drone1_num_success_hits",
                success_hit[..., 1].unsqueeze(-1),
            )

        self.stats["ball_before_spike_vel"] = torch.norm(
            self.ball_before_spike_vel, p=2, dim=-1, keepdim=True
        )
        self.stats["ball_before_spike_vel_x"] = self.ball_before_spike_vel[
            :, 0
        ].unsqueeze(-1)
        self.stats["ball_before_spike_vel_y"] = self.ball_before_spike_vel[
            :, 1
        ].unsqueeze(-1)
        self.stats["ball_before_spike_vel_z"] = self.ball_before_spike_vel[
            :, 2
        ].unsqueeze(-1)
        self.stats["ball_after_spike_vel"] = torch.norm(
            self.ball_after_spike_vel, p=2, dim=-1, keepdim=True
        )
        self.stats["ball_after_spike_vel_x"] = self.ball_after_spike_vel[
            :, 0
        ].unsqueeze(-1)
        self.stats["ball_after_spike_vel_y"] = self.ball_after_spike_vel[
            :, 1
        ].unsqueeze(-1)
        self.stats["ball_after_spike_vel_z"] = self.ball_after_spike_vel[
            :, 2
        ].unsqueeze(-1)
        self.stats["ball_spike_to_done_time"] = (
            self.stats["episode_len"] - self.ball_spike_time
        )

        self.stats["ball_done_x"] = self.ball_pos[..., 0]
        self.stats["ball_done_y"] = self.ball_pos[..., 1]
        self.stats["ball_done_z"] = self.ball_pos[..., 2]
        self.stats["ball_done_dist_to_target_xy"] = dist_to_target_xy

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.num_envs,
        )
