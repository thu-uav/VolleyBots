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

import numpy as np
from omni.isaac.orbit.sensors import ContactSensor, ContactSensorCfg

from .racket import (
    ball_post_vel_without_kd,
    get_ball_traj_without_kd,
    get_uav_collision_data_without_kd,
)
from .return_ball import cal_ori


def calculate_ball_cross_the_net(ball_vel: torch.Tensor):
    if ball_vel[0] < 0:
        return True
    else:
        return False


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


class MultiAttackVolleyballHard(IsaacEnv):
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
        self.board_mass: float = cfg.task.board_mass
        self.board_radius: float = cfg.task.board_radius
        self.board_height: float = cfg.task.board_height
        self.env_num: float = cfg.task.env.num_envs
        self.dist_max = cfg.task.dist_max
        self.rpy_max = cfg.task.rpy_max
        super().__init__(cfg, headless)

        # ensure just creating and excuting one new trajectory for board
        self.board_create_traj = torch.ones(self.num_envs, device=self.device)
        self.board_excute_traj = torch.zeros(self.num_envs, device=self.device)

        # the target pose for ball to return
        # the height of target pose must be higher than the height of collision
        self.hit_target_pose = torch.tensor([3.0, 1.0, 1.3], device=self.device)

        # x, y, z boundary for drone
        self.env_boundary_x = self.L / 2
        self.env_boundary_y = self.W / 2
        # self.env_boundary_z_top = 2.0
        # self.env_boundary_z_bot = 0.0

        # record new traj for board
        # self.p_l,self.v_l,self.o_l,self.p_l_re,self.v_l_re,self.o_l_re,self.ls = [],[],[],[],[],[],[]

        # for _ in range(self.num_envs):
        #     self.p_l.append([])
        #     self.v_l.append([])
        #     self.o_l.append([])
        #     self.p_l_re.append([])
        #     self.v_l_re.append([])
        #     self.o_l_re.append([])
        self.p_l = torch.zeros(
            (self.num_envs, 1, 3), device=self.device, dtype=self.dtype
        )
        self.v_l = torch.zeros(
            (self.num_envs, 1, 6), device=self.device, dtype=self.dtype
        )
        self.o_l = torch.zeros(
            (self.num_envs, 1, 4), device=self.device, dtype=self.dtype
        )
        self.rpy = torch.zeros(
            (self.num_envs, 1, 3), device=self.device, dtype=self.dtype
        )
        self.rpy_per_step = torch.zeros(
            (self.num_envs, 1, 3), device=self.device, dtype=self.dtype
        )
        self.dist = torch.zeros(
            (self.num_envs, 1, 3), device=self.device, dtype=self.dtype
        )
        self.current_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # env paras
        self.time_encoding = self.cfg.task.time_encoding
        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )
        self.return_ball = 0
        self.env_reset = 0
        self.env_reset_count = torch.zeros(
            self.num_envs, device=self.device, dtype=self.dtype
        )
        self.env_return_ball_count = torch.zeros(
            self.num_envs, device=self.device, dtype=self.dtype
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

        # board paras
        self.board = RigidPrimView(
            "/World/envs/env_*/board",
            reset_xform_properties=False,
            track_contact_forces=False,
        )
        self.board.initialize()

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

        self.init_board_offset = torch.tensor(cfg.task.board_offset, device=self.device)

        # utils
        self.turn = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.int64)
        self.last_hit_step = torch.zeros(self.num_envs, 2, device=self.device)
        self.ball_too_low_timer = torch.full(
            (self.num_envs, 1), -1, dtype=torch.float32, device=self.device
        )
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

        board = objects.DynamicCylinder(
            prim_path="/World/envs/env_0/board",
            radius=self.board_radius,
            height=self.board_height,
            mass=self.board_mass,
            color=torch.tensor([0.0, 0.2, 0.2]),
            physics_material=material,
        )

        cr_api_x = PhysxSchema.PhysxContactReportAPI.Apply(board.prim)
        cr_api_x.CreateThresholdAttr().Set(0.0)

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
                "ball_too_low_expired": UnboundedContinuousTensorSpec(1),
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
                "in_target_add": UnboundedContinuousTensorSpec(1),
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
                "env_success_rate": UnboundedContinuousTensorSpec(1),
                "env_return_ball_count": UnboundedContinuousTensorSpec(1),
                "env_reset_count": UnboundedContinuousTensorSpec(1),
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
        if ball_near_racket[0] == True:  # 球进入击球范围
            colors.append((0, 1, 0, 1))  # green
        elif true_hit[0] == True:  # 不在击球范围但有击球发生
            colors.append((1, 0, 0, 1))  # red
        else:  # 不在击球范围且无击球发生
            colors.append((1, 1, 0, 1))  # yellow
        if ball_near_racket[1] == True:  # 球进入击球范围
            colors.append((0, 1, 0, 1))  # green
        elif true_hit[1] == True:  # 不在击球范围但有击球发生
            colors.append((1, 0, 0, 1))  # red
        else:  # 不在击球范围且无击球发生
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
        self.env_reset_count[env_ids] = 0
        self.env_return_ball_count[env_ids] = 0
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
        # turn = torch.randint(0, 2, (len(env_ids),), device=self.device) # random initial turn
        turn = torch.zeros(len(env_ids), 1, device=self.device, dtype=torch.int64)
        self.turn[env_ids] = turn

        self.env_ids = env_ids

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

        # board
        board_pos = self.init_board_offset
        self.dtype = board_pos.dtype
        self.board.set_world_poses(
            board_pos + self.envs_positions[env_ids], ball_rot, env_ids
        )
        self.board.set_velocities(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids
        )
        # fix the mass now
        board_masses = torch.ones_like(env_ids) * self.board_mass
        self.board.set_masses(board_masses, env_ids)

        # env stats
        self.last_hit_step[env_ids] = -100.0
        self.stats[env_ids] = 0.0
        self.ball_too_low_timer[env_ids] = -1.0

        # draw
        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            self.ball_traj_vis.clear()
            self.draw.clear_lines()
            # self.debug_draw_region()
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

        ball_pose_now = self.ball_pos.squeeze()
        ball_vel_now = self.ball_vel.squeeze()
        env_positions = self.envs_positions.squeeze()

        # 获取满足条件的 env
        mask = (
            (self.turn[:, 0] == 2)
            & (ball_vel_now[:, 0] < -2.0)
            & (ball_vel_now[:, 2] < 0)
            & (self.board_create_traj == 1)
            & (ball_pose_now[:, 2] >= 1.2)
        )
        env_indices = torch.nonzero(mask).squeeze(-1)  # 取出符合条件的环境索引

        if env_indices.numel() > 0:  # 确保有满足条件的 env

            ball_predict_pose, ball_predict_vel, ball_predict_t = (
                get_ball_traj_without_kd(
                    ball_pose_now[env_indices],
                    ball_vel_now[env_indices],
                    1.1,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

            valid_mask = (
                (ball_predict_t >= 0.0)
                & (-9 <= ball_predict_pose[:, 0])
                & (ball_predict_pose[:, 0] <= 0)
                & (-4.5 <= ball_predict_pose[:, 1])
                & (ball_predict_pose[:, 1] <= 4.5)
            )
            valid_indices = env_indices[
                valid_mask
            ]  # 选取满足 ball_predict_t 和位置信息的 env

            if valid_indices.numel() > 0:

                ball_vel_post = ball_post_vel_without_kd(
                    self.hit_target_pose.view(-1, 3),
                    ball_predict_pose[valid_mask].view(-1, 3),
                    device=self.device,
                    dtype=self.dtype,
                )

                uav_data = get_uav_collision_data_without_kd(
                    ball_predict_pose[valid_mask],
                    ball_predict_vel[valid_mask],
                    0.77,
                    ball_vel_post,
                    device=self.device,
                    dtype=self.dtype,
                )

                self.env_reset_count[valid_indices] = 1

                self.board_create_traj[valid_indices] = 0
                self.board_excute_traj[valid_indices] = 1

                idx = self.current_index[valid_indices]  # 获取当前索引
                """
                只有1个轨迹点, 实际上 idx 这里都是0
                there may be some bugs.
                uav_data是用valid_mask算的，ball_predict_t是用env_indices算的
                """
                # 获取位移向量
                delta_dist = uav_data[:, :3] - self.init_board_offset  # (N,3)
                # 计算单位时间应移动的实际距离
                dist = torch.norm(delta_dist, dim=-1) / ball_predict_t[valid_mask]

                # 判断单位时间内板的移动距离是否大于限制
                # 如果是，设置移动距离为最大限制，否则设置为所求
                norm_uav_pos = torch.norm(delta_dist, dim=-1, keepdim=True)  #  (N,1)
                scale = torch.where(
                    dist.unsqueeze(-1) >= self.dist_max,
                    self.dist_max / norm_uav_pos,
                    dist.unsqueeze(-1) / norm_uav_pos,
                )  #  (N,1)

                self.dist[valid_indices, idx, :] = scale * delta_dist * self.dt
                self.p_l[valid_indices, idx, :] = self.dist[valid_indices, idx, :]

                # 获取角度速率, 分别求roll,pitch,yaw, 为了比较此处全部取绝对值
                # print("print(uav_data[:, 6:9])", uav_data[:, 6:9])
                rpy_r_vel = torch.abs(uav_data[:, 6]) / ball_predict_t[valid_mask]
                rpy_p_vel = torch.abs(uav_data[:, 7]) / ball_predict_t[valid_mask]
                rpy_y_vel = torch.abs(uav_data[:, 8]) / ball_predict_t[valid_mask]

                # 设置比较所需tensor
                rpy_m = (
                    torch.ones(rpy_r_vel.shape, device=self.device, dtype=self.dtype)
                    * self.rpy_max
                )

                # 计算实际的正负
                sign = torch.sign(uav_data[:, 6:9])
                # print(sign)
                # 判断单位时间内板的各向角度移动是否大于限制
                # 如果是，设置角度移动为最大限制，否则设置为所求
                rpy_r_vel = (
                    torch.where(rpy_r_vel >= rpy_m, rpy_m, rpy_r_vel)
                    * sign[:, 0]
                    * self.dt
                )
                rpy_p_vel = (
                    torch.where(rpy_p_vel >= rpy_m, rpy_m, rpy_p_vel)
                    * sign[:, 1]
                    * self.dt
                )
                rpy_y_vel = (
                    torch.where(rpy_y_vel >= rpy_m, rpy_m, rpy_y_vel)
                    * sign[:, 2]
                    * self.dt
                )

                self.rpy_per_step[valid_indices, idx] = torch.stack(
                    (rpy_r_vel, rpy_p_vel, rpy_y_vel), dim=1
                )
                qw, qx, qy, qz = cal_ori(
                    self.rpy_per_step[valid_indices, idx][:, 0],
                    self.rpy_per_step[valid_indices, idx][:, 1],
                    self.rpy_per_step[valid_indices, idx][:, 2],
                    device=self.device,
                )

                self.o_l[valid_indices, idx] = torch.stack(
                    (qw, qx, qy, qz), dim=1
                )  # 单位方向
                self.rpy[valid_indices, idx] = self.rpy_per_step[valid_indices, idx]
                self.v_l[valid_indices, idx] = torch.cat(
                    (
                        uav_data[:, 3:6],
                        torch.zeros(
                            (len(valid_indices), 3),
                            device=self.device,
                            dtype=self.dtype,
                        ),
                    ),
                    dim=1,
                )  # 速度

                self.current_index[valid_indices] += 1  # 更新轨迹索引

        # 获取执行轨迹的环境索引,实际上和上面的valid_indices是一样的
        excute_mask = (self.board_excute_traj == 1) & (self.board_create_traj == 0)
        excute_indices = torch.nonzero(excute_mask).squeeze(-1)

        if excute_indices.numel() > 0:
            # 选择 hit 还是 recover
            hit_mask = ball_vel_now[excute_indices, 0] <= 0
            recover_mask = ~hit_mask

            hit_indices = excute_indices[hit_mask]
            recover_indices = excute_indices[recover_mask]
            # print("hit_indices",hit_indices)
            # print("recover_indices",recover_indices)
            # **Hit 逻辑**
            """self.board.set_world_poses(board_pos + self.envs_positions[env_ids], ball_rot, env_ids)
            there may be some bugs.
            """

            if hit_indices.numel() > 0:
                # print("Hit")
                self.board_pos, _ = self.get_env_poses(self.board.get_world_poses())
                # print("self.board_pos",self.board_pos)
                self.board.set_world_poses(
                    positions=self.p_l[hit_indices, 0]
                    + self.envs_positions[hit_indices]
                    + self.init_board_offset,
                    # positions=self.envs_positions[hit_indices] + self.init_board_offset,
                    orientations=self.o_l[hit_indices, 0],
                    env_indices=hit_indices,
                )
                self.board.set_velocities(
                    self.v_l[hit_indices, 0], env_indices=hit_indices
                )
                self.board_pos, _ = self.get_env_poses(self.board.get_world_poses())

                self.p_l[hit_indices, 0] = (
                    self.p_l[hit_indices, 0] + self.dist[hit_indices, 0]
                )
                self.rpy[hit_indices, 0] = (
                    self.rpy[hit_indices, 0] + self.rpy_per_step[hit_indices, 0]
                )
                qw, qx, qy, qz = cal_ori(
                    self.rpy[hit_indices, 0][:, 0],
                    self.rpy[hit_indices, 0][:, 1],
                    self.rpy[hit_indices, 0][:, 2],
                    device=self.device,
                )
                self.o_l[hit_indices, 0] = torch.stack([qw, qx, qy, qz], dim=1)

            if recover_indices.numel() > 0:
                self.env_return_ball_count[recover_indices] = 1
                self.board_excute_traj[recover_indices] = 0
                self.board_create_traj[recover_indices] = 1
                self.p_l[recover_indices, 0] = 0  # 清空轨迹
                self.v_l[recover_indices, 0] = 0
                self.o_l[recover_indices, 0] = 0
                self.dist[recover_indices, 0] = 0
                self.rpy_per_step[recover_indices, 0] = 0
                self.rpy[recover_indices, 0] = 0
                self.current_index[recover_indices] = 0  # 轨迹索引归零
            # self.return_ball = 0

        # 确保 board 在 reset 时返回初始状态
        reset_mask = ball_vel_now[:, 0] == 0
        reset_indices = torch.nonzero(reset_mask).squeeze(-1)

        if reset_indices.numel() > 0:
            self.board_create_traj[reset_indices] = 1
            self.board_excute_traj[reset_indices] = 0

        # 创建新轨迹
        create_mask = self.board_create_traj == 1
        create_indices = torch.nonzero(create_mask).squeeze(-1)

        if create_indices.numel() > 0:
            self.board.set_world_poses(
                positions=self.init_board_offset + env_positions[create_indices],
                orientations=torch.tensor(
                    [1.0, 0.0, 0.0, 0], device=self.device
                ).expand(create_indices.shape[0], 4),
                env_indices=create_indices,
            )
            self.board.set_velocities(
                torch.zeros((create_indices.shape[0], 6), device=self.device),
                env_indices=create_indices,
            )
            self.p_l[create_indices] = 0  # 清空轨迹
            self.v_l[create_indices] = 0
            self.o_l[create_indices] = 0
            self.current_index[create_indices] = 0  # 轨迹索引归零

        # 更新 `board_create_traj` 状态
        update_mask = ball_vel_now[:, 0] >= 0.5
        update_indices = torch.nonzero(update_mask).squeeze(-1)

        if update_indices.numel() > 0:
            self.board_create_traj[update_indices] = 1

        env_success_rate = self.env_return_ball_count / (self.env_reset_count + 1e-6)
        mean_success_rate = torch.mean(env_success_rate)
        self.stats["env_success_rate"] = env_success_rate.unsqueeze(-1).float()

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.contact_sensor.update(self.dt)

    def _compute_state_and_obs(self):

        self.root_state = self.drone.get_state()
        # pos, quat(4), vel, omega
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.ball_pos, _ = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()[..., :3]

        self.board_pos, _ = self.get_env_poses(self.board.get_world_poses())
        self.board_vel = self.board.get_velocities()[..., :3]
        # relative position and heading
        self.rpos_ball = self.drone.pos - self.ball_pos

        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )
        rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot)
        self.drone_rot = rot

        self.rpos_drone = torch.stack(
            [
                # [..., drone_id, [x, y, z]]
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
        # print("obs", [o.shape for o in obs])
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

        # 只在 ball_too_low 第一次发生时更新计时器
        self.ball_too_low_timer[ball_too_low & (self.ball_too_low_timer == -1)] = (
            self.progress_buf.unsqueeze(-1)[
                ball_too_low & (self.ball_too_low_timer == -1)
            ]
        )

        ball_too_low_expired = (self.ball_too_low_timer != -1) & (
            (self.progress_buf.unsqueeze(-1) - self.ball_too_low_timer) >= 0
        )

        ball_too_high = self.ball_pos[..., 2] > 16  # (E, 1)
        ball_hit_net = self.check_hit_net(self.ball_pos, self.ball_radius)  # (E, 1)
        ball_out_of_court = self.check_out_of_court(self.ball_pos)  # (E, 1)
        ball_misbehave = (
            ball_too_low | ball_too_high | ball_hit_net | ball_out_of_court
        )  # (E, 1)

        # drone misbehave
        drone_too_low = self.drone.pos[..., 2] < 2 * self.racket_radius  # (E, 2)
        drone_cross_net = self.drone.pos[..., 0] < 0  # (E, 2)
        drone_misbehave = drone_too_low | drone_cross_net  # (E, 2)
        # drone_misbehave =  drone_too_low

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
        )  # (E, 2) 击球时间大于3个step才是正确的一次击球
        wrong_hit_sim = sim_hit & (
            (self.progress_buf.unsqueeze(-1) - self.last_hit_step) <= true_hit_step_gap
        )  # (E, 2) 击球时间小于3个step则为仿真器错误击球
        self.last_hit_step[sim_hit] = self.progress_buf[sim_hit.any(-1)]

        wrong_hit_turn: torch.Tensor = true_hit & (
            self.turn != hit_drone
        )  # (E, 2) 在非该无人机击球回合，该无人机击球成功，则为错误击球
        ball_near_racket = self.check_ball_near_racket()
        wrong_hit_racket = true_hit & torch.logical_not(ball_near_racket)
        wrong_hit = wrong_hit_turn | wrong_hit_racket  # (E, 2) 错误击球
        success_hit = true_hit & torch.logical_not(wrong_hit)  # (E, 2) 正确击球

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
        # in_target = ball_too_low & (self.turn == 2) & (dist_to_target_xy < self.target_radius)  # (E, 1)
        env_return_ball_count = (
            self.env_return_ball_count.clone().unsqueeze(-1).to(torch.bool)
        )
        env_reset_count = self.env_reset_count.clone().unsqueeze(-1).to(torch.bool)
        # self.env_return_ball_count = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        # self.env_reset_count = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        # print("env_return_ball_count",env_return_ball_count.shape)
        # in_target = (success_hit[..., 1].unsqueeze(-1)) & (self.turn == 2) & ((env_return_ball_count & ball_out_of_court) | ((~env_return_ball_count)&env_reset_count)) # (E, 1)
        in_target = (
            ball_too_low
            & (self.turn == 2)
            & (
                (env_return_ball_count & ball_out_of_court)
                | (env_return_ball_count & ball_hit_net)
                | ((~env_return_ball_count) & env_reset_count)
            )
        )

        # print("in_target",in_target.shape)
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
        # 无人机主动迎球
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
        )  # individual, sparse, (E, 2) 二传向攻手，攻手向目标点击球

        _spike_reward_coeff = 0.2  # 攻手击球后z方向速度大
        reward_spike_velocity = (
            _spike_reward_coeff
            * success_hit[..., 1].unsqueeze(-1)
            * (5 - self.ball_vel[..., 2]).clamp(min=0)
        )  # share, sparse, (E, 1)

        _target_reward_coeff = 0.2
        # dist_to_target_xy = torch.norm(self.ball_pos[..., :2] - self.target, p=2, dim=-1)  # (E, 1)
        reward_dist_to_target = (
            _target_reward_coeff
            * (self.turn == 2)
            * ball_too_low
            * (10 - dist_to_target_xy).clamp(min=0)
        )  # share, sparse, (E, 1)

        # shaping_reward = reward_dist_to_ball + reward_hit_direction + reward_spike_velocity + reward_dist_to_target  # (E, 2)
        shaping_reward = (
            reward_dist_to_ball + reward_hit_direction + reward_spike_velocity
        )  # (E, 2)

        reward = -misbehave_penalty + task_reward + self.reward_shaping * shaping_reward

        # done
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(
            -1
        )  # [E, 1]
        # terminated = ball_misbehave # ball_misbehave | drone_misbehave.any(-1, keepdim=True) | wrong_hit.any(-1, keepdim=True) # [E, 1]
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
        self.stats["ball_too_low_expired"] = ball_too_low_expired.float()
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
        self.stats["in_target_add"].add_(in_target.float())
        self.stats["env_return_ball_count"] = env_return_ball_count.float()
        self.stats["env_reset_count"] = env_reset_count.float()

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
