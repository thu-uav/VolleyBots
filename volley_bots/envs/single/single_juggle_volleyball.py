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


def turn_to_mask(turn: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        turn (torch.Tensor): (*,)

    Returns:
        torch.Tensor: (*,2)
    """
    table = torch.tensor([[True, False], [False, True]], device=turn.device)
    return table[turn]


def turn_to_reward(t: torch.Tensor):
    """convert representation of drone turn

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env,2)  res[:,i]=1.0 if t[:]==i else -1.0
    """
    table = torch.tensor(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        device=t.device,
    )
    return table[t]


def turn_to_obs(t: torch.Tensor):
    """convert representation of drone turn to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env, 2, 2)
    """
    table = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]],
        device=t.device,
    )
    return table[t]


def turn_shift(t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        t (torch.Tensor): (n_env,) int64
        h (torch.Tensor): (n_env,) bool

    Returns:
        torch.Tensor: (n_env,) int64
    """
    return (t + h.long()) % 2


# def calculate_ball_height_reward(
#     r_height: torch.Tensor, factor: float = 1.0
# ) -> torch.Tensor:
#     """_summary_

#     Args:
#         r_height (torch.Tensor): [E,1]
#         factor (float): _description_

#     Returns:
#         torch.Tensor: [E,1]
#     """
#     rwd = factor * torch.sqrt(r_height.clamp(min=0.0, max=5.0))
#     return rwd


def calculate_ball_height_reward(
    ball_pos: torch.Tensor, factor: float = 1.0
) -> torch.Tensor:
    """_summary_

    Args:
        ball_pos (torch.Tensor): [E,1,3]
        factor (float): _description_

    Returns:
        torch.Tensor: [E,1]
    """
    rwd = torch.sqrt((ball_pos[..., 2] - 1.5).clamp(min=0.0, max=5.0))
    return rwd


def calculate_penalty_anchor(
    dist_to_anchor: torch.Tensor,
    turn: torch.Tensor,
    factor_active: float = 1.0,
    factor_inactive: float = 1.0,
):
    """_summary_

    Args:
        dist_to_anchor (torch.Tensor): (E,1) each drone's distance to its anchor
        turn (torch.Tensor): (E,)

    Returns:
        torch.Tensor: (E,1) reward
    """
    mask_before = turn == 0
    mask_after = turn != 1

    p = (dist_to_anchor > 1.0).float() * factor_active * mask_before.float() + (
        dist_to_anchor > 0.5
    ).float() * factor_inactive * mask_after.float()

    return p


def get_hit_reward_mask(
    turn: torch.Tensor, is_first_touch: torch.Tensor
) -> torch.Tensor:
    """_summary_

    Args:
        turn (torch.Tensor): (E,) {0,1}
        is_first_touch (torch.Tensor): (E,) boolean

    Returns:
        torch.Tensor: (E,2)
    """

    last_turn = (turn + 1) % 2  # (E,)

    mask = (turn.unsqueeze(-1) == torch.arange(2, device=turn.device).unsqueeze(0)) | (
        (last_turn.unsqueeze(-1) == torch.arange(2, device=turn.device).unsqueeze(0))
        & torch.logical_not(is_first_touch.unsqueeze(-1))
    )  # (E,2)

    return mask


def calculate_penalty_drone_abs_y(drone_pos: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        drone_pos (torch.Tensor): (E,2,3)

    Returns:
        torch.Tensor: (E,2)
    """
    drone_pos_y = drone_pos[:, :, 1].abs()
    tmp = 1 / (drone_pos_y - 0.1).clamp(min=0.1, max=0.7) - 1 / 0.7

    return tmp * 0.02


def calculate_reward_rpos(turn: torch.Tensor, rpos_ball: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        turn (torch.Tensor): (E,)
        rpos_ball (torch.Tensor): (E,2,3)

    Returns:
        torch.Tensor: (E,2)
    """
    r = turn_to_reward(turn) / (
        1 + torch.norm(rpos_ball[..., :2], p=2, dim=-1)
    )  # (E,2)
    return r


def out_of_bounds(
    pos: torch.Tensor, env_boundary_x: float, env_boundary_y: float
) -> torch.Tensor:
    """_summary_

    Args:
        pos (torch.Tensor): (*,3)
        env_boundary_x (float): _description_
        env_boundary_y (float): _description_

    Returns:
        torch.Tensor: (*,)
    """
    return (pos[..., 0].abs() > env_boundary_x) | (pos[..., 1].abs() > env_boundary_y)


class SingleJuggleVolleyball(IsaacEnv):
    def __init__(self, cfg, headless):
        self.L: float = cfg.task.court.L
        self.W: float = cfg.task.court.W
        self.H_NET: float = cfg.task.court.H_NET  # height of the net
        self.W_NET: float = (
            cfg.task.court.W_NET
        )  # not width of the net, but the width of the net's frame
        self.ball_mass: float = cfg.task.ball_mass
        self.ball_radius: float = cfg.task.ball_radius
        self.min_height: float = cfg.task.min_height
        self.throttles_in_obs = cfg.task.throttles_in_obs
        self.use_ctbr = True if cfg.task.action_transform == "PIDrate" else False

        super().__init__(cfg, headless)

        # x, y, z boundary for drone
        self.env_boundary_x = self.L / 2
        self.env_boundary_y = self.W / 2
        # self.env_boundary_z_top = 2.0
        # self.env_boundary_z_bot = 0.0

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
        # (1,3) original positions of two drones without any offset
        self.anchor = torch.tensor(cfg.task.anchor, device=self.device)
        # self.target = torch.tensor(cfg.task.target, device=self.device)
        # (1,3) drones' initial positions with offsets
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
        # self.turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.last_hit_t = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.int64
        )
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.ball_traj_vis = []
        # one-hot id [E,2,2]
        # self.id = torch.zeros((cfg.task.env.num_envs, 2, 2), device=self.device)
        # self.id[:, 0, 0] = 1
        # self.id[:, 1, 1] = 1

        self.num_true_hits = torch.zeros(self.num_envs, 1, device=self.device)
        self.ball_z = torch.zeros(self.num_envs, 1, device=self.device)
        self.ball_z_max = torch.zeros(self.num_envs, 1, device=self.device)
        self.ball_current_z_max = torch.zeros(self.num_envs, 1, device=self.device)

        self.drone_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.drone_dist_to_anchor = torch.zeros(self.num_envs, 1, device=self.device)
        # self.right_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # self.right_dist_to_anchor = torch.zeros(self.num_envs, 1, device=self.device)

        # self.num_left_hits = torch.zeros(self.num_envs, 1, device=self.device)
        self.hit_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.hit_dist_to_anchor = torch.zeros(self.num_envs, 1, device=self.device)
        # self.num_right_hits = torch.zeros(self.num_envs, 1, device=self.device)
        # self.right_hit_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # self.right_hit_dist_to_anchor = torch.zeros(self.num_envs, 1, device=self.device)

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
        drone_prims = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = (
            drone_state_dim + 3 + 3 + 3
        )  # specified in function _compute_state_and_obs

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
                                (1, observation_dim)  # 2 drones
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
                            "action": torch.stack([self.drone.action_spec] * 1, dim=0),
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
                    "done": DiscreteTensorSpec(1, (1,), dtype=torch.bool),
                    "terminated": DiscreteTensorSpec(1, (1,), dtype=torch.bool),
                    "truncated": DiscreteTensorSpec(1, (1,), dtype=torch.bool),
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
        )

        _stats_spec = CompositeSpec(
            {
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "num_turns": UnboundedContinuousTensorSpec(1),
                "drone_x": UnboundedContinuousTensorSpec(1),
                "drone_y": UnboundedContinuousTensorSpec(1),
                "drone_z": UnboundedContinuousTensorSpec(1),
                "drone_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                # "right_x": UnboundedContinuousTensorSpec(1),
                # "right_y": UnboundedContinuousTensorSpec(1),
                # "right_z": UnboundedContinuousTensorSpec(1),
                # "right_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                "ball_z": UnboundedContinuousTensorSpec(1),
                "ball_z_max": UnboundedContinuousTensorSpec(1),
                # "max_episode_len_rate": UnboundedContinuousTensorSpec(1),
                # "wrong_hit_rate": UnboundedContinuousTensorSpec(1),
                "abs_x": UnboundedContinuousTensorSpec(2),
                "done": UnboundedContinuousTensorSpec(1),
                # "wrong_hit_1": UnboundedContinuousTensorSpec(1),
                # "wrong_hit_2": UnboundedContinuousTensorSpec(1),
                "wrong_hit": UnboundedContinuousTensorSpec(1),
                "misbehave": UnboundedContinuousTensorSpec(1),
                "truncated": UnboundedContinuousTensorSpec(1),
                "num_hits": UnboundedContinuousTensorSpec(1),
                "num_true_hits": UnboundedContinuousTensorSpec(1),
                # "num_hits_is_close": UnboundedContinuousTensorSpec(1),
                "hit_x": UnboundedContinuousTensorSpec(1),
                "hit_y": UnboundedContinuousTensorSpec(1),
                "hit_z": UnboundedContinuousTensorSpec(1),
                "hit_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                # "right_hit_x": UnboundedContinuousTensorSpec(1),
                # "right_hit_y": UnboundedContinuousTensorSpec(1),
                # "right_hit_z": UnboundedContinuousTensorSpec(1),
                # "right_hit_dist_to_anchor": UnboundedContinuousTensorSpec(1),
            }
        )
        self.stats_cfg: DictConfig = self.cfg.task.stats
        if self.stats_cfg.get("angular_deviation", False):
            # 0: cummulative deviation 1: count(score)
            _stats_spec.set("angular_deviation", UnboundedContinuousTensorSpec(2))

        if self.stats_cfg.get("done_drone_misbehave", False):
            _stats_spec.set("done_drone_misbehave", UnboundedContinuousTensorSpec(1))

        if self.stats_cfg.get("done_ball_misbehave", False):
            _stats_spec.set("done_ball_misbehave", UnboundedContinuousTensorSpec(1))

        if self.stats_cfg.get("done_drone_out_of_boundary", False):
            _stats_spec.set(
                "done_drone_out_of_boundary", UnboundedContinuousTensorSpec(1)
            )

        if self.stats_cfg.get("done_drone_too_low", False):
            _stats_spec.set("done_drone_too_low", UnboundedContinuousTensorSpec(1))

        if self.stats_cfg.get("done_drone_too_close_to_net", False):
            _stats_spec.set(
                "done_drone_too_close_to_net", UnboundedContinuousTensorSpec(1)
            )

        if self.stats_cfg.get("done_ball_too_low", False):
            _stats_spec.set("done_ball_too_low", UnboundedContinuousTensorSpec(1))

        if self.stats_cfg.get("done_ball_too_high", False):
            _stats_spec.set("done_ball_too_high", UnboundedContinuousTensorSpec(1))

        if self.stats_cfg.get("done_ball_out_of_boundary", False):
            _stats_spec.set(
                "done_ball_out_of_boundary", UnboundedContinuousTensorSpec(1)
            )

        if self.stats_cfg.get("reward", False):
            # _stats_spec.set("reward_rpos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_anchor", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_ball_height", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_success_hit", UnboundedContinuousTensorSpec(1))
            # _stats_spec.set("reward_score", UnboundedContinuousTensorSpec(1))
            # _stats_spec.set("angular_penalty", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_wrong_hit", UnboundedContinuousTensorSpec(1))
            # _stats_spec.set("penalty_drone_abs_y", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_drone_misbehave", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_ball_misbehave", UnboundedContinuousTensorSpec(1))
            # _stats_spec.set("penalty_expected_ball_h", UnboundedContinuousTensorSpec(1))

        stats_spec = _stats_spec.expand(self.num_envs).to(self.device)

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

    # def debug_draw_turn(self):
    #     turn = self.turn[self.central_env_idx]
    #     ori = self.envs_positions[self.central_env_idx].detach()
    #     points = self.anchor.clone() + ori
    #     points[:, -1] = 0
    #     points = points.tolist()
    #     colors = [(0, 1, 0, 1), (1, 0, 0, 1)]
    #     sizes = [10., 10.]
    #     if turn.item() == 1:
    #         colors = colors[::-1]
    #     # self.draw.clear_points()
    #     self.draw.draw_points(points, colors, sizes)

    def _reset_idx(self, env_ids: torch.Tensor):
        # drone
        self.drone._reset_idx(env_ids, self.training)
        drone_pos = self.init_drone_pos_dist.sample((*env_ids.shape, 1))
        drone_rpy = self.init_drone_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(
            torch.zeros(len(env_ids), 1, 6, device=self.device), env_ids
        )

        # ball and turn
        # turn = torch.randint(0, 2, (len(env_ids),), device=self.device) # random initial turn
        # turn = torch.zeros(len(env_ids), device=self.device, dtype=torch.int64)
        # self.turn[env_ids] = turn

        self.ball_z[env_ids] = torch.zeros(len(env_ids), 1, device=self.device)
        self.ball_z_max[env_ids] = torch.zeros(len(env_ids), 1, device=self.device)
        self.ball_current_z_max[env_ids] = torch.zeros(
            len(env_ids), 1, device=self.device
        )

        self.num_true_hits[env_ids] = torch.zeros(len(env_ids), 1, device=self.device)
        self.drone_pos[env_ids] = torch.zeros(len(env_ids), 3, device=self.device)
        self.drone_dist_to_anchor[env_ids] = torch.zeros(
            len(env_ids), 1, device=self.device
        )
        # self.right_pos[env_ids] = torch.zeros(len(env_ids), 3, device=self.device)
        # self.right_dist_to_anchor[env_ids] = torch.zeros(len(env_ids), 1, device=self.device)

        # self.num_left_hits[env_ids] = torch.zeros(len(env_ids), 1, device=self.device)
        self.hit_pos[env_ids] = torch.zeros(len(env_ids), 3, device=self.device)
        self.hit_dist_to_anchor[env_ids] = torch.zeros(
            len(env_ids), 1, device=self.device
        )
        # self.num_right_hits[env_ids] = torch.zeros(len(env_ids), 1, device=self.device)
        # self.right_hit_pos[env_ids] = torch.zeros(len(env_ids), 3, device=self.device)
        # self.right_hit_dist_to_anchor[env_ids] = torch.zeros(len(env_ids), 1, device=self.device)
        # ball_pos = self.init_ball_pos_dist.sample((*env_ids.shape, 1))
        ball_pos = (
            drone_pos[torch.arange(len(env_ids)), 0, :] + self.init_ball_offset
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
        self.last_hit_t[env_ids] = -100
        self.stats[env_ids] = 0.0

        # draw
        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            self.ball_traj_vis.clear()
            self.draw.clear_lines()
            # self.debug_draw_region()
            # self.debug_draw_turn()

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
        actions = tensordict[
            ("agents", "action")
        ].clone()  # For rotor command actions, not the actions output by the policy.
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

        if self.cfg.task.get("tanh_action", False):
            actions = torch.tanh(actions)

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

        # self.rpos_drone = torch.stack(
        #     [
        #         # [..., drone_id, [x, y, z]]
        #         self.drone.pos[..., 1, :] - self.drone.pos[..., 0, :],
        #         self.drone.pos[..., 0, :] - self.drone.pos[..., 1, :],
        #     ],
        #     dim=1,
        # )  # (E,2,3)

        rpos_anchor = self.drone.pos - self.anchor  # (E,2,3)

        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )
        if self.throttles_in_obs:
            obs = [
                self.root_state,  # (E,1,23)
                rpos_anchor,  # (E,1,3)
                # self.rpos_drone[..., :3],  # (E,2,3)
                self.rpos_ball,  # (E,1,3)
                self.ball_vel,  # (E,1,3)
                # turn_to_obs(self.turn), # (E,2,2)
                # self.id, # (E,2,2)
            ]
        else:
            obs = [
                pos,
                rot,
                vel,
                angular_vel,
                heading,
                up,
                rpos_anchor,  # (E,1,3)
                # self.rpos_drone[..., :3],  # (E,2,3)
                self.rpos_ball,  # (E,1,3)
                self.ball_vel,  # (E,1,3)
                # turn_to_obs(self.turn), # (E,2,2)
                # self.id, # (E,2,2)
            ]
        # obs = [
        #     self.root_state, # (E,1,23)
        #     rpos_anchor,  # (E,1,3)
        #     # self.rpos_drone[..., :3],  # (E,2,3)
        #     self.rpos_ball,  # (E,1,3)
        #     self.ball_vel,  # (E,1,3)
        #     # turn_to_obs(self.turn), # (E,2,2)
        #     # self.id, # (E,2,2)
        # ]

        # obs = [
        #     self.root_state, # (E,2,23)
        #     rpos_anchor,  # (E,2,3)
        #     self.rpos_drone[..., :3],  # (E,2,3)
        #     self.rpos_ball,  # (E,2,3)
        #     self.ball_vel.expand(-1, 2, 3),  # (E,2,3)
        #     turn_to_obs(self.turn), # (E,2,2)
        #     self.id, # (E,2,2)
        # ]
        # [drone_num(2),
        # each_obs_dim: root_state(rpos_anchor)+rpos_drone(3)+rpos_ball(3)+ball_vel(3)+turn(1)]

        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs.append(t.expand(-1, 1, self.time_encoding_dim))

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
        drone_out_of_boundary = out_of_bounds(
            self.drone.pos, self.env_boundary_x, self.env_boundary_y
        )  # (E,1)
        drone_too_low = self.drone.pos[..., 2] < 0.4
        drone_too_high = self.drone.pos[..., 2] > 2.5
        drone_too_close_to_net = self.drone.pos[..., 0].abs() < 0.01  # (E,1)
        drone_misbehave = (
            drone_too_low | drone_too_high | drone_too_close_to_net
        )  # (E,1)

        if self.stats_cfg.get("done_drone_too_close_to_net", False):
            self.stats["done_drone_too_close_to_net"] = drone_too_close_to_net.any(
                -1, keepdim=True
            ).float()
        if self.stats_cfg.get("done_drone_out_of_boundary", False):
            self.stats["done_drone_out_of_boundary"] = drone_out_of_boundary.any(
                -1, keepdim=True
            ).float()
        if self.stats_cfg.get("done_drone_too_low", False):
            self.stats["done_drone_too_low"] = drone_too_low.any(
                -1, keepdim=True
            ).float()
        if self.stats_cfg.get("done_drone_misbehave", False):
            self.stats["done_drone_misbehave"] = drone_misbehave.any(
                -1, keepdim=True
            ).float()

        ball_too_low = self.ball_pos[..., 2] < 0.15  # (E,1)
        ball_too_high = self.ball_pos[..., 2] > 16  # (E,1)
        ball_out_of_boundary = out_of_bounds(
            self.ball_pos, self.env_boundary_x, self.env_boundary_y
        )  # (E,1)
        ball_misbehave = ball_too_low | ball_too_high | ball_out_of_boundary  # (E,1)

        if self.stats_cfg.get("done_ball_too_low", False):
            self.stats["done_ball_too_low"] = ball_too_low.float()
        if self.stats_cfg.get("done_ball_too_high", False):
            self.stats["done_ball_too_high"] = ball_too_high.float()
        if self.stats_cfg.get("done_ball_out_of_boundary", False):
            self.stats["done_ball_out_of_boundary"] = ball_out_of_boundary.float()
        if self.stats_cfg.get("done_ball_misbehave", False):
            self.stats["done_ball_misbehave"] = ball_misbehave.float()

        self.ball_z += self.ball_pos[..., 2]
        self.drone_pos += self.drone.pos[:, 0]
        self.drone_dist_to_anchor += torch.norm(
            self.drone.pos[:, 0] - self.anchor, p=2, dim=-1, keepdim=True
        )
        # self.right_pos += self.drone.pos[:, 1]
        # self.right_dist_to_anchor += torch.norm(self.drone.pos[:, 1] - self.anchor[1], p=2, dim=-1, keepdim=True)

        # ball_contact_forces = self.ball.get_net_contact_forces() # (E,1,3)
        ball_contact_forces = self.contact_sensor.data.net_forces_w  # (E,1,3)

        # which_drone: torch.Tensor = self.rpos_ball.norm(p=2, dim=-1).argmin(dim=1, keepdim=True)  # (E,1) which drone is closer to the ball

        # is_close = self.rpos_ball[torch.arange(self.num_envs, device=which_drone.device), which_drone.squeeze(-1)].norm(dim=-1, keepdim=True) < 0.3 # (E,1)

        hit = ball_contact_forces.any(-1)  # (E,1)

        # hit_is_close = hit & is_close  # (E,1) 击球且距离球较近

        true_hit_step_gap = 25  # 25 * 0.016s = 0.4s
        true_hit = hit & (
            (self.progress_buf.unsqueeze(-1) - self.last_hit_t) > true_hit_step_gap
        )  # (E,1) 击球时间大于30个step才是正确的一次击球
        wrong_hit_step_gap = 25  # 25 * 0.016s = 0.4s
        wrong_hit = hit & (
            (self.progress_buf.unsqueeze(-1) - self.last_hit_t) <= wrong_hit_step_gap
        )  # (E,1) 击球时间小于30个step则为错误击球

        self.num_true_hits += true_hit
        self.ball_current_z_max = torch.max(
            self.ball_current_z_max, self.ball_pos[..., 2]
        )
        self.ball_z_max += (self.num_true_hits > 1) * true_hit * self.ball_current_z_max
        self.ball_current_z_max = torch.logical_not(true_hit) * self.ball_current_z_max

        # left_hit = true_hit & (self.turn.unsqueeze(-1) == 0)
        self.hit_pos += true_hit * self.drone.pos[:, 0]
        self.hit_dist_to_anchor += true_hit * torch.norm(
            self.drone.pos[:, 0] - self.anchor, p=2, dim=-1, keepdim=True
        )

        # right_hit = true_hit & (self.turn.unsqueeze(-1) == 1)
        # self.num_right_hits += right_hit
        # self.right_hit_pos += right_hit * self.drone.pos[:, 1]
        # self.right_hit_dist_to_anchor += right_hit * torch.norm(self.drone.pos[:, 1] - self.anchor[1], p=2, dim=-1, keepdim=True)

        # # TODO: simplify
        # new_drone_last_hit_t = torch.where(hit, self.progress_buf.unsqueeze(-1), drone_last_hit_t)  # (E,1) 更新drone_last_hit_t
        # new_last_hit_t = new_drone_last_hit_t * (which_drone == torch.arange(2, device=which_drone.device).unsqueeze(0)).long() # (E,2)
        # new_last_hit_t += self.last_hit_t * (which_drone != torch.arange(2, device=which_drone.device).unsqueeze(0)).long() # (E,2)
        # self.last_hit_t = new_last_hit_t # (E,2) 两个无人机各自的上次击球时刻
        if true_hit.any():
            self.last_hit_t[true_hit] = self.progress_buf.unsqueeze(-1)[true_hit].long()

        # switch_turn: torch.Tensor = true_hit & (self.turn.unsqueeze(-1) == which_drone)  # (E,1) 在该无人机击球回合，该无人机击球成功，则轮到另一个无人机击球
        # wrong_hit_2: torch.Tensor = true_hit & (self.turn.unsqueeze(-1) != which_drone)  # (E,1) 在非该无人机击球回合，该无人机击球成功，则为错误击球

        # wrong_hit = wrong_hit_1 | wrong_hit_2  # (E,1) 错误击球

        # if self._should_render(0) and switch_turn.any():
        #     self.debug_draw_turn()

        # self.stats["wrong_hit_rate"] = (self.stats["wrong_hit_rate"].bool() | wrong_hit).float() # (E,1)
        # old_turn = self.turn.clone() # (E,)

        # switch turn
        # if switch_turn.any():
        #     self.turn = turn_shift(self.turn, switch_turn.squeeze(-1))

        # reward_rpos = 0.03 * calculate_reward_rpos(self.turn, self.rpos_ball)  # (E,2) 有正有负：该回合无人机为正（离得远近值越大），非该回合无人机为负（离得越近值越小）

        # not_turn_with_success = turn_to_mask((self.turn + 1) % 2) & (self.stats["num_turns"] > 0) # (E,2)
        # reward_ball_height = 0.01 * calculate_ball_height_reward(self.ball_pos) * not_turn_with_success.float() # (E,2) 如果有成功打过去过，则：该回合无人机奖励为0，非该回合无人机为正，让打过去的球高度越高越好（有最大值限制）
        # reward_ball_height = 0.01 * calculate_ball_height_reward(self.ball_pos[..., 2] - self.anchor[:, 0, 2]) * not_turn_with_success.float() # (E,2) 如果有成功打过去过，则：该回合无人机奖励为0，非该回合无人机为正，让打过去的球高度越高越好（有最大值限制）

        dist_to_anchor: torch.Tensor = torch.norm(
            self.drone.pos - self.anchor, p=2, dim=-1
        )  # (E,1)
        # penalty_anchor = (calculate_penalty_anchor(dist_to_anchor) * 0.03)  # (E,1) 击球前无人机离其anchor距离大于1有惩罚，击球后无人机离其anchor距离大于0.5有惩罚

        penalty_anchor = (dist_to_anchor > 1.0).float() * 0.4
        # penalty_anchor = dist_to_anchor * 0.05

        # penalty_drone_abs_y = calculate_penalty_drone_abs_y(self.drone.pos)  # (E,2)

        # _reward_score_factor = 2.0
        # reward_score = switch_turn.float() * _reward_score_factor  # (E,1) # reward for a successful hit

        # hit_reward_mask = get_hit_reward_mask(old_turn, self.stats["num_turns"].squeeze(-1) == 0)  # (E,2)
        # reward_score = hit_reward_mask * reward_score # (E,2)

        # rpos_xy = self.rpos_drone[torch.arange(self.num_envs), old_turn, :2]  # (E,2)
        # ball_vel_xy = self.ball_vel[:, 0, :2]  # (E,2)
        # cosine_similarity = NNF.cosine_similarity(ball_vel_xy, rpos_xy, dim=-1)  # (E,)
        # angular_penalty = switch_turn.squeeze(-1).float() * (1 - cosine_similarity).clamp(min=0.04, max=1.0) * _reward_score_factor  # (E,)
        # angular_penalty = turn_to_mask(old_turn).float() * angular_penalty.unsqueeze(-1)  # only apply to the drone that's taking its turn  # (E,2) 击球之后瞬间，球速度和无人机相对位置方向的惩罚

        # 思路是，在xz平面上，球应该差不多打到锚点附近
        # t = (self.ball_pos[:, 0, 0].abs() + 1) / self.ball_vel[:, 0, 0].abs()  # (E,)
        # h = (
        #     self.ball_pos[:, 0, 2]
        #     + self.ball_vel[:, 0, 2] * t
        #     - 0.5 * 9.81 * torch.square(t)
        # )  # (E,)
        # penalty_expected_ball_h = ((h < 1.0) | (h > 2.5)).float().unsqueeze(
        #     -1
        # ) * 0.5  # (E,1)
        # penalty_expected_ball_h = (
        #     switch_turn.float()
        #     * penalty_expected_ball_h
        #     * turn_to_mask(old_turn).float()
        # )  # (E,2)

        # only penalize the drone who wrongly hit the ball
        penalty_wrong_hit = wrong_hit.float() * 10.0  # (E,1)
        # penalty_wrong_hit = turn_to_mask(which_drone.squeeze(-1)).float() * penalty_wrong_hit  # (E,2)
        penalty_drone_misbehave = drone_misbehave * 10.0  # (E,1)
        penalty_ball_misbehave = ball_misbehave * 10.0  # (E,1)

        # step_reward = reward_rpos + reward_ball_height - penalty_drone_abs_y - penalty_anchor
        # step_reward = reward_rpos + reward_ball_height - penalty_anchor
        above_min_height_reward = (
            self.ball_z_max / (self.num_true_hits - 1).clamp(min=1e-5)
        ) > self.min_height
        reward_ball_height = 3 * (true_hit & above_min_height_reward).float()  # (E, 1)
        # print("reward_ball_height", reward_ball_height)
        reward_success_hit = true_hit.any(-1, keepdim=True).float()  # (E, 1)
        # print("reward_success_hit", reward_success_hit)
        step_reward = reward_ball_height + reward_success_hit - penalty_anchor
        # conditional_reward = reward_score - angular_penalty
        end_penalty = (
            -penalty_wrong_hit - penalty_drone_misbehave - penalty_ball_misbehave
        )

        # reward: torch.Tensor = step_reward + conditional_reward + end_penalty  # (E,2)
        reward: torch.Tensor = step_reward + end_penalty  # (E,2)

        if self.stats_cfg.get("reward", False):
            # self.stats["reward_rpos"].add_(reward_rpos.abs().mean(dim=-1, keepdim=True))
            self.stats["penalty_anchor"].add_(penalty_anchor.mean(dim=-1, keepdim=True))
            self.stats["reward_ball_height"].add_(
                reward_ball_height.mean(dim=-1, keepdim=True)
            )
            self.stats["reward_success_hit"].add_(
                reward_success_hit.mean(dim=-1, keepdim=True)
            )
            # self.stats["reward_score"].add_(reward_score.mean(dim=-1, keepdim=True))
            # self.stats["angular_penalty"].add_(angular_penalty.mean(dim=-1, keepdim=True))
            self.stats["penalty_wrong_hit"].add_(
                penalty_wrong_hit.mean(dim=-1, keepdim=True)
            )
            # self.stats["penalty_drone_abs_y"].add_(penalty_drone_abs_y.mean(dim=-1, keepdim=True))
            self.stats["penalty_drone_misbehave"].add_(
                penalty_drone_misbehave.mean(dim=-1, keepdim=True)
            )
            self.stats["penalty_ball_misbehave"].add_(
                penalty_ball_misbehave.mean(dim=-1, keepdim=True)
            )

        misbehave = drone_misbehave.any(-1, keepdim=True) | ball_misbehave  # [E, 1]
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(
            -1
        )  # [E, 1]
        terminated = misbehave | wrong_hit  # [E, 1]
        done: torch.Tensor = truncated | terminated  # [E, 1]

        self.stats["return"].add_(reward.mean(dim=-1, keepdim=True))
        self.stats["episode_len"] = self.progress_buf.unsqueeze(1)
        # self.stats["num_turns"].add_(switch_turn.float())

        self.stats["ball_z"] = self.ball_z / self.stats["episode_len"]
        self.stats["ball_z_max"] = self.ball_z_max / (self.num_true_hits - 1).clamp(
            min=1e-5
        )

        self.stats["drone_x"] = (
            self.drone_pos[:, 0].unsqueeze(-1) / self.stats["episode_len"]
        )
        self.stats["drone_y"] = (
            self.drone_pos[:, 1].unsqueeze(-1) / self.stats["episode_len"]
        )
        self.stats["drone_z"] = (
            self.drone_pos[:, 2].unsqueeze(-1) / self.stats["episode_len"]
        )
        self.stats["drone_dist_to_anchor"] = (
            self.drone_dist_to_anchor / self.stats["episode_len"]
        )

        # self.stats["right_x"] = self.right_pos[:, 0].unsqueeze(-1) / self.stats["episode_len"]
        # self.stats["right_y"] = self.right_pos[:, 1].unsqueeze(-1) / self.stats["episode_len"]
        # self.stats["right_z"] = self.right_pos[:, 2].unsqueeze(-1) / self.stats["episode_len"]
        # self.stats["right_dist_to_anchor"] = self.right_dist_to_anchor / self.stats["episode_len"]

        self.stats["done"].add_(done.float())
        # self.stats["wrong_hit_1"].add_(wrong_hit_1.float())
        # self.stats["wrong_hit_2"].add_(wrong_hit_2.float())
        self.stats["wrong_hit"].add_(wrong_hit.float())
        self.stats["misbehave"].add_(misbehave.float())
        self.stats["truncated"].add_(truncated.float())

        self.stats["num_hits"].add_(hit.float())
        self.stats["num_true_hits"].add_(true_hit.float())

        self.stats["hit_x"] = self.hit_pos[:, 0].unsqueeze(
            -1
        ) / self.num_true_hits.clamp(min=1e-5)
        self.stats["hit_y"] = self.hit_pos[:, 1].unsqueeze(
            -1
        ) / self.num_true_hits.clamp(min=1e-5)
        self.stats["hit_z"] = self.hit_pos[:, 2].unsqueeze(
            -1
        ) / self.num_true_hits.clamp(min=1e-5)
        self.stats["hit_dist_to_anchor"] = (
            self.hit_dist_to_anchor / self.num_true_hits.clamp(min=1e-5)
        )

        # self.stats["right_hit_x"] = self.right_hit_pos[:, 0].unsqueeze(-1) / (self.num_right_hits + 1e-5)
        # self.stats["right_hit_y"] = self.right_hit_pos[:, 1].unsqueeze(-1) / (self.num_right_hits + 1e-5)
        # self.stats["right_hit_z"] = self.right_hit_pos[:, 2].unsqueeze(-1) / (self.num_right_hits + 1e-5)
        # self.stats["right_hit_dist_to_anchor"] = self.right_hit_dist_to_anchor / (self.num_right_hits + 1e-5)

        # if self.stats_cfg.get("angular_deviation", False):
        #     angular_deviation = torch.acos(cosine_similarity)  # (E,)
        #     self.stats["angular_deviation"][switch_turn.squeeze(-1), 0] += angular_deviation[switch_turn.squeeze(-1)] / torch.pi * 180
        #     self.stats["angular_deviation"][switch_turn.squeeze(-1), 1] += 1

        self.stats["abs_x"][:, 0].add_(self.drone.pos[:, :, 0].abs().mean(dim=-1))
        self.stats["abs_x"][:, 1].add_(1.0)

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.num_envs,
        )
