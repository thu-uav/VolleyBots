import logging
import os
from typing import Dict, List

import omni.isaac.core.materials as materials
import omni.isaac.core.objects as objects
import torch
import torch.nn.functional as F
import torch.distributions as D
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

import volley_bots.utils.kit as kit_utils
from volley_bots.envs.isaac_env import AgentSpec, IsaacEnv
from volley_bots.envs.volleyball.common import (
    _carb_float3_add,
    rectangular_cuboid_edges,
)
from volley_bots.envs.volleyball.draw import draw_court
from volley_bots.robots.drone import MultirotorBase
from volley_bots.utils.torch import euler_to_quaternion, quat_axis
from volley_bots.views import RigidPrimView

from .rules import determine_game_result_6v6, game_result_to_matrix


def _carb_float3_add(a: Float3, b: Float3) -> Float3:
    return Float3(a.x + b.x, a.y + b.y, a.z + b.z)


def _vec3_format(vec3: List[float]):
    return f"({vec3[0]:.2e},{vec3[1]:.2e},{vec3[2]:.2e})"


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


def turn_to_obs(t: torch.Tensor) -> torch.Tensor:
    """convert representation of drone turn to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env, 12, 2)
    """
    table = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
        ],
        device=t.device,
    )  # (2, 6, 2)

    return table[t]


# def turn_to_mask(turn: torch.Tensor, n: int = 2) -> torch.Tensor:
    # """_summary_

    # Args:
    #     turn (torch.Tensor): (*,)

    # Returns:
    #     torch.Tensor: (*,n)
    # """
    # assert n in [2, 6]
    # if n == 2:
    #     table = torch.tensor([[True, False], [False, True]], device=turn.device)
    # elif n == 6:
    #     table = torch.tensor(
    #         [
    #             [True, True, True, False, False, False],
    #             [False, False, False, True, True, True],
    #         ],
    #         device=turn.device,
    #     )
    # return table[turn]


def calculate_penalty_drone_too_near_boundary(
    drone_pos: torch.Tensor, L: float, W: float
) -> torch.Tensor:
    dist_x = L / 2 / 5
    penalty_x = ((L / 2 - drone_pos[:, :, 0].abs()) < dist_x) * 0.5  # (E,4)

    dist_y = W / 2 / 5
    penalty_y = (drone_pos[:, :, 1].abs() < dist_y).float() * 0.5 + (
        (W / 2 - drone_pos[:, :, 1].abs()) < dist_y
    ) * 0.5  # (E,4)

    return penalty_x + penalty_y  # (E,2)


def calculate_penalty_drone_pos(
    drone_pos: torch.Tensor,
    L: float,
    W: float,
    anchor,
    dist_x=0.5,
    dist_y=0.5,
    dist_z=1,
) -> torch.Tensor:
    """_summary_

    Args:
        drone_pos (torch.Tensor): (E,4,3)
        L (float): _description_
        W (float): _description_

    Returns:
        torch.Tensor: _description_
    """


    penalty_x = (drone_pos[:, :, 0].abs() < dist_x).float() * (
        dist_x - drone_pos[:, :, 0].abs()
    ) + ((L / 2 - drone_pos[:, :, 0].abs()) < dist_x).float() * (
        dist_x - (L / 2 - drone_pos[:, :, 0].abs())
    )  # (E,2)


    penalty_y = (drone_pos[:, :, 1].abs() < dist_y).float() * (
        dist_y - drone_pos[:, :, 1].abs()
    ) + ((W / 2 - drone_pos[:, :, 1].abs()) < dist_y).float() * (
        dist_y - (W / 2 - drone_pos[:, :, 1].abs())
    )  # (E,2)


    penalty_z = ((drone_pos[:, :, 2] - anchor[:, 2]).abs() > dist_z) * (
        (drone_pos[:, :, 2] - anchor[:, 2]).abs() - dist_z
    )  # (E,2)

    penalty = penalty_x + penalty_y + penalty_z
    return penalty


def true_hit_to_reward(true_hit: torch.Tensor) -> torch.Tensor:
    """
    compute shared reward for team members

    Args:
        true_hit: [E, 12]

    Returns:
        reward: [E, 12]
    """
    assert len(true_hit.shape) == 2
    assert true_hit.shape[1] == 12

    reward = torch.zeros_like(true_hit)

    mask_0_to_5 = true_hit[:, :6].sum(dim=1) > 0
    reward[mask_0_to_5, :6] = 1.0

    mask_6_to_11 = true_hit[:, 6:].sum(dim=1) > 0
    reward[mask_6_to_11, 6:] = 1.0

    return reward


def transfer_root_state_to_the_other_side(root_state):
    """Transfer the root state to the other side of the court

    Args:
        root_state: [E, 1, 23]

    Returns:
        root_state: [E, 1, 23]
    """

    assert len(root_state.shape) == 3
    assert root_state.shape[1] == 1
    assert root_state.shape[2] == 23

    pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
        root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
    )

    pos[..., :2] = -pos[..., :2]

    q_m = torch.tensor([[0, 0, 0, 1]] * rot.shape[0], device=rot.device)
    rot[:, 0, :] = quaternion_multiply(q_m, rot[:, 0, :])
    rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot)

    vel[..., :2] = -vel[..., :2]

    angular_vel[..., :2] = -angular_vel[..., :2]

    heading = quat_axis(rot, axis=0)

    up = quat_axis(rot, axis=2)

    return torch.cat([pos, rot, vel, angular_vel, heading, up, throttle], dim=-1)


def quaternion_multiply(q1, q2):
    assert q1.shape == q2.shape and q1.shape[-1] == 4
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=quaternion.dim() - 1)

    euler_angles: torch.Tensor = torch.stack(
        (
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin(2.0 * (w * y - z * x)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dim=-1,
    )

    return euler_angles


class Volleyball6v6(IsaacEnv):
    """
    Two teams of drones hit the ball adversarially.
    The net is positioned parallel to the y-axis.

    """

    def __init__(self, cfg: DictConfig, headless: bool):

        self.L: float = cfg.task.court.L
        self.W: float = cfg.task.court.W
        self.H_NET: float = cfg.task.court.H_NET  # height of the net
        self.W_NET: float = (
            cfg.task.court.W_NET
        )  # not width of the net, but the width of the net's frame
        self.ball_mass: float = cfg.task.get("ball_mass", 0.05)
        self.ball_radius: float = cfg.task.get("ball_radius", 0.1)
        self.symmetric_obs: bool = cfg.task.get("symmetric_obs", True)
        assert self.symmetric_obs, "Only symmetric observation is supported now."

        super().__init__(cfg, headless)

        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )
        print(f"Central env position:{self.central_env_pos}")
        self.drone.initialize()
        self.ball = RigidPrimView(
            "/World/envs/env_*/ball",
            reset_xform_properties=False,
            track_contact_forces=False,
            shape=(-1, 1),
        )
        self.ball.initialize()
        self.ball_initial_z_vel = cfg.task.initial.get("ball_initial_z_vel", 0.0)
        self.racket_radius = cfg.task.get("racket_radius", 0.2)

        self.near_side_anchor = torch.tensor(
            cfg.task.anchor, device=self.device
        )  # [6, 3]
        self.far_side_anchor = self.near_side_anchor.clone()
        self.far_side_anchor[:, :2] = -self.far_side_anchor[:, :2] # [6, 3]
        self.anchor = torch.cat(
            [self.near_side_anchor, self.far_side_anchor], dim=0
        ).expand(
            self.num_envs, 12, 3
        )  # [E, 12, 3]

        self.near_side_pos = torch.tensor(
            cfg.task.initial.init_pos, device=self.device
        )  # [6, 3]
        self.far_side_pos = self.near_side_pos.clone()
        self.far_side_pos[:, :2] = -self.far_side_pos[:, :2]  # [6, 3]
        self.init_pos = torch.cat(
            [self.near_side_pos, self.far_side_pos], dim=0
        ).expand(self.num_envs, 12, 3)  # [E, 12, 3]

        self.init_ball_offset = torch.tensor(
            cfg.task.initial.ball_offset, device=self.device
        ) # [3]
        self.init_on_the_spot = cfg.task.initial.get("init_on_the_spot", True)
        assert self.init_on_the_spot, "Only init_on_the_spot is supported now."

        self.which_side = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        ) # [E] 0: near side, 1: far side

        self.last_hit_team = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        ) # [E] 0: near side, 1: far side

        self.last_hit_step = torch.zeros(
            self.num_envs, 12, device=self.device, dtype=torch.int64
        ) # [E, 12]

        self.ball_last_vel = torch.zeros(
            self.num_envs, 1, 3, device=self.device, dtype=torch.float32
        ) # [E, 1, 3]

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.random_turn = cfg.task.get("random_turn", True)

        # one-hot id [E,6,6]
        self.id = F.one_hot(torch.arange(6), num_classes=6).to(self.device).unsqueeze(0).expand(self.num_envs, 6, 6)

        self.racket_near_ball = torch.zeros(
            (cfg.task.env.num_envs, 12), device=self.device, dtype=torch.bool
        )
        self.drone_near_ball = torch.zeros(
            (cfg.task.env.num_envs, 12), device=self.device, dtype=torch.bool
        )

        self.init_rpy = torch.zeros((cfg.task.env.num_envs, 12, 3), device=self.device)
        self.init_rpy[:, 6:, 2] += 3.1415926

        self.drones_already_hit_in_one_turn = torch.zeros(
            (cfg.task.env.num_envs, 12), device=self.device, dtype=torch.bool
        )

        self.not_reset_keys_in_stats = [
            "actor_0_wins",
            "actor_1_wins",
            "draws",
            "terminated",
            "truncated",
            "done",
        ]



        self.ball_vel_x_dir_world = torch.ones((self.num_envs, 12), device=self.device)
        self.ball_vel_x_dir_world[:, :6] = -1

        # coefs
        self.game_reward_coef = 100.0
        self.hit_reward_coef = 10.0
        self.ball_vel_x_reward_coef = 0.0
        self.rpos_reward_coef = 0.5
        self.rpos_anchor_reward_coef = 0.05
        self.rpy_penalty_coef = 0.0
        self.pos_z_penalty_coef = 0.0
        self.drone_hit_ground_penalty_coef = 100.0
        self.drone_too_close_penalty_coef = 100.0

        logging.info(
            f"""All reward coefs:
                     game_reward_coef: {self.game_reward_coef}, 
                     hit_reward_coef: {self.hit_reward_coef}, 
                     ball_vel_x_reward_coef: {self.ball_vel_x_reward_coef},
                     rpos_reward_coef: {self.rpos_reward_coef}, 
                     rpos_anchor_reward_coef: {self.rpos_anchor_reward_coef},
                     rpy_penalty_coef: {self.rpy_penalty_coef},
                     pos_z_penalty_coef: {self.pos_z_penalty_coef},
                     drone_hit_ground_penalty_coef: {self.drone_hit_ground_penalty_coef},
                     drone_too_close_penalty_coef: {self.drone_too_close_penalty_coef}"""
        )

        # settings
        self.drones_pos_or_rpos_obs = True  # True: pos, False: rpos
        self.shared_or_individual_rpos_reward = False  # True: shared, False: individual

        logging.info(
            f"""Settings:
                        drones_pos_or_rpos_obs: {self.drones_pos_or_rpos_obs}, 
                        shared_or_individual_rpos_reward: {self.shared_or_individual_rpos_reward}"""
        )

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

        # placeholders
        drone_prims = self.drone.spawn(
            translations=[(0.0, 0.0, 0.0) for _ in range(12)]
        )

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]  # 23

        observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 6 + 2 + 36

        self.observation_spec = (
            CompositeSpec(
                {
                    "agents": {
                        "observation": UnboundedContinuousTensorSpec(
                            (self.drone.n, observation_dim)
                        ),
                        # "state": UnboundedContinuousTensorSpec(state_dim),
                    }
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.action_spec = (
            CompositeSpec(
                {
                    "agents": {
                        "action": torch.stack(
                            [self.drone.action_spec] * self.drone.n, dim=0
                        ),
                    }
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.reward_spec = (
            CompositeSpec(
                {"agents": {"reward": UnboundedContinuousTensorSpec((self.drone.n, 1))}}
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
            self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state"),
        )

        _stats_spec = CompositeSpec(
            {
                "episode_len": UnboundedContinuousTensorSpec(1),
                "num_turns": UnboundedContinuousTensorSpec(1),
                "num_hits": UnboundedContinuousTensorSpec(1),
                "actor_0_wins": UnboundedContinuousTensorSpec(1),
                "actor_1_wins": UnboundedContinuousTensorSpec(1),
                "draws": UnboundedContinuousTensorSpec(1),
                "terminated": UnboundedContinuousTensorSpec(1),
                "truncated": UnboundedContinuousTensorSpec(1),
                "done": UnboundedContinuousTensorSpec(1),
            }
        )
        self.stats_cfg: DictConfig = self.cfg.task.stats
        if self.stats_cfg.get("win_case", False):
            _stats_spec.set("team_0_case_1", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("team_0_case_2", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("team_0_case_3", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("team_0_case_4", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("team_0_case_5", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("team_0_case_6", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("team_1_case_1", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("team_1_case_2", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("team_1_case_3", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("team_1_case_4", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("team_1_case_5", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("team_1_case_6", UnboundedContinuousTensorSpec(1))

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

    def debug_draw_region(self):

        color = [(0.0, 1.0, 0.0, 1.0)]
        # [topleft, topright, botleft, botright]

        points_start, points_end = rectangular_cuboid_edges(self.L, self.W, 0.0, 3.0)
        points_start = [_carb_float3_add(p, self.central_env_pos) for p in points_start]
        points_end = [_carb_float3_add(p, self.central_env_pos) for p in points_end]

        colors_line = color * len(points_start)
        sizes_line = [1.0] * len(points_start)
        self.draw.draw_lines(points_start, points_end, colors_line, sizes_line)

    def debug_draw_turn(self):
        turn = self.which_side[self.central_env_idx]
        ori = self.envs_positions[self.central_env_idx].detach()

        points_near_side = torch.tensor([2.0, 0.0, 0.0]).to(self.device) + ori
        points_far_side = torch.tensor([-2.0, 0.0, 0.0]).to(self.device) + ori

        colors = [(0, 1, 0, 1)]
        sizes = [20.0]
        if turn.item() == 1:
            points = points_far_side
        else:
            points = points_near_side
        points = [points.tolist()]

        self.draw.clear_points()
        self.draw.draw_points(points, colors, sizes)

    def debug_draw_win(self, game_result):
        env_id = self.central_env_idx
        ori = self.envs_positions[env_id].detach()
        win_near_side = torch.tensor([5.0, 0.0, 0.0]).to(self.device) + ori
        win_far_side = torch.tensor([-5.0, 0.0, 0.0]).to(self.device) + ori

        colors = [(1, 1, 0, 1)]
        sizes = [20.0]

        if game_result[env_id, 0].item() == 1:
            points = win_near_side
        elif game_result[env_id, 0].item() == -1:
            points = win_far_side
        points = [points.tolist()]

        self.draw.draw_points(points, colors, sizes)

    def _reset_idx(self, env_ids: torch.Tensor):
        """_summary_

        Args:
            env_ids (torch.Tensor): (n_envs_to_reset,)
        """

        # if self.central_env_idx in env_ids.tolist():
        #     print("Central environment reset!")

        self.drone._reset_idx(env_ids, self.training)

        drone_pos = self.init_pos[env_ids]
        drone_rpy = torch.zeros((len(env_ids), 12, 3), device=self.device)
        drone_rpy[:, 6:, 2] += 3.1415926

        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(
            torch.zeros((len(env_ids), 12, 6), device=self.device), env_ids
        )

        if self.random_turn:
            turn = torch.randint(
                0, 2, (len(env_ids),), device=self.device
            )  # randomly choose the first team
        else:
            turn = torch.zeros(
                len(env_ids), dtype=torch.long, device=self.device
            )  # always team 0 starts
            # turn = torch.ones(len(env_ids), dtype=torch.long, device=self.device) # always team 1 starts
        self.which_side[env_ids] = turn
        self.last_hit_team[env_ids] = turn  # init

        self.drones_already_hit_in_one_turn[env_ids] = True
        self.drones_already_hit_in_one_turn[env_ids, 6 * turn + 4] = False

        ball_pos = (
            drone_pos[torch.arange(len(env_ids)), 6 * turn + 4, :]
            + self.init_ball_offset
        )
        ball_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            len(env_ids), 1
        )
        self.ball.set_world_poses(
            ball_pos + self.envs_positions[env_ids], ball_rot, env_ids
        )

        ball_initial_vel = torch.zeros(len(env_ids), 6, device=self.device)
        ball_initial_vel[:, 2] = self.ball_initial_z_vel
        self.ball.set_velocities(ball_initial_vel, env_ids)
        self.ball_last_vel[env_ids, 0] = ball_initial_vel[:, :3]

        # fix the mass now
        ball_masses = torch.ones_like(env_ids) * self.ball_mass
        self.ball.set_masses(ball_masses, env_ids)

        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            if self.which_side[self.central_env_idx].item() == 0:
                logging.info("Reset central environment: team 0 starts")
            else:
                logging.info("Reset central environment: team 1 starts")
        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            self.draw.clear_lines()
            point_list_1, point_list_2, colors, sizes = draw_court(
                self.W, self.L, self.H_NET, self.W_NET, n=2
            )
            point_list_1 = [
                _carb_float3_add(p, self.central_env_pos) for p in point_list_1
            ]
            point_list_2 = [
                _carb_float3_add(p, self.central_env_pos) for p in point_list_2
            ]
            self.draw.draw_lines(point_list_1, point_list_2, colors, sizes)
            self.debug_draw_region()
            # self.debug_draw_turn()

        self.last_hit_step[env_ids] = -100

        self.racket_near_ball[env_ids] = False
        self.drone_near_ball[env_ids] = False

        # some stats keys will not reset
        stats_list = []
        for i in self.not_reset_keys_in_stats:
            stats_list.append(self.stats[i][env_ids].clone())
        self.stats[env_ids] = 0.0
        for i, key in enumerate(self.not_reset_keys_in_stats):
            self.stats[key][env_ids] = stats_list[i]

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions: torch.Tensor = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]

        self.ball_pos, _ = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()[..., :3]

        self.rpos = self.ball_pos - self.drone.pos  # (E, 12, 3)

        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )

        team_0_pos_obs = pos.clone()  # (E, 12, 3)
        drones_rpos_team_0 = [(team_0_pos_obs - team_0_pos_obs[:, i, :].unsqueeze(1)).reshape(-1, 36) for i in range(6)]
        # drones_rpos_0 = (team_0_pos_obs - team_0_pos_obs[:, 0, :].unsqueeze(1)).reshape(
        #     -1, 18
        # )
        # drones_rpos_1 = (team_0_pos_obs - team_0_pos_obs[:, 1, :].unsqueeze(1)).reshape(
        #     -1, 18
        # )
        # drones_rpos_2 = (team_0_pos_obs - team_0_pos_obs[:, 2, :].unsqueeze(1)).reshape(
        #     -1, 18
        # )

        idx = torch.cat([torch.arange(6, 12), torch.arange(0, 6)])
        team_1_pos_obs = pos.clone()[:, idx, :]
        team_1_pos_obs[..., :2] = - team_1_pos_obs[..., :2]  # symmetric pos
        drones_rpos_team_1 = [(team_1_pos_obs - team_1_pos_obs[:, i, :].unsqueeze(1)).reshape(-1, 36) for i in range(6)]

        drones_rpos = drones_rpos_team_0 + drones_rpos_team_1

        # rpos
        sym_drones_rpos_obs = torch.stack(drones_rpos, dim=1)  # [E,12,36]

        # pos
        temp_team_0_pos_obs = team_0_pos_obs.reshape(-1, 36).unsqueeze(1).repeat(1, 6, 1)
        temp_team_1_pos_obs = team_1_pos_obs.reshape(-1, 36).unsqueeze(1).repeat(1, 6, 1)
        sym_drones_pos_obs = torch.cat([temp_team_0_pos_obs, temp_team_1_pos_obs], dim=1)  # [E,12,36]

        if self.drones_pos_or_rpos_obs:
            drones_obs = sym_drones_pos_obs
        else:
            drones_obs = sym_drones_rpos_obs

        rot = torch.where(
            (rot[..., 0] < 0).unsqueeze(-1), -rot, rot
        )  # make sure w of (w,x,y,z) is positive
        self.root_state = torch.cat(
            [pos, rot, vel, angular_vel, heading, up, throttle], dim=-1
        )

        self.rpy = quaternion_to_euler(rot)

        self.drone_rot = rot  # world frame

        # root_state:
        near_side_root_state = self.root_state[:, :6, :]
        far_side_root_state = torch.cat(
            [(transfer_root_state_to_the_other_side(self.root_state[:, i, :].unsqueeze(1))) for i in range(6, 12)], 
            dim=1
        )

        sym_root_state = torch.cat([near_side_root_state, far_side_root_state], dim=1)
        
        (
            sym_pos,
            sym_rot,
            sym_vel,
            sym_angular_vel,
            sym_heading,
            sym_up,
            sym_throttle,
        ) = torch.split(
            sym_root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )

        # relative position of the ball to the drone
        self.rpos_ball = self.ball_pos - self.drone.pos  # (E,12,3) world frame
        sym_rpos_ball = self.rpos_ball.clone()
        sym_rpos_ball[:, 6:, :2] = - sym_rpos_ball[:, 6:, :2]

        # relative position of the drone to the anchor
        sym_rpos_anchor = self.drone.pos - self.anchor  # (E,12,3)
        sym_rpos_anchor[:, 6:, :2] = - sym_rpos_anchor[:, 6:, :2]

        sym_ball_vel = self.ball_vel.clone().repeat(1, 12, 1) # (E,12,3)
        sym_ball_vel[:, 6:, :2] = - sym_ball_vel[:, 6:, :2]

        already_hit_in_one_turn_to_obs = torch.nn.functional.one_hot(
            self.drones_already_hit_in_one_turn.long(), num_classes=2
        )

        sym_obs = [
            sym_pos,  # (E,12,3)
            sym_rot,  # (E,12,4)
            sym_vel,  # (E,12,3)
            sym_angular_vel,  # (E,12,3)
            sym_heading,  # (E,12,3)
            sym_up,  # (E,12,3)
            sym_throttle,  # (E,12,4)
            sym_rpos_anchor,  # [E,12,3]
            sym_rpos_ball,  # [E,12,3]
            sym_ball_vel,  # [E,12,3]
            turn_to_obs(self.which_side),  # [E,12,2]
            torch.cat((self.id, self.id), dim=1),  # [E,12,6]
            already_hit_in_one_turn_to_obs,  # [E,12,2]
            drones_obs,  # [E,12,36]
        ]
        sym_obs = torch.cat(sym_obs, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": sym_obs,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.num_envs,
        )

    def check_ball_near_racket(self, racket_radius, cylinder_height_coeff):
        z_direction_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        z_direction_world = quat_rotate(self.drone_rot, z_direction_local)  # (E,N,3)
        normal_vector_world = z_direction_world / torch.norm(
            z_direction_world, dim=-1
        ).unsqueeze(
            -1
        )  # (E,N,3)
        cylinder_bottom_center = self.drone.pos  # (E,N,3) cylinder bottom center
        cylinder_axis = cylinder_height_coeff * self.ball_radius * normal_vector_world

        ball_to_bottom = self.ball_pos - cylinder_bottom_center  # (E,N,3)
        projection_ratio = torch.sum(
            ball_to_bottom * cylinder_axis, dim=-1
        ) / torch.sum(
            cylinder_axis * cylinder_axis, dim=-1
        )  # (E,N) projection of ball_to_bottom on cylinder_axis / cylinder_axis
        within_height = (projection_ratio >= 0) & (projection_ratio <= 1)  # (E,N)

        projection_point = (
            cylinder_bottom_center + projection_ratio.unsqueeze(-1) * cylinder_axis
        )  # (E,N,3)
        distance_to_axis = torch.norm(self.ball_pos - projection_point, dim=-1)  # (E,N)
        within_radius = distance_to_axis <= racket_radius  # (E,N)

        return within_height & within_radius  # (E,N)

    def check_hit(self, sim_dt, racket_radius=0.2, cylinder_height_coeff=2.0):
        racket_near_ball_last_step = self.racket_near_ball.clone()
        drone_near_ball_last_step = self.drone_near_ball.clone()

        self.racket_near_ball = self.check_ball_near_racket(
            racket_radius=racket_radius, cylinder_height_coeff=cylinder_height_coeff
        )  # (E,N)
        self.drone_near_ball = torch.norm(self.rpos_ball, dim=-1) < 0.5  # (E,N)

        ball_vel_z_change = (
            self.ball_vel[..., 2] - self.ball_last_vel[..., 2]
        ) > 9.8 * sim_dt  # (E,1)
        ball_vel_x_y_change = (
            self.ball_vel[..., :2] - self.ball_last_vel[..., :2]
        ).norm(
            dim=-1
        ) > 0.5  # (E,1)
        ball_vel_change = ball_vel_z_change | ball_vel_x_y_change  # (E,1)

        drone_hit_ball = (
            drone_near_ball_last_step | self.drone_near_ball
        ) & ball_vel_change  # (E,N)
        racket_hit_ball = (
            racket_near_ball_last_step | self.racket_near_ball
        ) & ball_vel_change  # (E,N)

        racket_hit_ball = racket_hit_ball & (
            self.progress_buf.unsqueeze(-1) - self.last_hit_step > 3
        )  # (E,N)
        drone_hit_ball = drone_hit_ball & (
            self.progress_buf.unsqueeze(-1) - self.last_hit_step > 3
        )  # (E,N)

        return racket_hit_ball, drone_hit_ball

    def log_results(
        self, draw, game_result, cases, drone_hit_ground, drone_too_close, env_id=0
    ):
        assert draw.shape == torch.Size([self.num_envs, 1])
        assert game_result.shape == torch.Size([self.num_envs, 1])
        assert cases.shape == torch.Size([self.num_envs, 2, 6])
        assert drone_hit_ground.shape == torch.Size([self.num_envs, 12])
        assert drone_too_close.shape == torch.Size([self.num_envs, 12])

        if draw[env_id, 0].item():
            log_str = "Draw! "
        elif game_result[env_id, 0].item() == 1:
            log_str = "Team 0 win! "
        elif game_result[env_id, 0].item() == -1:
            log_str = "Team 1 win! "
        elif drone_hit_ground[env_id, :6].any(-1).item() == 1:
            log_str = "Team 0 drone hit the ground! "
        elif drone_hit_ground[env_id, 6:].any(-1).item() == 1:
            log_str = "Team 1 drone hit the ground! "
        elif drone_too_close[env_id, :6].any(-1).item() == 1:
            log_str = "Team 0 drones are too close to each other! "
        elif drone_too_close[env_id, 6:].any(-1).item() == 1:
            log_str = "Team 1 drones are too close to each other! "
        else:
            log_str = ""

        for team in range(2):
            if cases[env_id, team, 0].item():
                log_str += f"team {team} case 1 drone hit on wrong turn; "
            if cases[env_id, team, 1].item():
                log_str += f"team {team} case 2 drone hit not with racket; "
            if cases[env_id, team, 2].item():
                log_str += f"team {team} case 3 drone hit net; "
            if cases[env_id, team, 3].item():
                log_str += f"team {team} case 4 ball land in court; "
            if cases[env_id, team, 4].item():
                log_str += f"team {team} case 5 hits ball out of court; "
            if cases[env_id, team, 5].item():
                log_str += f"team {team} case 6 hits ball into net; "

        if log_str != "" and self._should_render(0):
            logging.info(log_str)

        for team in range(2):
            for case in range(6):
                self.stats[f"team_{team}_case_{case+1}"] = (
                    cases[:, team, case].unsqueeze(-1).float()
                )

    def _compute_drone_too_close(self, threshold=0.8):

        def _compute_team_too_close(team_drone_pos, threshold=0.8):
            drone_pos_0 = team_drone_pos[:, 0, :]
            drone_pos_1 = team_drone_pos[:, 1, :]
            drone_pos_2 = team_drone_pos[:, 2, :]
            drone_pos_3 = team_drone_pos[:, 3, :]
            drone_pos_4 = team_drone_pos[:, 4, :]
            drone_pos_5 = team_drone_pos[:, 5, :]

            too_close_01 = torch.norm(drone_pos_0 - drone_pos_1, dim=-1) < threshold  # (E,)
            too_close_02 = torch.norm(drone_pos_0 - drone_pos_2, dim=-1) < threshold  # (E,)
            too_close_03 = torch.norm(drone_pos_0 - drone_pos_3, dim=-1) < threshold  # (E,)
            too_close_04 = torch.norm(drone_pos_0 - drone_pos_4, dim=-1) < threshold  # (E,)
            too_close_05 = torch.norm(drone_pos_0 - drone_pos_5, dim=-1) < threshold  # (E,)
            too_close_12 = torch.norm(drone_pos_1 - drone_pos_2, dim=-1) < threshold  # (E,)
            too_close_13 = torch.norm(drone_pos_1 - drone_pos_3, dim=-1) < threshold  # (E,)
            too_close_14 = torch.norm(drone_pos_1 - drone_pos_4, dim=-1) < threshold  # (E,)
            too_close_15 = torch.norm(drone_pos_1 - drone_pos_5, dim=-1) < threshold  # (E,)
            too_close_23 = torch.norm(drone_pos_2 - drone_pos_3, dim=-1) < threshold  # (E,)
            too_close_24 = torch.norm(drone_pos_2 - drone_pos_4, dim=-1) < threshold  # (E,)
            too_close_25 = torch.norm(drone_pos_2 - drone_pos_5, dim=-1) < threshold  # (E,)
            too_close_34 = torch.norm(drone_pos_3 - drone_pos_4, dim=-1) < threshold  # (E,)
            too_close_35 = torch.norm(drone_pos_3 - drone_pos_5, dim=-1) < threshold  # (E,)
            too_close_45 = torch.norm(drone_pos_4 - drone_pos_5, dim=-1) < threshold  # (E,)
            team_too_close = torch.stack(
                [
                    (too_close_01 | too_close_02 | too_close_03 | too_close_04 | too_close_05),
                    (too_close_01 | too_close_12 | too_close_13 | too_close_14 | too_close_15),
                    (too_close_02 | too_close_12 | too_close_23 | too_close_24 | too_close_25),
                    (too_close_03 | too_close_13 | too_close_23 | too_close_34 | too_close_35),
                    (too_close_04 | too_close_14 | too_close_24 | too_close_34 | too_close_45),
                    (too_close_05 | too_close_15 | too_close_25 | too_close_35 | too_close_45),
                ],
                dim=1,
            )  # (E, 6)

            return team_too_close

        team_0_too_close = _compute_team_too_close(self.drone.pos[:, :6, :], threshold)
        team_1_too_close = _compute_team_too_close(self.drone.pos[:, 6:, :], threshold)

        return torch.cat([team_0_too_close, team_1_too_close], dim=1)  # (E, 12)

    def _rpos_to_dec_reward(self) -> torch.Tensor:
        """
        compute decentralized reward for team members

        Returns:
            reward: [E, 12]

        """
        rpos = self.drone.pos - self.ball_pos

        effective_dist = (1 - self.drones_already_hit_in_one_turn.float()) * rpos.norm(
            p=2, dim=-1
        )  # [E, 12]
        effective_dist[effective_dist == 0] = float("inf")

        return effective_dist  # [E, 12]

    def _rpos_to_shared_reward(self) -> torch.Tensor:
        """
        compute shared reward for team members

        Returns:
            reward: [E, 2]

        """
        rpos = self.drone.pos - self.ball_pos

        effective_dist = (1 - self.drones_already_hit_in_one_turn.float()) * rpos.norm(
            p=2, dim=-1
        )  # [E, 12]
        effective_dist[effective_dist == 0] = float("inf")

        team_0_dist_min = effective_dist[:, :6].min(dim=1).values  # [E]
        team_1_dist_min = effective_dist[:, 6:].min(dim=1).values  # [E]

        return torch.stack([team_0_dist_min, team_1_dist_min], dim=1)  # [E, 2]

    def _compute_reward_and_done(self):
        racket_hit_ball, drone_hit_ball = self.check_hit(sim_dt=self.dt)  # (E, 12)
        wrong_hit_racket = (
            drone_hit_ball & ~racket_hit_ball
        )

        true_hit = (
            racket_hit_ball & ~self.drones_already_hit_in_one_turn
        )
        wrong_hit_turn = racket_hit_ball & self.drones_already_hit_in_one_turn

        self.drones_already_hit_in_one_turn = (
            self.drones_already_hit_in_one_turn | racket_hit_ball
        )

        self.last_hit_team = torch.where(
            true_hit[:, :6].any(-1), 0, self.last_hit_team
        )  # (E,)
        self.last_hit_team = torch.where(
            true_hit[:, 6:].any(-1), 1, self.last_hit_team
        )  # (E,)

        drone_hit_ground = self.drone.pos[..., 2] < 0.3
        drone_too_close = self._compute_drone_too_close()

        self.ball_last_vel = self.ball_vel.clone()
        self.last_hit_step[racket_hit_ball] = self.progress_buf[
            racket_hit_ball.any(-1)
        ].long()

        penalty_drone_hit_ground = (
            self.drone_hit_ground_penalty_coef * drone_hit_ground
        )  # (E,12)
        penalty_drone_too_close = self.drone_too_close_penalty_coef * drone_too_close


        if self.shared_or_individual_rpos_reward:
            reward_rpos = self.rpos_reward_coef / (
                1 + self._rpos_to_shared_reward()
            )  # (E,2) shared
            reward_rpos = torch.stack(
                [reward_rpos[:, 0] for _ in range(6)] + [reward_rpos[:, 1] for _ in range(6)],
                dim=1,
            )  # (E,12) shared
        else:
            reward_rpos = self.rpos_reward_coef / (
                1 + self._rpos_to_dec_reward()
            )  # (E,12) not shared

        reward_hit = (
            self.hit_reward_coef * true_hit_to_reward(true_hit).float()
        )  # (E,12) shared

        reward_ball_vel_x = (
            self.ball_vel_x_reward_coef
            * true_hit_to_reward(true_hit).float()
            * self.ball_vel_x_dir_world
        )  # (E,12) shared

        # reward_rpos_anchor = self.rpos_anchor_reward_coef * (1 - turn_to_mask(self.which_side, n=6).float()) / (1 + (self.drone.pos - self.anchor).norm(p=2, dim=-1)) # (E,6)
        reward_rpos_anchor = self.rpos_anchor_reward_coef / (
            1 + (self.drone.pos - self.anchor).norm(p=2, dim=-1)
        )  # (E,12)

        penalty_rpy = self.rpy_penalty_coef * (
            self.rpy[:, :, 0].abs()
            + self.rpy[:, :, 1].abs()
            + 5 * (self.rpy[:, :, 2] - self.init_rpy[:, :, 2]).abs()
        )  # (E,12)
        penalty_pos_z = (
            self.pos_z_penalty_coef
            * self.drones_already_hit_in_one_turn.float()
            * (self.drone.pos[..., 2] - 2.0).abs()
        )  # (E,12)

        # self.stats["num_hits_drone_0"] += true_hit[:, 0].float().unsqueeze(-1)
        # self.stats["num_hits_drone_1"] += true_hit[:, 1].float().unsqueeze(-1)
        # self.stats["num_hits_drone_2"] += true_hit[:, 2].float().unsqueeze(-1)

        # switch turn
        last_which_side = self.which_side.clone()
        self.which_side = torch.where(
            self.ball_pos[..., 0].squeeze(-1) > 0, 0, 1
        )  # update new turn
        switch_side = torch.where(
            (self.which_side ^ last_which_side).bool(), True, False
        )  # (E,)
        # if self._should_render(0) and switch_side[self.central_env_idx].item():
        #     self.debug_draw_turn()

        switch_side_idx = torch.nonzero(switch_side, as_tuple=True)[0]
        self.drones_already_hit_in_one_turn[switch_side_idx] = torch.where(
            self.which_side[switch_side_idx].unsqueeze(-1) == 0,
            torch.tensor(
                [False for _ in range(6)] + [True for _ in range(6)], device=self.device
            ).unsqueeze(0),
            torch.tensor(
                [True for _ in range(6)] + [False for _ in range(6)], device=self.device
            ).unsqueeze(0),
        )

        draw, game_result, cases = determine_game_result_6v6(
            L=self.L,
            W=self.W,
            H_NET=self.H_NET,
            ball_radius=self.ball_radius,
            ball_pos=self.ball_pos,
            last_hit_team=self.last_hit_team,
            wrong_hit_turn=wrong_hit_turn,
            wrong_hit_racket=wrong_hit_racket,
            drone_pos=self.drone.pos,
            anchor=self.anchor,
        )  # (E,1)

        if self._should_render(0) and game_result[self.central_env_idx].item():
            self.draw.clear_points()
            self.debug_draw_win(game_result)

        self.log_results(
            draw,
            game_result,
            cases,
            drone_hit_ground,
            drone_too_close,
            self.central_env_idx,
        )

        # # reward
        reward_win = self.game_reward_coef * game_result_to_matrix(game_result)  # (E,2)
        reward_win = torch.stack(
            [reward_win[:, 0] for i in range(6)] + [reward_win[:, 1] for i in range(6)],
            dim=1,
        )  # (E,12) shared

        misbahave = -penalty_drone_hit_ground - penalty_drone_too_close
        task = reward_win
        # shaping = reward_rpos + reward_hit + reward_rpos_anchor + reward_ball_vel_x - penalty_rpy - penalty_pos_z
        shaping = reward_rpos + reward_hit + reward_rpos_anchor

        reward = misbahave + task + shaping

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated = (
            (game_result != 0)
            | draw
            | drone_hit_ground.any(-1, keepdim=True)
            | drone_too_close.any(-1, keepdim=True)
        )
        done = truncated | terminated  # [E, 1]

        self.stats["actor_0_wins"] = (game_result == 1).float()
        self.stats["actor_1_wins"] = (game_result == -1).float()
        self.stats["draws"] = done.float() - (game_result != 0).float()

        self.stats["truncated"][:] = truncated.float()
        self.stats["terminated"][:] = terminated.float()
        self.stats["done"][:] = done.float()
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["num_turns"].add_(switch_side.float().unsqueeze(-1))
        self.stats["num_hits"].add_(true_hit.any(dim=-1, keepdim=True).float())

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "stats": self.stats,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
