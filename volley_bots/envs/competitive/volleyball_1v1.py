import logging
import os
from typing import Dict, List

import omni.isaac.core.materials as materials
import omni.isaac.core.objects as objects
import torch
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

from .utils.rules import determine_game_result, game_result_to_matrix


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


def get_extended_rpy_dist(dist_cfg: Dict, drone_0_near_side: bool, device):
    near_low = torch.tensor(dist_cfg["low"], device=device)
    near_high = torch.tensor(dist_cfg["high"], device=device)
    far_low = near_low.clone()
    far_low[-1] = 3.1415926
    far_high = near_high.clone()
    far_high[-1] = 3.1415926

    if drone_0_near_side:
        return D.Uniform(
            low=torch.stack((near_low, far_low), dim=0),
            high=torch.stack((near_high, far_high), dim=0),
        )
    else:
        return D.Uniform(
            low=torch.stack((far_low, near_low), dim=0),
            high=torch.stack((far_high, near_high), dim=0),
        )


def get_extended_pos_dist(
    x_low: float,
    y_low: float,
    z_low: float,
    x_high: float,
    y_high: float,
    z_high: float,
    drone_0_near_side: bool,
    device,
):
    """_summary_

    Args:
        initial position distribution of drone_near_side
        device (_type_): _description_

    Returns:
        _type_: _description_
    """

    if drone_0_near_side:  # drone 0 is at the near side of the court
        return D.Uniform(
            low=torch.tensor(
                [
                    [x_low, y_low, z_low],
                    [-x_high, y_low, z_low],
                ],
                device=device,
            ),
            high=torch.tensor(
                [
                    [x_high, y_high, z_high],
                    [-x_low, y_high, z_high],
                ],
                device=device,
            ),
        )
    else:  # drone 0 is at the far side of the court
        return D.Uniform(
            low=torch.tensor(
                [
                    [-x_high, y_low, z_low],
                    [x_low, y_low, z_low],
                ],
                device=device,
            ),
            high=torch.tensor(
                [
                    [-x_low, y_high, z_high],
                    [x_high, y_high, z_high],
                ],
                device=device,
            ),
        )


def turn_to_obs(t: torch.Tensor, symmetric_obs: bool = False) -> torch.Tensor:
    """convert representation of drone turn to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env, 2, 2)
    """
    if symmetric_obs:
        table = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            device=t.device,
        )
    else:
        table = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]],
            device=t.device,
        )

    return table[t]


def turn_to_mask(turn: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        turn (torch.Tensor): (*,)

    Returns:
        torch.Tensor: (*,2)
    """
    table = torch.tensor([[True, False], [False, True]], device=turn.device)
    return table[turn]


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


class Volleyball1v1(IsaacEnv):
    """Two drones hit the ball adversarially.

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
        self.symmetric_obs: bool = cfg.task.get("symmetric_obs", False)

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

        contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/ball",
        )
        self.contact_sensor: ContactSensor = contact_sensor_cfg.class_type(
            contact_sensor_cfg
        )
        self.contact_sensor._initialize_impl()

        self.drone_0_near_side = cfg.task.initial.drone_0_near_side
        if self.drone_0_near_side:
            self.anchor = torch.tensor(
                cfg.task.anchor, device=self.device
            )  # (2,3) original positions of two drones without any offset
        else:
            self.anchor = torch.tensor(cfg.task.anchor, device=self.device)
            self.anchor[0, 0] = -self.anchor[0, 0]
            self.anchor[1, 0] = -self.anchor[1, 0]

        self.init_ball_offset = torch.tensor(
            cfg.task.initial.ball_offset, device=self.device
        )
        self.init_on_the_spot = cfg.task.initial.get("init_on_the_spot", False)
        self.init_drone_pos_dist = get_extended_pos_dist(
            *cfg.task.initial.drone_xyz_dist_near.low,
            *cfg.task.initial.drone_xyz_dist_near.high,
            drone_0_near_side=self.drone_0_near_side,
            device=self.device,
        )
        self.init_drone_rpy_dist = get_extended_rpy_dist(
            cfg.task.initial.drone_rpy_dist_near,
            drone_0_near_side=self.drone_0_near_side,
            device=self.device,
        )  # unit: \pi

        # (n_envs,) 0/1
        self.turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_hit_step = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.int64
        )
        self.ball_last_vel = torch.zeros(
            self.num_envs, 1, 3, device=self.device, dtype=torch.float32
        )
        self._reward_drone_0 = torch.zeros(self.num_envs, device=self.device)
        self._reward_drone_1 = torch.zeros(self.num_envs, device=self.device)
        self._num_hits_drone_0 = torch.zeros(self.num_envs, device=self.device)
        self._num_hits_drone_1 = torch.zeros(self.num_envs, device=self.device)

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.random_turn = cfg.task.get("random_turn", False)

        # one-hot id [E,2,2]
        self.id = torch.zeros((cfg.task.env.num_envs, 2, 2), device=self.device)
        self.id[:, 0, 0] = 1
        self.id[:, 1, 1] = 1

        self.racket_near_ball = torch.zeros(
            (cfg.task.env.num_envs, 2), device=self.device, dtype=torch.bool
        )
        self.drone_near_ball = torch.zeros(
            (cfg.task.env.num_envs, 2), device=self.device, dtype=torch.bool
        )

        self.not_reset_keys_in_stats = [
            "actor_0_wins",
            "actor_1_wins",
            "terminated",
            "truncated",
            "done",
        ]



        # coefs
        self.game_reward_coef = 100.0
        self.hit_reward_coef = 5.0
        self.rpos_reward_coef = 0.5
        self.drone_pos_penalty_coef = 0.2
        self.drone_hit_ground_penalty_coef = 100
        logging.info(
            f"""All reward coefs:
                     game_reward_coef: {self.game_reward_coef}, 
                     hit_reward_coef: {self.hit_reward_coef}, 
                     rpos_reward_coef: {self.rpos_reward_coef}, 
                     drone_pos_penalty_coef: {self.drone_pos_penalty_coef}
                     drone_hit_ground_penalty_coef: {self.drone_hit_ground_penalty_coef}"""
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
            translations=[
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 2.0),
            ]
        )

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]  # 23

        if self.symmetric_obs:
            observation_dim = drone_state_dim + 3 + 3 + 3 + 3 + 2
        else:
            observation_dim = drone_state_dim + 3 + 3 + 3 + 3 + 2 + 2

        self.observation_spec = (
            CompositeSpec(
                {
                    "agents": {
                        "observation": UnboundedContinuousTensorSpec(
                            (2, observation_dim)
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
            CompositeSpec({"agents": {"reward": UnboundedContinuousTensorSpec((2, 1))}})
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
            state_key=("agents", "state"),
        )

        _stats_spec = CompositeSpec(
            {
                "episode_len": UnboundedContinuousTensorSpec(1),
                "num_hits": UnboundedContinuousTensorSpec(1),
                "num_hits_drone_0": UnboundedContinuousTensorSpec(1),
                "num_hits_drone_1": UnboundedContinuousTensorSpec(1),
                "win_reward": UnboundedContinuousTensorSpec(1),
                "lose_reward": UnboundedContinuousTensorSpec(1),
                "drone_0_reward": UnboundedContinuousTensorSpec(1),
                "drone_1_reward": UnboundedContinuousTensorSpec(1),
                "actor_0_wins": UnboundedContinuousTensorSpec(1),
                "actor_1_wins": UnboundedContinuousTensorSpec(1),
                "terminated": UnboundedContinuousTensorSpec(1),
                "truncated": UnboundedContinuousTensorSpec(1),
                "done": UnboundedContinuousTensorSpec(1),
            }
        )
        self.stats_cfg: DictConfig = self.cfg.task.stats
        if self.stats_cfg.get("drone_height", False):
            # 0: cummulative deviation 1: count(num_hits)
            _stats_spec.set("drone_height", UnboundedContinuousTensorSpec(1))
        if self.stats_cfg.get("win_case", False):
            _stats_spec.set("drone_0_case_1", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_case_2", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_case_3", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_case_4", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_case_5", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_case_6", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_case_1", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_case_2", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_case_3", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_case_4", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_case_5", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_case_6", UnboundedContinuousTensorSpec(1))
        if self.stats_cfg.get("drone_0_complete_reward", False):
            _stats_spec.set("drone_0_reward_rpos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_reward_hit", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_reward_win", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_penalty_pos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set(
                "drone_0_penalty_hit_ground", UnboundedContinuousTensorSpec(1)
            )
        if self.stats_cfg.get("drone_1_complete_reward", False):
            _stats_spec.set("drone_1_reward_rpos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_reward_hit", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_reward_win", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_penalty_pos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set(
                "drone_1_penalty_hit_ground", UnboundedContinuousTensorSpec(1)
            )
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

    def debug_draw_turn(self, drone_0_near_side: bool):
        turn = self.turn[self.central_env_idx]
        ori = self.envs_positions[self.central_env_idx].detach()
        points = self.anchor.clone() + ori
        points[:, -1] = 0
        points = points.tolist()
        colors = [(0, 1, 0, 1), (1, 0, 0, 1)]
        sizes = [20.0, 20.0]
        if (turn.item() == 1 and drone_0_near_side) or (
            turn.item() == 0 and not drone_0_near_side
        ):
            colors = colors[::-1]
        # self.draw.clear_points()
        self.draw.draw_points(points, colors, sizes)

    def debug_draw_win(self, game_result):
        env_id = self.central_env_idx
        ori = self.envs_positions[env_id].detach()
        win_near_side = torch.tensor([2.0, 0.0, 0.0]).to(self.device) + ori
        win_far_side = torch.tensor([-2.0, 0.0, 0.0]).to(self.device) + ori

        colors = [(1, 1, 0, 1)]
        sizes = [20.0]

        if game_result[env_id, 0].item() == 1:
            if self.drone_0_near_side:
                points = win_near_side
            else:
                points = win_far_side
        elif game_result[env_id, 0].item() == -1:
            if self.drone_0_near_side:
                points = win_far_side
            else:
                points = win_near_side
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

        if self.init_on_the_spot:
            drone_rpy = torch.tensor(
                [[0.0, 0.0, 0.0], [0, 0, 3.1415926]], device=self.device
            )
            if self.drone_0_near_side:
                drone_pos = self.anchor.expand(len(env_ids), 2, 3)
                drone_rpy = drone_rpy.expand(len(env_ids), 2, 3)
            else:
                drone_pos = self.anchor[:, [1, 0]].expand(len(env_ids), 2, 3)
                drone_rpy = drone_rpy[:, [1, 0]].expand(len(env_ids), 2, 3)
        else:
            drone_pos = self.init_drone_pos_dist.sample(env_ids.shape) + self.anchor
            drone_rpy = self.init_drone_rpy_dist.sample(env_ids.shape)

        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(
            torch.zeros(len(env_ids), 2, 6, device=self.device), env_ids
        )

        if self.random_turn:
            turn = torch.randint(
                0, 2, (len(env_ids),), device=self.device
            )  # randomly choose the first player
        else:
            turn = torch.zeros(
                len(env_ids), dtype=torch.long, device=self.device
            )  # always player 0 starts
            # turn = torch.ones(len(env_ids), dtype=torch.long, device=self.device) # always player 1 starts
        self.turn[env_ids] = turn

        ball_pos = (
            drone_pos[torch.arange(len(env_ids)), turn, :] + self.init_ball_offset
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
            if self.turn[self.central_env_idx].item() == 0:
                logging.info("Reset central environment: Player 0 starts")
            else:
                logging.info("Reset central environment: Player 1 starts")
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
            self.debug_draw_turn(self.drone_0_near_side)

        self.last_hit_step[env_ids] = -100
        self._reward_drone_0[env_ids] = 0.0
        self._reward_drone_1[env_ids] = 0.0
        self._num_hits_drone_0[env_ids] = 0.0
        self._num_hits_drone_1[env_ids] = 0.0

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

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.contact_sensor.update(self.dt)

    def _compute_state_and_obs(self):
        # clone here
        self.root_state = self.drone.get_state()
        # pos(3), quat(4), vel, omega
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.ball_pos, _ = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()[..., :3]

        if self.symmetric_obs:
            root_state = self.root_state.clone()
            pos = root_state[..., :3]
            pos[:, 1, :2] = -pos[:, 1, :2]
            rot = root_state[..., 3:7]
            rot[:, 0, :] = torch.where(
                (rot[:, 0, 0] < 0).unsqueeze(-1), -rot[:, 0, :], rot[:, 0, :]
            )  # make sure w of (w,x,y,z) is positive

            def quaternion_multiply(q1, q2):
                assert q1.shape == q2.shape and q1.shape[-1] == 4
                w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
                w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
                w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                return torch.stack([w, x, y, z], dim=-1)


            q_m = torch.tensor([[0, 0, 0, 1]] * rot.shape[0], device=rot.device)
            rot[:, 1, :] = quaternion_multiply(q_m, rot[:, 1, :])
            rot[:, 1, :] = torch.where(
                (rot[:, 1, 0] < 0).unsqueeze(-1), -rot[:, 1, :], rot[:, 1, :]
            )  # make sure w of (w,x,y,z) is positive
            self.drone_rot = rot

            vel = root_state[..., 7:10]
            vel[:, 1, :2] = -vel[:, 1, :2]
            omega = root_state[..., 10:13]
            omega[:, 1, :2] = -omega[:, 1, :2]
            heading = root_state[..., 13:16]
            heading[:, 1, :] = quat_axis(rot[:, 1, :], axis=0)
            up = root_state[..., 16:19]
            up[:, 1, :] = quat_axis(rot[:, 1, :], axis=2)
            rotors = root_state[..., 19:23]

            # relative position and heading
            self.rpos_ball = self.drone.pos - self.ball_pos
            rpos_ball = self.rpos_ball.clone()
            rpos_ball[:, 1, :2] = -rpos_ball[:, 1, :2]
            self.rpos_drone = torch.stack(
                [
                    # [..., drone_id, [x, y, z]]
                    self.drone.pos[..., 1, :] - self.drone.pos[..., 0, :],
                    self.drone.pos[..., 0, :] - self.drone.pos[..., 1, :],
                ],
                dim=1,
            )  # (E,2,3)
            rpos_drone = self.rpos_drone.clone()
            rpos_drone[:, 1, :2] = -rpos_drone[:, 1, :2]
            rpos_anchor = self.drone.pos - self.anchor  # (E,2,3)
            rpos_anchor[:, 1, :2] = -rpos_anchor[:, 1, :2]
            ball_vel = self.ball_vel.clone().repeat(1, 2, 1)
            ball_vel[:, 1, :2] = -ball_vel[:, 1, :2]

            obs = [
                pos,  # (E,2,3)
                rot,  # (E,2,4)
                vel,  # (E,2,3)
                omega,  # (E,2,3)
                heading,  # (E,2,3)
                up,  # (E,2,3)
                rotors,  # (E,2,4)
                rpos_anchor,  # [E,2,3]
                rpos_drone,  # [E,2,3]
                rpos_ball,  # [E,2,3]
                ball_vel,  # [E,2,3]
                turn_to_obs(self.turn, symmetric_obs=True),  # [E,2,2]
                # self.id, # [E,2,2]
            ]
            obs = torch.cat(obs, dim=-1)

        else:
            # relative position and heading
            self.rpos_ball = self.drone.pos - self.ball_pos
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
                rpos_anchor,  # [E,2,3]
                self.rpos_drone,  # [E,2,3]
                self.rpos_ball,  # [E,2,3]
                self.ball_vel.expand(-1, 2, 3),  # [E,2,3]
                turn_to_obs(self.turn),  # [E,2,2]
                self.id,  # [E,2,2]
            ]
            obs = torch.cat(obs, dim=-1)

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

    def log_results(self, draw, game_result, cases, drone_hit_ground, env_id=0):
        assert draw.shape == torch.Size([self.num_envs, 1])
        assert game_result.shape == torch.Size([self.num_envs, 1])
        assert cases.shape == torch.Size([self.num_envs, 2, 6])
        assert drone_hit_ground.shape == torch.Size([self.num_envs, 2])

        if draw[env_id, 0].item():
            log_str = "Draw! "
        elif game_result[env_id, 0].item() == 1:
            log_str = "Drone 0 win! "
        elif game_result[env_id, 0].item() == -1:
            log_str = "Drone 1 win! "
        elif drone_hit_ground[env_id, 0].item() == 1:
            log_str = "Drone 0 hit the ground! "
        elif drone_hit_ground[env_id, 1].item() == 1:
            log_str = "Drone 1 hit the ground! "
        else:
            log_str = ""

        for drone in range(2):
            if cases[env_id, drone, 0].item():
                log_str += f"Drone {drone} case 1 drone hit on wrong turn; "
            if cases[env_id, drone, 1].item():
                log_str += f"Drone {drone} case 2 drone hit not with racket; "
            if cases[env_id, drone, 2].item():
                log_str += f"Drone {drone} case 3 drone hit net; "
            if cases[env_id, drone, 3].item():
                log_str += f"Drone {drone} case 4 ball land in court; "
            if cases[env_id, drone, 4].item():
                log_str += f"Drone {drone} case 5 hits ball out of court; "
            if cases[env_id, drone, 5].item():
                log_str += f"Drone {drone} case 6 hits ball into net; "

        if log_str != "" and self._should_render(0):
            logging.info(log_str)

        for drone in range(2):
            for case in range(6):
                self.stats[f"drone_{drone}_case_{case+1}"] = (
                    cases[:, drone, case].unsqueeze(-1).float()
                )

    def _compute_reward_and_done(self):
        racket_hit_ball, drone_hit_ball = self.check_hit(sim_dt=self.dt)
        wrong_hit_racket = (
            drone_hit_ball & ~racket_hit_ball
        )
        which_drone = self.rpos_ball.norm(p=2, dim=-1).argmin(
            dim=1
        )  # (E,1) which drone is closer to the ball
        true_hit = racket_hit_ball & (self.turn == which_drone).unsqueeze(
            -1
        )
        switch_turn = true_hit.any(-1, keepdim=True)  # (E, 1)
        wrong_hit_turn = racket_hit_ball & (self.turn != which_drone).unsqueeze(
            -1
        )
        drone_hit_ground = self.drone.pos[..., 2] < 0.3

        self.ball_last_vel = self.ball_vel.clone()
        self.last_hit_step[racket_hit_ball] = self.progress_buf[
            racket_hit_ball.any(-1)
        ].long()


        penalty_drone_hit_ground = self.drone_hit_ground_penalty_coef * drone_hit_ground
        penalty_pos = self.drone_pos_penalty_coef * calculate_penalty_drone_pos(
            self.drone.pos, self.L, self.W, self.anchor
        )  # (E,2)

        reward_rpos = self.rpos_reward_coef * (
            turn_to_mask(self.turn).float()
            / (1 + torch.norm(self.rpos_ball[..., :2], p=2, dim=-1))
        )  # (E,2)
        reward_hit = (
            self.hit_reward_coef * turn_to_mask(self.turn).float() * switch_turn.float()
        )  # (E,2)

        self._num_hits_drone_0.add_(
            (self.turn == 0).float() * switch_turn.squeeze(-1).float()
        )
        self._num_hits_drone_1.add_(
            (self.turn == 1).float() * switch_turn.squeeze(-1).float()
        )
        self.stats["num_hits_drone_0"] = self._num_hits_drone_0.unsqueeze(-1)
        self.stats["num_hits_drone_1"] = self._num_hits_drone_1.unsqueeze(-1)

        # switch turn
        self.turn = (self.turn + switch_turn.squeeze(-1).long()) % 2  # update new turn
        if self._should_render(0) and switch_turn[self.central_env_idx].item():
            self.debug_draw_turn(self.drone_0_near_side)

        draw, game_result, cases = determine_game_result(
            L=self.L,
            W=self.W,
            H_NET=self.H_NET,
            ball_radius=self.ball_radius,
            ball_pos=self.ball_pos,
            turn=self.turn,
            wrong_hit_turn=wrong_hit_turn,
            wrong_hit_racket=wrong_hit_racket,
            drone_pos=self.drone.pos,
            anchor=self.anchor,
        )  # (E,1)

        if self._should_render(0) and game_result[self.central_env_idx].item():
            self.draw.clear_points()
            self.debug_draw_win(game_result)

        self.log_results(
            draw, game_result, cases, drone_hit_ground, self.central_env_idx
        )

        # reward
        reward_win = self.game_reward_coef * game_result_to_matrix(game_result)  # (E,2)

        misbahave = -penalty_pos - penalty_drone_hit_ground
        task = reward_win
        shaping = reward_rpos + reward_hit
        reward = misbahave + task + shaping

        self._reward_drone_0.add_(reward[:, 0])
        self._reward_drone_1.add_(reward[:, 1])
        if self.stats_cfg.get("drone_0_complete_reward", False):
            self.stats["drone_0_reward_rpos"].add_(reward_rpos[:, 0].unsqueeze(-1))
            self.stats["drone_0_reward_hit"].add_(reward_hit[:, 0].unsqueeze(-1))
            self.stats["drone_0_reward_win"].add_(reward_win[:, 0].unsqueeze(-1))
            self.stats["drone_0_penalty_pos"].sub_(penalty_pos[:, 0].unsqueeze(-1))
            self.stats["drone_0_penalty_hit_ground"].add_(
                penalty_drone_hit_ground[:, 0].unsqueeze(-1)
            )
        if self.stats_cfg.get("drone_1_complete_reward", False):
            self.stats["drone_1_reward_rpos"].add_(reward_rpos[:, 1].unsqueeze(-1))
            self.stats["drone_1_reward_hit"].add_(reward_hit[:, 1].unsqueeze(-1))
            self.stats["drone_1_reward_win"].add_(reward_win[:, 1].unsqueeze(-1))
            self.stats["drone_1_penalty_pos"].sub_(penalty_pos[:, 1].unsqueeze(-1))
            self.stats["drone_1_penalty_hit_ground"].add_(
                penalty_drone_hit_ground[:, 1].unsqueeze(-1)
            )

        self.stats["drone_0_reward"] = self._reward_drone_0.unsqueeze(-1)
        self.stats["drone_1_reward"] = self._reward_drone_1.unsqueeze(-1)
        self.stats["win_reward"] = (
            game_result == -1
        ).float() * self._reward_drone_1.unsqueeze(-1)
        self.stats["win_reward"] += (
            game_result == 1
        ).float() * self._reward_drone_0.unsqueeze(-1)
        self.stats["lose_reward"] = (
            game_result == -1
        ).float() * self._reward_drone_0.unsqueeze(-1)
        self.stats["lose_reward"] += (
            game_result == 1
        ).float() * self._reward_drone_1.unsqueeze(-1)
        self.stats["actor_0_wins"] = (game_result == 1).float()
        self.stats["actor_1_wins"] = (game_result == -1).float()

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated = (game_result != 0) | draw | drone_hit_ground.any(-1, keepdim=True)
        done = truncated | terminated  # [E, 1]

        self.stats["truncated"][:] = truncated.float()
        self.stats["terminated"][:] = terminated.float()
        self.stats["done"][:] = done.float()
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["num_hits"].add_(switch_turn.float())
        self.stats["drone_height"] = self.drone.pos[..., 2].mean(dim=-1, keepdim=True)

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
