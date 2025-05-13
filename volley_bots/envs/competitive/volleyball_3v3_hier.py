from omegaconf import DictConfig

import volley_bots.utils.kit as kit_utils
from volley_bots.utils.torch import euler_to_quaternion
import omni.isaac.core.objects as objects
import omni.isaac.core.materials as materials
import torch
import torch.distributions as D

from volley_bots.envs.isaac_env import AgentSpec, IsaacEnv
from volley_bots.robots.drone import MultirotorBase
from volley_bots.views import RigidPrimView

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
)
from omni.isaac.debug_draw import _debug_draw

from pxr import UsdShade, PhysxSchema


from typing import Dict, List

from .draw import draw_court
from .rules import determine_game_result_3v3, game_result_to_matrix

from carb import Float3

from omni.isaac.orbit.sensors import ContactSensorCfg, ContactSensor

from .common import rectangular_cuboid_edges,_carb_float3_add


from volley_bots.utils.torch import quat_axis
import logging
import os

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
    rot_matrix = torch.stack([
        ww + xx - yy - zz,
        2 * (xy - wz),
        2 * (xz + wy),
        
        2 * (xy + wz),
        ww - xx + yy - zz,
        2 * (yz - wx),
        
        2 * (xz - wy),
        2 * (yz + wx),
        ww - xx - yy + zz
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    v_expanded = v.expand(*q.shape[:-1], 3)
    
    # Rotate the vector using the rotation matrix
    return torch.matmul(rot_matrix, v_expanded.unsqueeze(-1)).squeeze(-1)


def turn_to_obs(t: torch.Tensor) -> torch.Tensor:
    """convert representation of drone turn to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env, 6, 2)
    """
    table = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ],
        device=t.device,
    ) # (2, 6, 2)

    return table[t]


def turn_to_mask(turn: torch.Tensor, n: int = 2) -> torch.Tensor:
    """_summary_

    Args:
        turn (torch.Tensor): (*,)

    Returns:
        torch.Tensor: (*,n)
    """
    assert n in [2, 6]
    if n == 2:
        table = torch.tensor([[True, False], [False, True]], device=turn.device)
    elif n == 6:
        table = torch.tensor(
            [
                [True, True, True, False, False, False],
                [False, False, False, True, True, True],
            ],
            device=turn.device,
        )
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
    drone_pos: torch.Tensor, L: float, W: float, anchor, dist_x=0.5, dist_y=0.5, dist_z=1
) -> torch.Tensor:
    """_summary_

    Args:
        drone_pos (torch.Tensor): (E,4,3)
        L (float): _description_
        W (float): _description_

    Returns:
        torch.Tensor: _description_
    """

    # 
    penalty_x = (drone_pos[:, :, 0].abs() < dist_x).float() * (dist_x - drone_pos[:, :, 0].abs()) + (
        (L / 2 - drone_pos[:, :, 0].abs()) < dist_x).float() * (dist_x - (L / 2 - drone_pos[:, :, 0].abs()))  # (E,2)

    # 
    penalty_y = (drone_pos[:, :, 1].abs() < dist_y).float() * (dist_y - drone_pos[:, :, 1].abs()) + (
        (W / 2 - drone_pos[:, :, 1].abs()) < dist_y).float() * (dist_y - (W / 2 - drone_pos[:, :, 1].abs()))  # (E,2)

    # 
    penalty_z = ((drone_pos[:, :, 2] - anchor[:, 2]).abs() > dist_z) * ((drone_pos[:, :, 2] - anchor[:, 2]).abs() - dist_z)  # (E,2)

    penalty = penalty_x + penalty_y + penalty_z
    return penalty


def true_hit_to_reward(true_hit: torch.Tensor) -> torch.Tensor:
    '''
    compute shared reward for team members
    
    Args:
        true_hit: [E, 6]
    
    Returns:
        reward: [E, 6]
    '''
    assert len(true_hit.shape) == 2
    assert true_hit.shape[1] == 6

    reward = torch.zeros_like(true_hit)
    
    mask_0_to_2 = true_hit[:, :3].sum(dim=1) > 0
    reward[mask_0_to_2, :3] = 1.0

    mask_3_to_5 = true_hit[:, 3:].sum(dim=1) > 0
    reward[mask_3_to_5, 3:] = 1.0

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
    
    pos[..., :2] = - pos[..., :2]

    q_m = torch.tensor([[0, 0, 0, 1]] * rot.shape[0], device=rot.device)
    rot[:, 0, :] = quaternion_multiply(q_m, rot[:, 0, :])
    rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot)

    vel[..., :2] = - vel[..., :2]

    angular_vel[..., :2] = - angular_vel[..., :2]

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


class Volleyball3v3_hier(IsaacEnv):
    """
    Two teams of drones hit the ball adversarially.
    The net is positioned parallel to the y-axis.

    """

    def __init__(self, cfg: DictConfig, headless: bool):

        self.L: float = cfg.task.court.L
        self.W: float = cfg.task.court.W
        self.H_NET: float = cfg.task.court.H_NET # height of the net
        self.W_NET: float = cfg.task.court.W_NET # not width of the net, but the width of the net's frame
        self.ball_mass: float = cfg.task.get("ball_mass", 0.05)
        self.ball_radius: float = cfg.task.get("ball_radius", 0.1)
        self.symmetric_obs: bool = cfg.task.get("symmetric_obs", True)
        assert self.symmetric_obs, "Only symmetric observation is supported now."

        super().__init__(cfg, headless)

        self.central_env_pos = Float3(*self.envs_positions[self.central_env_idx].tolist())
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
        
        self.near_side_anchor = torch.tensor(cfg.task.anchor, device=self.device) # [3, 3]
        self.far_side_anchor = self.near_side_anchor.clone()
        self.far_side_anchor[:, :2] = - self.far_side_anchor[:, :2]
        self.anchor = torch.cat([self.near_side_anchor, self.far_side_anchor], dim=0).expand(self.num_envs, 6, 3) # [E, 6, 3]

        self.near_side_init_pos = torch.tensor(cfg.task.initial.near_side_pos, device=self.device) # [3, 3]
        self.far_side_init_pos = self.near_side_init_pos.clone()
        self.far_side_init_pos[:, :2] = - self.far_side_init_pos[:, :2]
        
        self.serve_near_side_pos = torch.tensor(cfg.task.initial.serve_near_side_pos, device=self.device) # [3, 3]
        self.init_serve_pos = torch.cat([self.serve_near_side_pos, self.far_side_init_pos], dim=0).expand(self.num_envs, 6, 3) # [E, 6, 3]
        self.defend_near_side_pos = torch.tensor(cfg.task.initial.defend_near_side_pos, device=self.device) # [3, 3]
        self.init_defend_pos = torch.cat([self.defend_near_side_pos, self.far_side_init_pos], dim=0).expand(self.num_envs, 6, 3) # [E, 6, 3]

        self.init_ball_offset = torch.tensor(cfg.task.initial.ball_offset, device=self.device)
        self.init_on_the_spot = cfg.task.initial.get("init_on_the_spot", True)
        assert self.init_on_the_spot, "Only init_on_the_spot is supported now."

        # (n_envs,) 0/1
        self.which_side = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_hit_team = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        self.last_hit_step = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.int64)
        self.ball_last_vel = torch.zeros(self.num_envs, 1, 3, device=self.device, dtype=torch.float32)

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.random_turn = cfg.task.get("random_turn", True)
        
        # one-hot id [E,3,3]
        self.id = torch.zeros((cfg.task.env.num_envs, 3, 3), device=self.device)
        self.id[:, 0, 0] = 1
        self.id[:, 1, 1] = 1
        self.id[:, 2, 2] = 1

        self.racket_near_ball = torch.zeros((cfg.task.env.num_envs, 6), device=self.device, dtype=torch.bool)
        self.drone_near_ball = torch.zeros((cfg.task.env.num_envs, 6), device=self.device, dtype=torch.bool)
        self.init_rpy = torch.zeros((cfg.task.env.num_envs, 6, 3), device=self.device)
        self.init_rpy[:, 3:, 2] += 3.1415926

        self.drones_already_hit_in_one_turn = torch.zeros((cfg.task.env.num_envs, 6), device=self.device, dtype=torch.bool)

        self.not_reset_keys_in_stats = ["actor_0_wins", "actor_1_wins", "draws", "terminated", "truncated", "done"]

        self.ball_vel_x_dir_world = torch.ones((self.num_envs, 6), device=self.device)
        self.ball_vel_x_dir_world[:, :3] = -1
        
        # settings
        self.drones_pos_or_rpos_obs = True # True: pos, False: rpos
        self.shared_or_individual_rpos_reward = False # True: shared, False: individual

        logging.info(f'''Settings:
                        drones_pos_or_rpos_obs: {self.drones_pos_or_rpos_obs}, 
                        shared_or_individual_rpos_reward: {self.shared_or_individual_rpos_reward}''')
        
        self.FirstPass_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: goto; 1: firstpass; 2: hover; 3: serve
        self.FirstPass_already_hit = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self.FirstPass_last_hit_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.FirstPass_goto_pos_before_hit = torch.tensor(cfg.task.get("FirstPass_goto_pos_before_hit"), device=self.device).expand(self.num_envs, 1, 3)
        self.FirstPass_hover_pos_after_hit = torch.tensor(cfg.task.get("FirstPass_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)
        #self.FirstPass_serve_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: left; 1: middle; 2: right

        self.SecPass_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: goto; 1: secpass; 3: hover
        self.SecPass_already_hit = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self.SecPass_last_hit_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.SecPass_goto_pos_before_hit = torch.tensor(cfg.task.get("SecPass_goto_pos_before_hit"), device=self.device).expand(self.num_envs, 1, 3)
        self.SecPass_hover_pos_after_hit = torch.tensor(cfg.task.get("SecPass_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)

        self.Att_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: goto; 1: Att; 2: hover
        self.Att_attacking_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # 0: left; 1: right
        self.Att_already_hit = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self.Att_last_hit_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.Att_goto_pos_before_hit = torch.tensor(cfg.task.get("Att_goto_pos_before_hit"), device=self.device).expand(self.num_envs, 1, 3)
        self.Att_hover_pos_after_hit = torch.tensor(cfg.task.get("Att_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)

        hover_target_rpy = torch.zeros((1, 1, 3), device=self.device) # (1, 1, 3)
        hover_target_rot = euler_to_quaternion(hover_target_rpy) # (1, 1, 4)
        self.hover_target_heading = quat_axis(hover_target_rot.squeeze(1), 0).unsqueeze(1).expand(self.num_envs, 1, 3) # (E, 1, 3)
        
        self.FirstPass_serve_ball_rpos = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.hier_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: before Opp hit, 1: FirstPass, 2: SecPass, 3: Att, 4: after Att hit; 5: Serve
        self.switch_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.hier_serve_ball_anchor = torch.zeros((self.num_envs, 1, 3), device=self.device) 

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
        cr_api.CreateThresholdAttr().Set(0.)

        if self.use_local_usd:
            # use local usd resources
            usd_path = os.path.join(os.path.dirname(__file__), os.pardir, "assets", "default_environment.usd")
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                usd_path=usd_path
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
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ]
        )

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]  # 23

        Opp_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 3 + 2 + 18

        FirstPass_serve_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 3
        FirstPass_serve_hover_observation_dim = drone_state_dim + 3
        FirstPass_goto_observation_dim = drone_state_dim + 3
        FirstPass_observation_dim = drone_state_dim + 3 + 3 + 3 + 2
        FirstPass_hover_observation_dim = drone_state_dim + 3
        
        SecPass_goto_observation_dim = drone_state_dim + 3
        SecPass_observation_dim = drone_state_dim + 3 + 3 + 3 + 2
        SecPass_hover_observation_dim = drone_state_dim + 3
        
        Att_goto_observation_dim = drone_state_dim + 3
        Att_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 2
        Att_hover_observation_dim = drone_state_dim + 3

        self.observation_spec = (
            CompositeSpec(
                {
                    "agents": {
                        "Opp_observation": UnboundedContinuousTensorSpec((6, Opp_observation_dim)),
                        
                        "FirstPass_serve_observation": UnboundedContinuousTensorSpec((1, FirstPass_serve_observation_dim)),
                        "FirstPass_serve_hover_observation": UnboundedContinuousTensorSpec((1, FirstPass_serve_hover_observation_dim)),
                        "FirstPass_goto_observation": UnboundedContinuousTensorSpec((1, FirstPass_goto_observation_dim)),
                        "FirstPass_observation": UnboundedContinuousTensorSpec((1, FirstPass_observation_dim)),
                        "FirstPass_hover_observation": UnboundedContinuousTensorSpec((1, FirstPass_hover_observation_dim)),
                        
                        "SecPass_goto_observation": UnboundedContinuousTensorSpec((1, SecPass_goto_observation_dim)),
                        "SecPass_observation": UnboundedContinuousTensorSpec((1, SecPass_observation_dim)),
                        "SecPass_hover_observation": UnboundedContinuousTensorSpec((1, SecPass_hover_observation_dim)),
                        
                        "Att_goto_observation": UnboundedContinuousTensorSpec((1, Att_goto_observation_dim)),
                        "Att_observation": UnboundedContinuousTensorSpec((1, Att_observation_dim)),
                        "Att_hover_observation": UnboundedContinuousTensorSpec((1, Att_hover_observation_dim)),
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
                        "Opp_action": torch.stack([self.drone.action_spec] * (6), dim=0),
                        
                        "FirstPass_serve_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                        "FirstPass_serve_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                        "FirstPass_goto_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                        "FirstPass_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                        "FirstPass_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),

                        "SecPass_goto_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                        "SecPass_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                        "SecPass_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),

                        "Att_goto_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                        "Att_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                        "Att_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                    }
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.reward_spec = (
            CompositeSpec(
                {"agents": {
                    "Opp_reward": UnboundedContinuousTensorSpec((6, 1)),

                    "FirstPass_reward": UnboundedContinuousTensorSpec((1, 1)),
                    "SecPass_reward": UnboundedContinuousTensorSpec((1, 1)),
                    "Att_reward": UnboundedContinuousTensorSpec((1, 1)),
                }
            })
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

        self.agent_spec["Opp"] = AgentSpec(
            "Opp",
            6,
            observation_key=("agents", "Opp_observation"),
            action_key=("agents", "Opp_action"),
            reward_key=("agents", "Opp_reward"),
            state_key=("agents", "Opp_state"),
        )

        self.agent_spec["FirstPass_serve"] = AgentSpec(
            "FirstPass_serve",
            1,
            observation_key=("agents", "FirstPass_serve_observation"),
            action_key=("agents", "FirstPass_serve_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_serve_state"),
        )
        self.agent_spec["FirstPass_serve_hover"] = AgentSpec(
            "FirstPass_serve_hover",
            1,
            observation_key=("agents", "FirstPass_serve_hover_observation"),
            action_key=("agents", "FirstPass_serve_hover_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_serve_hover_state"),
        )
        self.agent_spec["FirstPass_goto"] = AgentSpec(
            "FirstPass_goto",
            1,
            observation_key=("agents", "FirstPass_goto_observation"),
            action_key=("agents", "FirstPass_goto_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_goto_state"),
        )
        self.agent_spec["FirstPass"] = AgentSpec(
            "FirstPass",
            1,
            observation_key=("agents", "FirstPass_observation"),
            action_key=("agents", "FirstPass_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_state"),
        )
        self.agent_spec["FirstPass_hover"] = AgentSpec(
            "FirstPass_hover",
            1,
            observation_key=("agents", "FirstPass_hover_observation"),
            action_key=("agents", "FirstPass_hover_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_hover_state"),
        )

        self.agent_spec["SecPass_goto"] = AgentSpec(
            "SecPass_goto",
            1,
            observation_key=("agents", "SecPass_goto_observation"),
            action_key=("agents", "SecPass_goto_action"),
            reward_key=("agents", "SecPass_reward"),
            state_key=("agents", "SecPass_goto_state"),
        )
        self.agent_spec["SecPass"] = AgentSpec(
            "SecPass",
            1,
            observation_key=("agents", "SecPass_observation"),
            action_key=("agents", "SecPass_action"),
            reward_key=("agents", "SecPass_reward"),
            state_key=("agents", "SecPass_state"),
        )
        self.agent_spec["SecPass_hover"] = AgentSpec(
            "SecPass_hover",
            1,
            observation_key=("agents", "SecPass_hover_observation"),
            action_key=("agents", "SecPass_hover_action"),
            reward_key=("agents", "SecPass_reward"),
            state_key=("agents", "SecPass_hover_state"),
        )

        self.agent_spec["Att_goto"] = AgentSpec(
            "Att_goto",
            1,
            observation_key=("agents", "Att_goto_observation"),
            action_key=("agents", "Att_goto_action"),
            reward_key=("agents", "Att_reward"),
            state_key=("agents", "Att_goto_state"),
        )
        self.agent_spec["Att"] = AgentSpec(
            "Att",
            1,
            observation_key=("agents", "Att_observation"),
            action_key=("agents", "Att_action"),
            reward_key=("agents", "Att_reward"),
            state_key=("agents", "Att_state"),
        )
        self.agent_spec["Att_hover"] = AgentSpec(
            "Att_hover",
            1,
            observation_key=("agents", "Att_hover_observation"),
            action_key=("agents", "Att_hover_action"),
            reward_key=("agents", "Att_reward"),
            state_key=("agents", "Att_hover_state"),
        )

        _stats_spec = CompositeSpec(
            {
                "episode_len": UnboundedContinuousTensorSpec(1),
                "num_turns": UnboundedContinuousTensorSpec(1),

                "actor_0_wins": UnboundedContinuousTensorSpec(1),
                "actor_1_wins": UnboundedContinuousTensorSpec(1),
                "draws": UnboundedContinuousTensorSpec(1),
                
                "terminated": UnboundedContinuousTensorSpec(1),
                "truncated": UnboundedContinuousTensorSpec(1),
                "done": UnboundedContinuousTensorSpec(1),
            }
        )
        self.stats_cfg: DictConfig = self.cfg.task.stats
            
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

        color = [(0., 1., 0., 1.)]
        # [topleft, topright, botleft, botright]
        
        points_start, points_end = rectangular_cuboid_edges(self.L, self.W, 0.0, 3.0)
        points_start = [_carb_float3_add(p, self.central_env_pos) for p in points_start]
        points_end = [_carb_float3_add(p, self.central_env_pos) for p in points_end]
        
        colors_line = color * len(points_start)
        sizes_line = [1.] * len(points_start)
        self.draw.draw_lines(points_start,
                                 points_end, colors_line, sizes_line)

    def debug_draw_turn(self):
        turn = self.which_side[self.central_env_idx]
        ori = self.envs_positions[self.central_env_idx].detach()

        points_near_side = torch.tensor([2., 0., 0.]).to(self.device) + ori
        points_far_side = torch.tensor([-2., 0., 0.]).to(self.device) + ori

        colors = [(0, 1, 0, 1)]
        sizes = [20.]
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
        win_near_side = torch.tensor([5., 0., 0.]).to(self.device) + ori
        win_far_side = torch.tensor([-5., 0., 0.]).to(self.device) + ori

        colors = [(1, 1, 0, 1)]
        sizes = [20.]

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

        if self.random_turn:
            turn = torch.randint(0, 2, (len(env_ids),), device=self.device) # randomly choose the first team
        else:
            turn = torch.zeros(len(env_ids), dtype=torch.long, device=self.device) # always team 0 starts
            # turn = torch.ones(len(env_ids), dtype=torch.long, device=self.device) # always team 1 starts
        self.which_side[env_ids] = turn
        self.last_hit_team[env_ids] = turn # init

        self.hier_turn[env_ids] = torch.where(
            turn.bool(),
            0, # All hover
            5, # serve
        )
        self.FirstPass_turn[env_ids] = torch.where(
            turn.bool(),
            0, # goto
            3, # serve
        )
        self.SecPass_turn[env_ids] = torch.zeros(len(env_ids), device=self.device, dtype=torch.long) # goto
        self.Att_turn[env_ids] = torch.zeros(len(env_ids), device=self.device, dtype=torch.long) # goto

        drone_pos = torch.where(
            turn.unsqueeze(1).unsqueeze(2).bool(),
            self.init_defend_pos[env_ids],
            self.init_serve_pos[env_ids],
        )
        drone_rpy = torch.zeros((len(env_ids), 6, 3), device=self.device)
        drone_rpy[:, 3:, 2] += 3.1415926

        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos +
            self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(torch.zeros((len(env_ids), 6, 6), device=self.device), env_ids)        

        self.drones_already_hit_in_one_turn[env_ids] = True
        self.drones_already_hit_in_one_turn[env_ids, 3 * turn + 2] = False

        ball_pos = drone_pos[torch.arange(len(env_ids)), 3 * turn + 2, :] + self.init_ball_offset
        ball_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        self.ball.set_world_poses(ball_pos + self.envs_positions[env_ids], ball_rot, env_ids)
        
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
            point_list_1, point_list_2, colors, sizes = draw_court(self.W, self.L, self.H_NET, self.W_NET, n=2)
            point_list_1 = [_carb_float3_add(p, self.central_env_pos) for p in point_list_1]
            point_list_2 = [_carb_float3_add(p, self.central_env_pos) for p in point_list_2]
            self.draw.draw_lines(point_list_1, point_list_2, colors, sizes)
            self.debug_draw_region()
            # self.debug_draw_turn()

        self.last_hit_step[env_ids] = -100

        self.racket_near_ball[env_ids] = False
        self.drone_near_ball[env_ids] = False
        self.switch_turn[env_ids] = False

        self.Att_attacking_target[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device, dtype=torch.bool)
        #self.FirstPass_serve_target[env_ids] = 2 * torch.randint(0, 2, (len(env_ids),), device=self.device, dtype=torch.long) # 0: left; 2: right

        self.hier_serve_ball_anchor[env_ids, 0, 0] = torch.rand(len(env_ids), device=self.device) * 1.8 + 4.2  # x: [5, 6]
        self.hier_serve_ball_anchor[env_ids, 0, 1] = torch.rand(len(env_ids), device=self.device) * 4 - 2  # y: [-0.5, 0.5]
        self.hier_serve_ball_anchor[env_ids, 0, 2] = 0  # z: 1

        # some stats keys will not reset
        stats_list = []
        for i in self.not_reset_keys_in_stats:
            stats_list.append(self.stats[i][env_ids].clone())
        self.stats[env_ids] = 0.0
        for i, key in enumerate(self.not_reset_keys_in_stats):
            self.stats[key][env_ids] = stats_list[i]

        self.draw.clear_points()
        # self.debug_draw_turn()

    def _pre_sim_step(self, tensordict: TensorDictBase):
        Opp_action: torch.Tensor = tensordict[("agents", "Opp_action")]

        FirstPass_serve_action = tensordict[("agents", "FirstPass_serve_action")]
        FirstPass_serve_hover_action = tensordict[("agents", "FirstPass_serve_hover_action")]
        FirstPass_goto_action = tensordict[("agents", "FirstPass_goto_action")]
        FirstPass_action = tensordict[("agents", "FirstPass_action")]
        FirstPass_hover_action = tensordict[("agents", "FirstPass_hover_action")]
        FirstPass_real_action = torch.where(
            (self.hier_turn == 1).unsqueeze(-1).unsqueeze(-1).bool(),
            FirstPass_action,
            torch.where(
                (self.hier_turn == 5).unsqueeze(-1).unsqueeze(-1).bool(),
                FirstPass_serve_action,
                torch.where(
                    (self.hier_turn == 0).unsqueeze(-1).unsqueeze(-1).bool(),
                    FirstPass_goto_action,
                    torch.where(
                            (self.hier_turn == 6).unsqueeze(-1).unsqueeze(-1).bool(),
                            FirstPass_serve_hover_action,                
                            FirstPass_hover_action
                        )
                )
            )
        )

        SecPass_goto_action = tensordict[("agents", "SecPass_goto_action")]
        SecPass_action = tensordict[("agents", "SecPass_action")]
        SecPass_hover_action = tensordict[("agents", "SecPass_hover_action")]
        SecPass_real_action = torch.where(
            (self.hier_turn == 2).unsqueeze(-1).unsqueeze(-1).bool(),
            SecPass_action,
            torch.where(
                ((self.hier_turn == 3) | (self.hier_turn == 4)).unsqueeze(-1).unsqueeze(-1).bool(),
                SecPass_hover_action,
                SecPass_goto_action
            )
        )

        Att_goto_action = tensordict[("agents", "Att_goto_action")]
        Att_action = tensordict[("agents", "Att_action")]
        Att_hover_action = tensordict[("agents", "Att_hover_action")]
        Att_real_action = torch.where(
            (self.hier_turn == 3).unsqueeze(-1).unsqueeze(-1).bool(),
            Att_action,
            torch.where(
                (self.hier_turn == 4).unsqueeze(-1).unsqueeze(-1).bool(),
                Att_hover_action,
                Att_goto_action
            )
        )

        actions = torch.cat(
            [
                SecPass_real_action,
                Att_real_action,
                FirstPass_real_action,
                Opp_action[:, 3:, :],
            ],
            dim=1
        )

        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13] 

        self.ball_pos, _ = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()[..., :3]

        self.rpos = self.ball_pos - self.drone.pos # (E, 6, 3)
        FirstPass_rpos = self.rpos[:, 2, :].unsqueeze(1)
        SecPass_rpos = self.rpos[:, 0, :].unsqueeze(1)
        Att_rpos = self.rpos[:, 1, :].unsqueeze(1)

        hier_serve_ball_anchor_transfer = self.hier_serve_ball_anchor.clone()
        hier_serve_ball_anchor_transfer[..., :2] = - hier_serve_ball_anchor_transfer[..., :2]        
        self.FirstPass_serve_ball_rpos = self.ball_pos - hier_serve_ball_anchor_transfer # (E, 1, 3)
        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )

        team_0_pos_obs = pos.clone() # (E, 6, 3)
        drones_rpos_0 = (team_0_pos_obs - team_0_pos_obs[:, 0, :].unsqueeze(1)).reshape(-1, 18)
        drones_rpos_1 = (team_0_pos_obs - team_0_pos_obs[:, 1, :].unsqueeze(1)).reshape(-1, 18)
        drones_rpos_2 = (team_0_pos_obs - team_0_pos_obs[:, 2, :].unsqueeze(1)).reshape(-1, 18)

        team_1_pos_obs = pos.clone()[:, [3, 4, 5, 0, 1, 2], :]
        team_1_pos_obs[..., :2] = -team_1_pos_obs[..., :2] # symmetric pos
        drones_rpos_3 = (team_1_pos_obs - team_1_pos_obs[:, 0, :].unsqueeze(1)).reshape(-1, 18)
        drones_rpos_4 = (team_1_pos_obs - team_1_pos_obs[:, 1, :].unsqueeze(1)).reshape(-1, 18)
        drones_rpos_5 = (team_1_pos_obs - team_1_pos_obs[:, 2, :].unsqueeze(1)).reshape(-1, 18)

        # rpos
        sym_drones_rpos_obs = torch.stack([drones_rpos_0, drones_rpos_1, drones_rpos_2, drones_rpos_3, drones_rpos_4, drones_rpos_5], dim=1) # [E,6,18]

        # pos
        sym_drones_pos_obs = torch.stack([
                team_0_pos_obs.reshape(-1, 18), 
                team_0_pos_obs.reshape(-1, 18), 
                team_0_pos_obs.reshape(-1, 18), 
                team_1_pos_obs.reshape(-1, 18),
                team_1_pos_obs.reshape(-1, 18),
                team_1_pos_obs.reshape(-1, 18),
            ], dim=1) # [E,6,18]
        
        if self.drones_pos_or_rpos_obs:
            drones_obs = sym_drones_pos_obs
        else:
            drones_obs = sym_drones_rpos_obs
        
        rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot) # make sure w of (w,x,y,z) is positive
        self.root_state = torch.cat([pos, rot, vel, angular_vel, heading, up, throttle], dim=-1)

        self.rpy = quaternion_to_euler(rot)

        self.drone_rot = rot # world frame

        # root_state:
        FirstPass_root_state = self.root_state[:, 2, :].unsqueeze(1)
        SecPass_root_state = self.root_state[:, 0, :].unsqueeze(1)
        Att_root_state = self.root_state[:, 1, :].unsqueeze(1)

        near_side_root_state = self.root_state[:, :3, :]
        far_side_root_state = torch.cat(
            [
                transfer_root_state_to_the_other_side(self.root_state[:, 3, :].unsqueeze(1)),
                transfer_root_state_to_the_other_side(self.root_state[:, 4, :].unsqueeze(1)),
                transfer_root_state_to_the_other_side(self.root_state[:, 5, :].unsqueeze(1)),
            ],
            dim=1,
        )
        sym_root_state = torch.cat([near_side_root_state, far_side_root_state], dim=1)
        sym_pos, sym_rot, sym_vel, sym_angular_vel, sym_heading, sym_up, sym_throttle = torch.split(
            sym_root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )

        # relative position of the ball to the drone
        self.rpos_ball = self.ball_pos - self.drone.pos  # (E,6,3) world frame
        sym_rpos_ball = self.rpos_ball.clone()
        sym_rpos_ball[:, 3:, :2] = - sym_rpos_ball[:, 3:, :2]

        # relative position of the drone to the anchor
        sym_rpos_anchor = self.drone.pos - self.anchor  # (E,6,3)
        sym_rpos_anchor[:, 3:, :2] = - sym_rpos_anchor[:, 3:, :2]

        sym_ball_vel = self.ball_vel.clone().repeat(1, 6, 1)
        sym_ball_vel[:, 3:, :2] = - sym_ball_vel[:, 3:, :2]

        already_hit_in_one_turn_to_obs = torch.nn.functional.one_hot(self.drones_already_hit_in_one_turn.long(), num_classes=2)

        sym_obs = [
            sym_pos, # (E,6,3)
            sym_rot, # (E,6,4)
            sym_vel, # (E,6,3)
            sym_angular_vel, # (E,6,3)
            sym_heading, # (E,6,3)
            sym_up, # (E,6,3)
            sym_throttle, # (E,6,4)
            sym_rpos_anchor,  # [E,6,3]
            sym_rpos_ball,  # [E,6,3]
            sym_ball_vel,  # [E,6,3]
            turn_to_obs(self.which_side), # [E,6,2]
            torch.cat((self.id, self.id), dim=1), # [E,6,3]
            already_hit_in_one_turn_to_obs, # [E,6,2]
            drones_obs, # [E,6,18]
        ]
        sym_obs = torch.cat(sym_obs, dim=-1) # [E,6,obs_dim]

        Opp_obs = sym_obs

        # FirstPass_serve_obs
        FirstPass_serve_obs = [
            FirstPass_root_state,
            self.ball_pos, # [E, 1, 3]
            FirstPass_rpos, # [E, 1, 3]
            self.ball_vel[..., :3], # [E, 1, 3]
            serve_turn_to_obs(self.FirstPass_turn), # [E, 1, 2]
            self.FirstPass_serve_ball_rpos,
        ] # obs_dim: root_state + rpos(3) + ball_pos(3) + ball_vel(3) + turn(2) + start_point(3)
        FirstPass_serve_obs = torch.cat(FirstPass_serve_obs, dim=-1)

        # FirstPass_serve_hover_obs
        FirstPass_serve_hover_rpos = self.FirstPass_hover_pos_after_hit -  FirstPass_root_state[..., :3] # (E, 1, 3)
        FirstPass_serve_hover_rheading = self.hover_target_heading - FirstPass_root_state[..., 13:16] # (E, 1, 3)
        FirstPass_serve_hover_obs = [
            FirstPass_serve_hover_rpos, # (E, 1, 3)
            FirstPass_root_state[..., 3:], # (E, 1, root_state_dim - 3)
            FirstPass_serve_hover_rheading, # (E, 1, 3)
        ]
        FirstPass_serve_hover_obs = torch.cat(FirstPass_serve_hover_obs, dim=-1)

        # FirstPass_goto_obs
        FirstPass_goto_rpos = self.FirstPass_goto_pos_before_hit - FirstPass_root_state[..., :3] # (E, 1, 3)
        FirstPass_goto_rheading = self.hover_target_heading - FirstPass_root_state[..., 13:16] # (E, 1, 3)
        FirstPass_goto_obs = [
            FirstPass_goto_rpos, # (E, 1, 3)
            FirstPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            FirstPass_goto_rheading, # (E, 1, 3)
        ]
        FirstPass_goto_obs = torch.cat(FirstPass_goto_obs, dim=-1)
        
        # FirstPass_obs
        FirstPass_obs = [
            FirstPass_root_state, # (E, 1, 23)
            self.ball_pos, # (E, 1, 3)
            FirstPass_rpos, # (E, 1, 3)
            self.ball_vel, # (E, 1, 3)
            individual_turn_to_obs(self.FirstPass_turn), # (E, 1, 2)
        ]
        FirstPass_obs = torch.cat(FirstPass_obs, dim=-1)

        # FirstPass_hover_obs
        FirstPass_hover_rpos = self.FirstPass_hover_pos_after_hit - FirstPass_root_state[..., :3] # (E, 1, 3)
        FirstPass_hover_rheading = self.hover_target_heading - FirstPass_root_state[..., 13:16] # (E, 1, 3)
        FirstPass_hover_obs = [
            FirstPass_hover_rpos, # (E, 1, 3)
            FirstPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            FirstPass_hover_rheading, # (E, 1, 3)
        ]
        FirstPass_hover_obs = torch.cat(FirstPass_hover_obs, dim=-1)

        # SecPass_goto_obs
        SecPass_goto_rpos = self.SecPass_goto_pos_before_hit - SecPass_root_state[..., :3] # (E, 1, 3)
        SecPass_goto_rheading = self.hover_target_heading - SecPass_root_state[..., 13:16] # (E, 1, 3)
        SecPass_goto_obs = [
            SecPass_goto_rpos, # (E, 1, 3)
            SecPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            SecPass_goto_rheading, # (E, 1, 3)
        ]
        SecPass_goto_obs = torch.cat(SecPass_goto_obs, dim=-1)

        # SecPass_obs
        SecPass_obs = [
            SecPass_root_state, # (E, 1, 23)
            self.ball_pos, # (E, 1, 3)
            SecPass_rpos, # (E, 1, 3)
            self.ball_vel, # (E, 1, 3)
            individual_turn_to_obs(self.SecPass_turn), # (E, 1, 2)
        ]
        SecPass_obs = torch.cat(SecPass_obs, dim=-1)

        # SecPass_hover_obs
        SecPass_hover_rpos = self.SecPass_hover_pos_after_hit - SecPass_root_state[..., :3] # (E, 1, 3)
        SecPass_hover_rheading = self.hover_target_heading - SecPass_root_state[..., 13:16] # (E, 1, 3)
        SecPass_hover_obs = [
            SecPass_hover_rpos, # (E, 1, 3)
            SecPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            SecPass_hover_rheading, # (E, 1, 3)
        ]
        SecPass_hover_obs = torch.cat(SecPass_hover_obs, dim=-1)

        # Att_goto_obs
        Att_goto_rpos = self.Att_goto_pos_before_hit - Att_root_state[..., :3] # (E, 1, 3)
        Att_goto_rheading = self.hover_target_heading - Att_root_state[..., 13:16] # (E, 1, 3)
        Att_goto_obs = [
            Att_goto_rpos, # (E, 1, 3)
            Att_root_state[..., 3:], # (E, 1, root_state_dim-3)
            Att_goto_rheading, # (E, 1, 3)
        ]
        Att_goto_obs = torch.cat(Att_goto_obs, dim=-1)

        # Att_obs
        Att_obs = [
            Att_root_state, # (E, 1, 23)
            self.ball_pos, # (E, 1, 3)
            Att_rpos, # (E, 1, 3)
            self.ball_vel, # (E, 1, 3)
            individual_turn_to_obs(self.Att_turn), # (E, 1, 2)
            attacking_target_to_obs(self.Att_attacking_target)
        ]
        Att_obs = torch.cat(Att_obs, dim=-1)

        # Att_hover_obs
        Att_hover_rpos = self.Att_hover_pos_after_hit - Att_root_state[..., :3] # (E, 1, 3)
        Att_hover_rheading = self.hover_target_heading - Att_root_state[..., 13:16] # (E, 1, 3)
        Att_hover_obs = [
            Att_hover_rpos, # (E, 1, 3)
            Att_root_state[..., 3:], # (E, 1, root_state_dim-3)
            Att_hover_rheading, # (E, 1, 3)
        ]
        Att_hover_obs = torch.cat(Att_hover_obs, dim=-1)


        return TensorDict(
            {
                "agents": {
                    "Opp_observation": Opp_obs,

                    "FirstPass_serve_observation": FirstPass_serve_obs,
                    "FirstPass_serve_hover_observation": FirstPass_serve_hover_obs,
                    "FirstPass_goto_observation": FirstPass_goto_obs,
                    "FirstPass_observation": FirstPass_obs,
                    "FirstPass_hover_observation": FirstPass_hover_obs,

                    "SecPass_goto_observation": SecPass_goto_obs,
                    "SecPass_observation": SecPass_obs,
                    "SecPass_hover_observation": SecPass_hover_obs,

                    "Att_goto_observation": Att_goto_obs,
                    "Att_observation": Att_obs,
                    "Att_hover_observation": Att_hover_obs,
                },
                "stats": self.stats, # 
                "info": self.info,
            },
            self.num_envs,
        )

    def check_ball_near_racket(self, racket_radius, cylinder_height_coeff):
        z_direction_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        z_direction_world = quat_rotate(self.drone_rot, z_direction_local) # (E,N,3)
        normal_vector_world = z_direction_world / torch.norm(z_direction_world, dim=-1).unsqueeze(-1)  # (E,N,3)
        cylinder_bottom_center = self.drone.pos  # (E,N,3) cylinder bottom center
        cylinder_axis = cylinder_height_coeff * self.ball_radius * normal_vector_world 

        ball_to_bottom = self.ball_pos - cylinder_bottom_center  # (E,N,3)
        projection_ratio = torch.sum(ball_to_bottom * cylinder_axis, dim=-1) / torch.sum(cylinder_axis * cylinder_axis, dim=-1)  # (E,N) projection of ball_to_bottom on cylinder_axis / cylinder_axis
        within_height = (projection_ratio >= 0) & (projection_ratio <= 1)  # (E,N)

        projection_point = cylinder_bottom_center + projection_ratio.unsqueeze(-1) * cylinder_axis  # (E,N,3)
        distance_to_axis = torch.norm(self.ball_pos - projection_point, dim=-1)  # (E,N)
        within_radius = distance_to_axis <= racket_radius  # (E,N)

        return (within_height & within_radius)  # (E,N)
    
    def check_hit(self, sim_dt, racket_radius=0.2, cylinder_height_coeff=2.0):
        racket_near_ball_last_step = self.racket_near_ball.clone()
        drone_near_ball_last_step = self.drone_near_ball.clone()

        self.racket_near_ball = self.check_ball_near_racket(racket_radius=racket_radius, cylinder_height_coeff=cylinder_height_coeff)  # (E,N)
        self.drone_near_ball = (torch.norm(self.rpos_ball, dim=-1) < 0.5) # (E,N)

        ball_vel_z_change = ((self.ball_vel[..., 2] - self.ball_last_vel[..., 2]) > 9.8 * sim_dt) # (E,1)
        ball_vel_x_y_change = (self.ball_vel[..., :2] - self.ball_last_vel[..., :2]).norm(dim=-1) > 0.5 # (E,1)
        ball_vel_change = ball_vel_z_change | ball_vel_x_y_change # (E,1)
        
        drone_hit_ball = (drone_near_ball_last_step | self.drone_near_ball) & ball_vel_change # (E,N)
        racket_hit_ball = (racket_near_ball_last_step | self.racket_near_ball) & ball_vel_change # (E,N)

        racket_hit_ball = racket_hit_ball & (self.progress_buf.unsqueeze(-1) - self.last_hit_step > 3) # (E,N)
        drone_hit_ball = drone_hit_ball & (self.progress_buf.unsqueeze(-1) - self.last_hit_step > 3) # (E,N)

        return racket_hit_ball, drone_hit_ball

    def log_results(self, draw, game_result, cases, drone_hit_ground, drone_too_close, env_id=0):
        assert draw.shape == torch.Size([self.num_envs, 1])
        assert game_result.shape == torch.Size([self.num_envs, 1])
        assert cases.shape == torch.Size([self.num_envs, 2, 6])
        assert drone_hit_ground.shape == torch.Size([self.num_envs, 6])
        assert drone_too_close.shape == torch.Size([self.num_envs, 6])

        if draw[env_id, 0].item():
            log_str = "Draw! "
        elif game_result[env_id, 0].item() == 1:
            log_str = "Team 0 win! "
        elif game_result[env_id, 0].item() == -1:
            log_str = "Team 1 win! "
        elif drone_hit_ground[env_id, :3].any(-1).item() == 1:
            log_str = "Team 0 drone hit the ground! "
        elif drone_hit_ground[env_id, 3:].any(-1).item() == 1:
            log_str = "Team 1 drone hit the ground! "
        elif drone_too_close[env_id, :3].any(-1).item() == 1:
            log_str = "Team 0 drones are too close to each other! "
        elif drone_too_close[env_id, 3:].any(-1).item() == 1:
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
                self.stats[f"team_{team}_case_{case+1}"] = (cases[:, team, case].unsqueeze(-1).float())

    def _compute_drone_too_close(self, threshold=0.8):
        drone_pos = self.drone.pos # (E, 6, 3)
        
        # team 0
        drone_pos_0 = drone_pos[:, 0, :] # (E, 3)
        drone_pos_1 = drone_pos[:, 1, :] # (E, 3)
        drone_pos_2 = drone_pos[:, 2, :] # (E, 3)
        too_close_01 = (torch.norm(drone_pos_0 - drone_pos_1, dim=-1) < threshold) # (E,)
        too_close_02 = (torch.norm(drone_pos_0 - drone_pos_2, dim=-1) < threshold) # (E,)
        too_close_12 = (torch.norm(drone_pos_1 - drone_pos_2, dim=-1) < threshold) # (E,)
        team_0_too_close = torch.stack([
                (too_close_01 | too_close_02),
                (too_close_01 | too_close_12),
                (too_close_02 | too_close_12),
            ], dim=1
        ) # (E, 3)

        # team 1
        drone_pos_3 = drone_pos[:, 3, :] # (E, 3)
        drone_pos_4 = drone_pos[:, 4, :] # (E, 3)
        drone_pos_5 = drone_pos[:, 5, :] # (E, 3)
        too_close_34 = (torch.norm(drone_pos_3 - drone_pos_4, dim=-1) < threshold) # (E,)
        too_close_35 = (torch.norm(drone_pos_3 - drone_pos_5, dim=-1) < threshold) # (E,)
        too_close_45 = (torch.norm(drone_pos_4 - drone_pos_5, dim=-1) < threshold) # (E,)
        team_1_too_close = torch.stack([
                (too_close_34 | too_close_35),
                (too_close_34 | too_close_45),
                (too_close_35 | too_close_45),
            ], dim=1
        ) # (E, 3)

        return torch.cat([team_0_too_close, team_1_too_close], dim=1) # (E, 6)
    

    def _rpos_to_dec_reward(self) -> torch.Tensor:
        '''
        compute decentralized reward for team members

        Returns:
            reward: [E, 6]
        
        '''
        rpos = self.drone.pos - self.ball_pos
        
        effective_dist = (1 - self.drones_already_hit_in_one_turn.float()) * rpos.norm(p=2, dim=-1) # [E, 6]
        effective_dist[effective_dist == 0] = float('inf')

        return effective_dist # [E, 6]
    

    def _rpos_to_shared_reward(self) -> torch.Tensor:
        '''
        compute shared reward for team members

        Returns:
            reward: [E, 2]
        
        '''
        rpos = self.drone.pos - self.ball_pos
        
        effective_dist = (1 - self.drones_already_hit_in_one_turn.float()) * rpos.norm(p=2, dim=-1) # [E, 6]
        effective_dist[effective_dist == 0] = float('inf')

        team_0_dist_min = effective_dist[:, :3].min(dim=1).values # [E]
        team_1_dist_min = effective_dist[:, 3:].min(dim=1).values # [E]

        return torch.stack([team_0_dist_min, team_1_dist_min], dim=1) # [E, 2]

    
    def _compute_reward_and_done(self):
        
        Opp_reward = torch.zeros((self.num_envs, 6), device=self.device)
        FirstPass_reward = torch.zeros((self.num_envs, 1), device=self.device)
        SecPass_reward = torch.zeros((self.num_envs, 1), device=self.device)
        Att_reward = torch.zeros((self.num_envs, 1), device=self.device)

        racket_hit_ball, drone_hit_ball = self.check_hit(sim_dt=self.dt) # (E, 6)
        wrong_hit_racket = drone_hit_ball & ~racket_hit_ball # (E, 6) 

        true_hit = racket_hit_ball & ~self.drones_already_hit_in_one_turn  # (E, 6)
        wrong_hit_turn = racket_hit_ball & self.drones_already_hit_in_one_turn

        Opp_hit = (true_hit[:, 3:].any(-1) | wrong_hit_racket[:, 3:].any(-1)) # (E,) 
        FirstPass_hit = (true_hit[:, 2] | wrong_hit_racket[:, 2]) # (E,) 
        SecPass_hit = (true_hit[:, 0] | wrong_hit_racket[:, 0]) # (E,) 
        Att_hit = (true_hit[:, 1] | wrong_hit_racket[:, 1]) # (E,) 

        self.drones_already_hit_in_one_turn = self.drones_already_hit_in_one_turn | racket_hit_ball

        self.last_hit_team = torch.where(true_hit[:, :3].any(-1), 0, self.last_hit_team) # (E,)
        self.last_hit_team = torch.where(true_hit[:, 3:].any(-1), 1, self.last_hit_team) # (E,)
        
        drone_hit_ground = self.drone.pos[..., 2] < 0.3 # (E, 6) 
        drone_too_close = self._compute_drone_too_close() # (E, 6) 

        self.ball_last_vel = self.ball_vel.clone()
        self.last_hit_step[racket_hit_ball] = self.progress_buf[racket_hit_ball.any(-1)].long()

        # switch turn
        last_which_side = self.which_side.clone()
        self.which_side = torch.where(self.ball_pos[..., 0].squeeze(-1) > 0, 0, 1) # update new turn
        switch_side = torch.where((self.which_side ^ last_which_side).bool(), True, False) # (E,)

        switch_side_idx = torch.nonzero(switch_side, as_tuple=True)[0]
        self.drones_already_hit_in_one_turn[switch_side_idx] = torch.where(
            self.which_side[switch_side_idx].unsqueeze(-1) == 0,
            torch.tensor([False, False, False, True, True, True], device=self.device).unsqueeze(0),
            torch.tensor([True, True, True, False, False, False], device=self.device).unsqueeze(0),
        )

        draw, game_result, cases = determine_game_result_3v3(
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

        self.log_results(draw, game_result, cases, drone_hit_ground, drone_too_close, self.central_env_idx)

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        # terminated = (game_result != 0) | draw | drone_hit_ground.any(-1, keepdim=True) | drone_too_close.any(-1, keepdim=True)
        terminated = (game_result != 0)
        ball_hit_ground = (self.ball_pos[..., 2] < 0.2)
        # terminated = ball_hit_ground
        done = truncated | terminated # [E, 1]

        # change turn
        last_turn = self.hier_turn.clone() # (E,)

        self.hier_turn = torch.where((self.hier_turn == 0) & Opp_hit, 1, self.hier_turn) # (E,)
        self.hier_turn = torch.where((self.hier_turn == 1) & FirstPass_hit, 2, self.hier_turn) # (E,)
        self.hier_turn = torch.where((self.hier_turn == 2) & SecPass_hit, 3, self.hier_turn) # (E,)
        self.hier_turn = torch.where((self.hier_turn == 3) & Att_hit, 4, self.hier_turn) # (E,)

        self.FirstPass_turn = torch.where((self.hier_turn == 1), 1, self.FirstPass_turn) # (E,)
        self.SecPass_turn = torch.where((self.hier_turn == 2), 1, self.SecPass_turn) # (E,)
        self.Att_turn = torch.where((self.hier_turn == 3), 1, self.Att_turn) # (E,)
        
        self.hier_turn = torch.where((self.hier_turn == 5) & FirstPass_hit, 6, self.hier_turn) # (E,)

        self.switch_turn = torch.where((self.hier_turn ^ last_turn).bool(), True, False) # (E,)

        # if self.switch_turn[self.central_env_idx]:
        #     # self.debug_draw_turn()

        self.stats["actor_0_wins"] = (game_result == 1).float()
        self.stats["actor_1_wins"] = (game_result == -1).float()
        self.stats["draws"] = done.float() - (game_result != 0).float()

        self.stats["truncated"][:] = truncated.float()
        self.stats["terminated"][:] = terminated.float()
        self.stats["done"][:] = done.float()
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["num_turns"].add_(switch_side.float().unsqueeze(-1))
        
        return TensorDict(
            {
                "agents": {
                    "Opp_reward": Opp_reward,
                    "FirstPass_reward": FirstPass_reward,
                    "SecPass_reward": SecPass_reward,
                    "Att_reward": Att_reward,
                },
                "stats": self.stats, #
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
    
    def debug_draw_turn(self):

        turn = self.hier_turn[self.central_env_idx]
        
        ori = self.envs_positions[self.central_env_idx].detach()
        
        points_turn = torch.tensor([2., 0., 0.]).to(self.device) + ori
        points_turn = [points_turn.tolist()]

        if turn.item() == 0:
            colors_turn = [(1, 0, 0, 1)]
        elif turn.item() == 1:
            colors_turn = [(0, 1, 0, 1)]
        elif turn.item() == 2:
            colors_turn = [(0, 0, 1, 1)]
        elif turn.item() == 3:
            colors_turn = [(1, 1, 0, 1)]
        elif turn.item() == 4:
            colors_turn = [(0.5, 0, 1, 1)]
        elif turn.item() == 5:
            colors_turn = [(0.5, 1, 0, 1)]
        
        sizes = [15.]

        self.draw.draw_points(points_turn, colors_turn, sizes)

def attacking_target_to_obs(t: torch.Tensor, Att_or_FirstPass=True) -> torch.Tensor:
    """convert representation of ball target to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)
        Att_or_FirstPass (bool):
            True: Att
            False: FirstPass

    Returns:
        torch.Tensor: (n_env, 1, 2)
    """
    if Att_or_FirstPass:
        table = torch.tensor(
            [
                [
                    [0.0, 1.0], # left
                ],
                [
                    [1.0, 0.0], # right
                ]
            ],
            device=t.device,
        )
    else:
        table = torch.tensor(
            [
                [
                    [1.0, 0.0], # left
                ],
                [
                    [0.0, 1.0], # right
                ]
            ],
            device=t.device,
        )
    if t.dtype != torch.long:
        t = t.long()

    return table[t]


def individual_turn_to_obs(t: torch.Tensor) -> torch.Tensor:
    """convert representation of drone turn to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env, 1, 2)
    """
    table = torch.tensor(
        [
            [
                [0.0, 1.0], # hover
            ],
            [
                [1.0, 0.0], # my turn
            ],
            [
                [0.0, 1.0], # hover
            ],
            [
                [0.0, 1.0], # hover
            ],
        ],
        device=t.device,
    )
    if t.dtype != torch.long:
        t = t.long()

    return table[t]

def serve_target_to_obs_3(t: torch.Tensor, near_side=True) -> torch.Tensor:
    """convert representation of ball target to one-hot vector with left, mid, right targets

    Args:
        t (torch.Tensor): (n_env,) tensor, containing the target labels (0 for left, 1 for mid, 2 for right)
        near_side (bool): if the ball is hit from the near side

    Returns:
        torch.Tensor: (n_env, 1, 3) tensor, one-hot encoded target vectors
    """
    if near_side:
        table = torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0] # left
                ],
                [  
                    [0.0, 1.0, 0.0] # mid
                ],  
                [
                    [1.0, 0.0, 0.0]  # right
                ],
            ],
            device=t.device,
        )
    else:
        table = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0] # left
                ],  
                [
                    [0.0, 1.0, 0.0] # mid
                ],  
                [
                    [0.0, 0.0, 1.0] # right
                ],  
            ],
            device=t.device,
        )

    if t.dtype != torch.long:
        t = t.long()

    return table[t]


def serve_turn_to_obs(t: torch.Tensor) -> torch.Tensor:
    """convert representation of drone turn to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env, 1, 2)
    """
    table = torch.tensor(
        [
            [
                [0.0, 1.0],
            ],
            [
                [0.0, 1.0],
            ],
            [
                [0.0, 1.0],
            ],
            [
                [1.0, 0.0], # serve turn
            ]
        ],
        device=t.device,
    )
    if t.dtype != torch.long:
        t = t.long()

    return table[t]