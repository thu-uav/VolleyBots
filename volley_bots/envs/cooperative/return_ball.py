import math
import os
import signal
import sys
from enum import Enum
from math import cos, sin

import hydra
import numpy as np
import quaternion
import torch
from omegaconf import OmegaConf

import volley_bots.envs.cooperative.quadrocoptertrajectory as quadtraj

import volley_bots.envs.cooperative.racket as racket
from volley_bots import init_simulation_app


class func_gen:
    def __init__(self, item):
        self.item5 = item[5]
        self.item4 = item[4]
        self.item3 = item[3]
        self.item2 = item[2]
        self.item1 = item[1]
        self.item0 = item[0]

    def cal(self, t):
        res = (
            self.item5 * t**5
            + self.item4 * t**4
            + self.item3 * t**3
            + self.item2 * t**2
            + self.item1 * t**1
            + self.item0
        )
        return res


def cal_ori(roll, pitch, yaw, device):
    # u1 = torch.sqrt(acc_x**2 + acc_y**2 + (acc_z+9.81)**2)
    # roll = torch.asin(-acc_y / u1)
    # pitch = torch.atan2(acc_x, (acc_z+9.81))

    # pitch = torch.where(pitch > math.pi, pitch - math.pi, pitch)
    # pitch = torch.where(pitch < -math.pi, pitch + math.pi, pitch)

    # yaw = torch.zeros_like(pitch)  # 设定 yaw 角

    qx = torch.sin(roll / 2) * torch.cos(pitch / 2) * torch.cos(yaw / 2) - torch.cos(
        roll / 2
    ) * torch.sin(pitch / 2) * torch.sin(yaw / 2)
    qy = torch.cos(roll / 2) * torch.sin(pitch / 2) * torch.cos(yaw / 2) + torch.sin(
        roll / 2
    ) * torch.cos(pitch / 2) * torch.sin(yaw / 2)
    qz = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.sin(yaw / 2) - torch.sin(
        roll / 2
    ) * torch.sin(pitch / 2) * torch.cos(yaw / 2)
    qw = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.cos(yaw / 2) + torch.sin(
        roll / 2
    ) * torch.sin(pitch / 2) * torch.sin(yaw / 2)
    return qw.to(device), qx.to(device), qy.to(device), qz.to(device)


def cal_pos_vel_ori(
    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, dtpye, device
):
    pose = torch.tensor([pos_x, pos_y, pos_z], device=device, dtype=dtpye)
    vel = torch.tensor([vel_x, vel_y, vel_z, 0.0, 0.0, 0.0], device=device, dtype=dtpye)
    qw, qx, qy, qz = cal_ori(acc_x, acc_y, acc_z)
    ori = torch.tensor([qw, qx, qy, qz], device=device, dtype=dtpye)
    return pose, vel, ori


def uav_traj_create(traj_s, traj_v, traj_a, end_t, dtpye, device, time_period):
    time_period = time_period
    end_time = end_t

    pos_x = func_gen(traj_s[0])
    pos_y = func_gen(traj_s[1])
    pos_z = func_gen(traj_s[2])
    vel_x = func_gen(traj_v[0])
    vel_y = func_gen(traj_v[1])
    vel_z = func_gen(traj_v[2])
    acc_x = func_gen(traj_a[0])
    acc_y = func_gen(traj_a[1])
    acc_z = func_gen(traj_a[2])
    px, py, pz, vx, vy, vz, ax, ay, az = 0, 0, 0, 0, 0, 0, 0, 0, 0
    p_l, v_l, o_l = [], [], []

    t = 0

    while t <= end_time:
        px = pos_x.cal(t)
        py = pos_y.cal(t)
        pz = pos_z.cal(t)

        vx = vel_x.cal(t)
        vy = vel_y.cal(t)
        vz = vel_z.cal(t)

        ax = acc_x.cal(t)
        ay = acc_y.cal(t)
        az = acc_z.cal(t)

        pose, vel, ori = cal_pos_vel_ori(
            px, py, pz, vx, vy, vz, ax, ay, az, dtpye, device
        )
        p_l.append(pose)
        v_l.append(vel)
        o_l.append(ori)

        t += time_period

    for _ in range(12):
        pose, vel, ori = cal_pos_vel_ori(
            px, py, pz, vx, vy, vz, ax, ay, az, dtpye, device
        )
        p_l.append(pose)
        v_l.append(vel)
        o_l.append(ori)

    # np.save('traj_dict.npy', uav_traj)
    # print("new traj has been created")
    return p_l, v_l, o_l


def cal_uav_traj(
    uav_data, drone_pose_now, drone_vel_linear_now, drone_acc_linear_now, ball_predict_t
):
    # Define the trajectory starting state:
    pos0 = drone_pose_now  # position
    vel0 = drone_vel_linear_now  # velocity
    acc0 = drone_acc_linear_now  # acceleration
    # Define the goal state:
    posf = uav_data[0:3]  # position
    velf = uav_data[3:6]  # velocity
    accf = uav_data[6:9]  # acceleration
    # Define the duration:
    Tf = ball_predict_t
    # Define the input limits:
    fmin = 2  # [m/s**2]
    fmax = 45  # [m/s**2]
    wmax = 8  # [rad/s]
    minTimeSec = 0.02  # [s]
    # Define how gravity lies:
    gravity = [0.0, 0.0, -9.81]
    traj = quadtraj.RapidTrajectory(pos0, vel0, acc0, gravity)
    traj.set_goal_position(posf)
    traj.set_goal_velocity(velf)
    traj.set_goal_acceleration(accf)
    traj.generate(Tf)
    # Test input feasibility
    inputsFeasible = traj.check_input_feasibility(fmin, fmax, wmax, minTimeSec)
    # Test whether we fly into the floor
    floorPoint = [0, 0, 0]  # a point on the floor
    floorNormal = [0, 0, 1]  # we want to be in this direction of the point (upwards)
    positionFeasible = traj.check_position_feasibility(floorPoint, floorNormal)
    return traj, inputsFeasible, positionFeasible


def return_ball_to_target_position_traj(
    ball_pose_now: np.ndarray,
    ball_vel_now: np.ndarray,
    ball_target_pose: np.ndarray,
    hit_target_height: float,
    kd_now: float,
    board_pose_now: np.ndarray,
    board_vel_now: np.ndarray,
    device,
    ddt,
):

    init_pose = torch.tensor([-3, 0.0, 1.0], device=device)
    p_l, v_l, o_l = [], [], []
    p_l_re, v_l_re, o_l_re = [], [], []
    uav_data = np.zeros(9)
    ball_vel_post = np.array([0.0, 0.0, 0.0])

    # ball_pose, _ = ball.get_world_poses()
    # ball_pose = ball_pose.squeeze().cpu().numpy()
    # ball_vel = ball.get_linear_velocities().squeeze().cpu().numpy()
    # if np.sqrt(np.dot((ball_pose-ball_target_pose),(ball_pose-ball_target_pose))) < 0.5:
    #     print("chabie:",ball_pose-ball_target_pose)

    if np.abs(ball_vel_now[2]) >= 1.0 and ball_pose_now[2] >= hit_target_height + 0.1:
        ball_predict_pose, ball_predict_vel, ball_predict_t = racket.get_ball_traj(
            ball_pose_now, ball_vel_now, hit_target_height, kd_now, 0.005
        )
        # print(ball_predict_pose)
        if ball_predict_t >= 0.4 and ball_pose_now[2] >= hit_target_height:
            # drone_pose_now,_ = board.get_world_poses()
            drone_pose_now = board_pose_now
            drone_vel_linear_now = board_vel_now
            drone_acc_now = [0.0, 0.0, 0.0]
            drone_init_data = [
                board_pose_now[0],
                board_pose_now[1],
                board_pose_now[2],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            uav_traj_s, uav_traj_v, uav_traj_a = [], [], []
            uav_traj_s_re, uav_traj_v_re, uav_traj_a_re = [], [], []
            ball_vel_post = racket.ball_post_vel_without_kd(
                ball_target_pose, ball_predict_pose, ddt
            )
            uav_data = racket.get_uav_collision_data(
                ball_predict_pose, ball_predict_vel, 0.75, ball_vel_post
            )
            traj, inputsFeasible, positionFeasible = cal_uav_traj(
                uav_data,
                drone_pose_now[0:3],
                drone_vel_linear_now[0:3],
                drone_acc_now,
                ball_predict_t - 4 * ddt,
            )
            traj_re, inputsFeasible_re, positionFeasibl_re = cal_uav_traj(
                drone_init_data, uav_data[0:3], uav_data[3:6], uav_data[6:9], 0.6
            )
            if positionFeasible == 0 and positionFeasibl_re == 0:
                for i in range(3):
                    alpha, beta, gamma = (
                        traj.get_param_alpha(i),
                        traj.get_param_beta(i),
                        traj.get_param_gamma(i),
                    )
                    alpha_re, beta_re, gamma_re = (
                        traj_re.get_param_alpha(i),
                        traj_re.get_param_beta(i),
                        traj_re.get_param_gamma(i),
                    )

                    drone_s = np.array(
                        [
                            drone_pose_now[i],
                            drone_vel_linear_now[i],
                            drone_acc_now[i] / 2,
                            gamma / 6,
                            beta / 24,
                            alpha / 120,
                        ]
                    )
                    drone_re_s = np.array(
                        [
                            uav_data[0:3][i],
                            uav_data[3:6][i],
                            uav_data[6:9][i] / 2,
                            gamma_re / 6,
                            beta_re / 24,
                            alpha_re / 120,
                        ]
                    )

                    drone_v = np.array(
                        [
                            drone_s[1],
                            drone_s[2] * 2,
                            drone_s[3] * 3,
                            drone_s[4] * 4,
                            drone_s[5] * 5,
                            0.0,
                        ]
                    )
                    drone_re_v = np.array(
                        [
                            drone_re_s[1],
                            drone_re_s[2] * 2,
                            drone_re_s[3] * 3,
                            drone_re_s[4] * 4,
                            drone_re_s[5] * 5,
                            0.0,
                        ]
                    )

                    drone_a = np.array(
                        [
                            drone_v[1],
                            drone_v[2] * 2,
                            drone_v[3] * 3,
                            drone_v[4] * 4,
                            0.0,
                            0.0,
                        ]
                    )
                    drone_re_a = np.array(
                        [
                            drone_re_v[1],
                            drone_re_v[2] * 2,
                            drone_re_v[3] * 3,
                            drone_re_v[4] * 4,
                            0.0,
                            0.0,
                        ]
                    )

                    uav_traj_s.append(drone_s)
                    uav_traj_v.append(drone_v)
                    uav_traj_a.append(drone_a)

                    uav_traj_s_re.append(drone_re_s)
                    uav_traj_v_re.append(drone_re_v)
                    uav_traj_a_re.append(drone_re_a)

                p_l, v_l, o_l = uav_traj_create(
                    uav_traj_s,
                    uav_traj_v,
                    uav_traj_a,
                    ball_predict_t - 4 * ddt,
                    init_pose.dtype,
                    device,
                    time_period=ddt,
                )
                p_l_re, v_l_re, o_l_re = uav_traj_create(
                    uav_traj_s_re,
                    uav_traj_v_re,
                    uav_traj_a_re,
                    0.6,
                    init_pose.dtype,
                    device,
                    time_period=ddt,
                )

                # print("ball_vel_post", ball_vel_post)
                # print("ball_predict_vel",ball_predict_vel)
                # print("uav_data",uav_data[3:6])

        return p_l, v_l, o_l, p_l_re, v_l_re, o_l_re

    else:
        # print("fail to create traj")
        return [], [], [], [], [], []


# @hydra.main(version_base=None, config_path='.', config_name="demo")
# def main(cfg):
#     """
#     a sample to show how to use the script
#     """
#     kd_now = 0.00                                                           # air drag of ball, but it is not so important
#     target_height = 1.5                                                     # the height when board collides with call
#     m = 0.003                                                               # the mass of ball
#     p_l,v_l,o_l,p_l_re,v_l_re,o_l_re = [],[],[],[],[],[]
#     new_traj = False                                                        # if true, the borad will execute traj
#     enable_create_traj = False                                              # if true, the script will create traj

#     ball_target_pose = np.array([1.0, 1.0, 1.5])                            # the position that the ball needs to reach
#                                                                             #please don't let the height of the ball be less than the net's

#     OmegaConf.resolve(cfg)                                                  # env
#     simulation_app = init_simulation_app(cfg)
#     print(OmegaConf.to_yaml(cfg))

#     import volley_bots.utils.scene as scene_utils
#     from omni.isaac.core.simulation_context import SimulationContext
#     from volley_bots.controllers import RateController, Px4RateController
#     from volley_bots.robots.drone import MultirotorBase
#     from omni.isaac.core.objects import DynamicSphere, DynamicCylinder
#     from omni.isaac.core.prims import RigidPrimView
#     import omni.isaac.core.materials as materials

#     sim = SimulationContext(
#         stage_units_in_meters=1.0,
#         physics_dt=cfg.sim.dt,
#         rendering_dt=cfg.sim.dt,
#         sim_params=cfg.sim,
#         backend="torch",
#         device=cfg.sim.device
#     )
#     print("finish sim")

#     material = materials.PhysicsMaterial(
#             prim_path="/World/Physics_Materials/physics_material_0",
#             restitution=0.8
#     )

#     # create a ball
#     ball = DynamicSphere(
#         prim_path="/dynamic_sphere",
#         name="dynamic_sphere",
#         mass=m,
#         position=np.array([5.0, 5.0, 0.0]),
#         radius=0.02,
#         color=np.array([255, 0, 255]),
#         physics_material=material
#     )
#     ball = RigidPrimView(
#         prim_paths_expr=ball.prim_path,
#         reset_xform_properties=False,
#         track_contact_forces=True,
#     )

#     # create a board
#     board = DynamicCylinder(
#         prim_path="/dynamic_cylinder",
#         name="dynamic_cylinder",
#         mass=m*200, # big mass to ignore the impact of collision
#         position=np.array([-1.5, -1.5, 0.0]),
#         radius=0.25,
#         height=0.02,
#         color= np.array([0, 0, 255]),
#         physics_material=material
#     )

#     board = RigidPrimView(
#         prim_paths_expr=board.prim_path,
#         reset_xform_properties=False,
#         track_contact_forces=False,
#     )

#     print("finish ball")
#     scene_utils.design_scene()
#     sim.reset()
#     ball.initialize()
#     board.initialize()


#     sim._physics_sim_view.flush()
#     ball.enable_rigid_body_physics()
#     board.enable_rigid_body_physics()
#     print("finish setting")


#     # Exit with ctrl+c
#     global stop_while
#     stop_while = False
#     def signal_handler(signal, frame):
#         print ('\ntap Ctrl+C to exit!')
#         global stop_while
#         stop_while = True
#     signal.signal(signal.SIGINT, signal_handler)

#     sim_cnt = 0

#     while True:
#         if sim.is_stopped():
#             break
#         if not sim.is_playing():
#             sim.render()
#             continue
#         sim.step(render=True)
#         sim_cnt += 1

#         if sim_cnt == 400:
#             p = torch.tensor([[1., 1., 1.5]], device=sim.device)
#             o = torch.tensor([[1.0, 0., 0., 0.]], device=sim.device)
#             v = torch.tensor([[-1.0, -1.0, 7.0, 0., 0., 0.]], device=sim.device)
#             init_pose = torch.tensor([[-1.5, -1.5, 1.5]], device=sim.device)
#             init_ori = torch.tensor([[1.0, 0., 0., 0]], device=sim.device)
#             board.set_world_poses(init_pose, init_ori)
#             ball.set_world_poses(p,o)
#             ball.set_velocities(v)
#             print("<<<BALL THROWN!>>>")

#             # When the ball moves towards the board, it is allowed to create trajectories
#             enable_create_traj = True

#         if sim_cnt >= 400:
#             ball_pose, _ = ball.get_world_poses()
#             ball_pose = ball_pose.squeeze().cpu().numpy()
#             # ball_vel = ball.get_linear_velocities().squeeze().cpu().numpy()
#             if np.sqrt(np.dot((ball_pose-ball_target_pose),(ball_pose-ball_target_pose))) < 0.5:
#                 print("chabie:",ball_pose-ball_target_pose)

#             # when the script has finished created traj, it is allowed the board to execute traj
#             if new_traj:
#                 if len(p_l) > 1:
#                     board.set_world_poses(positions=p_l[0], orientations=o_l[0])
#                     board.set_velocities(v_l[0])
#                     p_l.pop(0)
#                     o_l.pop(0)
#                     v_l.pop(0)
#                 else:
#                     board.set_world_poses(positions=p_l_re[0], orientations=o_l_re[0])
#                     board.set_velocities(v_l_re[0])
#                     p_l_re.pop(0)
#                     o_l_re.pop(0)
#                     v_l_re.pop(0)
#                     if len(p_l_re) < 2:
#                         new_traj = False
#                         board.set_world_poses(positions=init_pose, orientations=init_ori)
#                         board.set_velocities(torch.tensor([[0., 0., 0., 0., 0., 0.]], device=sim.device))

#             if enable_create_traj:
#                 # create new traj for board
#                 p_l,v_l,o_l,p_l_re,v_l_re,o_l_re = return_ball_to_target_position_traj(ball_target_pose, target_height, kd_now, board,ball, sim, cfg)
#                 # let the board execute traj
#                 if len(p_l) >= 2:
#                     enable_create_traj = False
#                     new_traj = True

#         if stop_while:
#             break

#     simulation_app.close()
#     print('Finish.')

if __name__ == "__main__":
    p00 = torch.tensor([0, -0.0, 0.0], device="cpu", dtype=float)
    w, x, y, z = cal_ori(p00[0], p00[1], p00[2])
    print(w, x, y, z)
