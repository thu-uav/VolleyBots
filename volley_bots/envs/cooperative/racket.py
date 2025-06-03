import math
from typing import List, Tuple, Union

import numpy as np
import torch
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF


def y3(x: np.float64, par: np.ndarray) -> np.float64:
    return par[0] * x**3 + par[1] * x**2 + par[2] * x + par[3]


def y7(x: np.float64, par: np.ndarray) -> np.float64:
    return (
        (
            ((((par[0] * x + par[1]) * x + par[2]) * x + par[3]) * x + par[4]) * x
            + par[5]
        )
        * x
        + par[6]
    ) * x + par[7]


def equation_solving(par: np.ndarray, target: float) -> List[np.float64]:
    """
    二分法求解三次方程
    :param par: 方程系数
    :param target: 目标点
    :return: 方程的实根
    """
    root = []

    for i in np.arange(-2, 2, 0.25):
        left, right = i, i + 0.25
        y1, y2 = y3(left, par), y3(right, par)
        if y1 == target:
            root.append(y1)
        if (y1 - target) * (y2 - target) < 0:
            for j in range(10):
                mid = (left + right) / 2
                if (y3(mid, par) - target) * (y3(right, par) - target) <= 0:
                    left = mid
                else:
                    right = mid
            root.append(right)

    return root


def get_ball_traj_without_kd(p00, v00, target_height: float, device, dtype):
    """
    计算多个环境中的球在无空气阻力下的轨迹

    参数：
    - p00: 形状为 (N, 3) 的球初始位置张量
    - v00: 形状为 (N, 3) 的球初始速度张量
    - target_height: 目标高度（标量）
    - device: 计算设备
    - dtype: 数据类型

    返回：
    - target_ball_pose: 形状为 (N, 3) 的预测目标位置
    - target_ball_vel: 形状为 (N, 3) 的预测目标速度
    - t: 形状为 (N,) 的预测时间
    """
    a = -0.5 * 9.81

    b = v00[:, 2]

    c = -(p00[:, 2] - target_height)

    t1 = torch.zeros_like(b)
    t2 = torch.zeros_like(b)
    t = torch.zeros_like(b)
    valid_mask = (b**2 + 4 * a * c) >= 0

    if valid_mask.any():
        sqrt_term = torch.sqrt(b[valid_mask] ** 2 + 4 * a * c[valid_mask])
        t1[valid_mask] = (-b[valid_mask] + sqrt_term) / (2 * a)
        t2[valid_mask] = (-b[valid_mask] - sqrt_term) / (2 * a)


    t[valid_mask] = torch.where(
        v00[valid_mask, 2] <= 0,
        torch.min(t1[valid_mask], t2[valid_mask]),
        torch.max(t1[valid_mask], t2[valid_mask]),
    )


    t[valid_mask] = torch.where(
        (t1[valid_mask] >= 0) & (t2[valid_mask] >= 0),
        t[valid_mask],
        torch.where(
            t1[valid_mask] >= 0, t1[valid_mask], t2[valid_mask]
        ),
    )


    x = v00[:, 0] * t + p00[:, 0]
    y = v00[:, 1] * t + p00[:, 1]

    target_ball_pose = torch.stack(
        [x, y, torch.full_like(x, target_height)], dim=1
    )
    target_ball_vel = torch.stack(
        [v00[:, 0], v00[:, 1], v00[:, 2] - 9.81 * t], dim=1
    )

    return target_ball_pose, target_ball_vel, t


def get_ball_traj(
    p00: np.ndarray,
    v00: np.ndarray,
    target_height: float,
    kd_est: float,
    ddt: float = 0.005,
):
    """
    :param p00: 球初始位置
    :param v00: 球初始速度
    :param target_height: 给定的预测的下限高度
    :param kd_est: 估计的空气阻力系数
    :param ddt: 模拟步长
    :return: 球的位置轨迹(三次式)参数及速度轨迹(三次式)参数【共六条轨迹】
    """
    """
    如果已知空气阻力系数，那么前面可以省略求取空气阻力系数部分，只需要根据动捕数据求解球速
    """

    a = np.array([0, 0, -9.81])

    ball_pose = []
    ball_vel = []
    # t = []
    ball_pose.append(p00)
    ball_vel.append(v00)
    t_inl = 0.0
    # t.append(t_inl)

    if kd_est != 0:
        while (ball_pose[-1][2] >= target_height) and (abs(ball_vel[-1][2]) <= 20):
            p = ball_pose[-1] + ball_vel[-1] * ddt + 0.5 * a * ddt**2
            # p = ball_pose[-1] + ball_vel[-1]*ddt
            ball_pose.append(p)
            v = (
                ball_vel[-1]
                - kd_est * np.linalg.norm(ball_vel[-1]) * ddt * ball_vel[-1]
                + a * ddt
            )
            ball_vel.append(v)
            t_inl += ddt

    if kd_est == 0:
        while (ball_pose[-1][2] >= target_height) and (abs(ball_vel[-1][2]) <= 20):
            p = ball_pose[-1] + ball_vel[-1] * ddt + 0.5 * a * ddt**2
            # p = ball_pose[-1] + ball_vel[-1]*ddt
            ball_pose.append(p)
            v = ball_vel[-1] + a * ddt
            ball_vel.append(v)
            t_inl += ddt
        # t.append(t_inl)
    """
    此时已大致完成了球的轨迹的求取，由于均为散点，因此多一步多项式拟合
    但也可以直接根据指定的碰撞高度，找到对应的相近的ball_pose及对应的时间，同样可以得到所需要的碰撞点的数据
    ————碰撞点坐标(x,y,z)、碰撞点球速(vx,vy,vz)
    """
    target_ball_pose = (ball_pose[-2] + ball_pose[-1]) / 2
    target_ball_vel = (ball_vel[-2] + ball_vel[-1]) / 2
    t_inl = (2 * t_inl - ddt) / 2
    # target_ball_pose = ball_pose[-1]
    # target_ball_vel = ball_vel[-1]
    return target_ball_pose, target_ball_vel, t_inl


def ball_post_vel_without_kd(ball_target_pose, ball_collision_pose, device, dtype):
    """
    计算多个环境的球碰撞后的速度
    参数：
    - ball_target_pose: (N, 3) 目标位置
    - ball_collision_pose: (N, 3) 碰撞位置
    - device: 计算设备
    - dtype: 数据类型

    返回：
    - ball_post_v: (N, 3) 碰撞后的速度
    """
    t = 1.5


    del_x = ball_target_pose[:, 0] - ball_collision_pose[:, 0]
    del_y = ball_target_pose[:, 1] - ball_collision_pose[:, 1]
    vx = del_x / t
    vy = del_y / t
    vz = (ball_target_pose[:, 2] - ball_collision_pose[:, 2] + 0.5 * 9.81 * t**2) / t

    ball_post_v = torch.stack([vx, vy, vz], dim=1).to(
        device=device, dtype=dtype
    )

    return ball_post_v


def ball_post_vel(kd, ball_target_pose, ball_collision_pose):

    ball_hor_move = np.array(
        [
            ball_target_pose[0] - ball_collision_pose[0],
            ball_target_pose[1] - ball_collision_pose[1],
        ]
    )
    ball_pose_move = np.array([np.linalg.norm(ball_hor_move), ball_target_pose[2]])
    # print(ball_pose_move)
    ball_hor_move_angel = np.arctan2(ball_hor_move[1], ball_hor_move[0])

    a = np.array([0, -9.81])
    ddt = 0.001
    ball_post_v = np.array([0, 0, 0])


    ball_pose = []
    ball_vel = []
    # t = []
    ball_pose.append(ball_pose_move)
    for i in range(1, 250):
        v00 = np.array([i * 0.02, -i * 0.04])
        ball_vel.append(v00)
        ball_pose.append(ball_pose_move)
        while (ball_pose[-1][1] >= ball_collision_pose[2]) and (
            abs(ball_vel[-1][1]) <= 20
        ):
            p = ball_pose[-1] - ball_vel[-1] * ddt - 0.5 * a * ddt**2
            # p = ball_pose[-1] + ball_vel[-1]*ddt
            ball_pose.append(p)
            # print(p)
            v = (
                ball_vel[-1]
                + kd * np.linalg.norm(ball_vel[-1]) * ddt * ball_vel[-1]
                - a * ddt
            )
            # print(v)
            ball_vel.append(v)
        if ball_pose[-1][0] <= 1e-3:
            ball_post_v = np.array(
                [
                    ball_vel[-1][0] * np.cos(ball_hor_move_angel),
                    ball_vel[-1][0] * np.sin(ball_hor_move_angel),
                    ball_vel[-1][1],
                ]
            )
            break
        else:
            ball_pose.clear()
            ball_vel.clear()

    return ball_post_v


def get_uav_collision_data_without_kd(
    ball_pose, ball_vel_per_collision, beta, ball_vel_post_collision, device, dtype
):
    f = torch.tensor(
        [
            [
                6.5,
                7.0,
                7.5,
                8.0,
                8.5,
                9.0,
                9.5,
                10.0,
                10.5,
                11.0,
                11.5,
                12.0,
                12.5,
                13.0,
                13.5,
            ]
        ]
    ).T.to(device=device, dtype=dtype)
    N = ball_pose.shape[0]
    uav_data_planning = torch.zeros((N, 9), dtype=dtype, device=device)
    uav_data_planning[:, 0:3] = ball_pose
    uav_data_planning[:, 3:6] = (
        1 / (1 + beta) * (beta * ball_vel_per_collision + ball_vel_post_collision)
    )


    ball_val_delta = ball_vel_per_collision - ball_vel_post_collision  # (N, 3)
    n_des = -ball_val_delta / torch.norm(ball_val_delta, dim=-1, keepdim=True)  # (N, 3)

    # n_des = (sinpcosr,-sinr,cospcosr)
    roll = -torch.asin(n_des[:, 1])
    pitch = torch.atan2(n_des[:, 0], n_des[:, 2])

    pitch = torch.where(pitch >= math.pi, pitch - math.pi, pitch)
    pitch = torch.where(pitch <= -math.pi, pitch + math.pi, pitch)

    yaw = torch.zeros(N, dtype=dtype, device=device)

    uav_data_planning[:, 6:9] = torch.stack((roll, pitch, yaw), dim=1)

    return uav_data_planning


    # ball_val_delta = (ball_vel_per_collision - ball_vel_post_collision) # (N, 3)
    # n_des = -ball_val_delta / torch.norm(ball_val_delta, dim=-1, keepdim=True) # (N, 3)
    # # print("ball_val_delta",ball_val_delta.shape)
    # # print("n_des",n_des.shape)
    # f = f.expand(-1, N) # (15, N)


    # a_cal = torch.norm(a_f + torch.tensor([0., 0., -9.81], dtype=dtype, device=device), dim=-1) # (15,)

    # a_real = a_f[min_idx] + torch.tensor([0., 0., -9.81], dtype=dtype, device=device)  # (N, 3)




    # ball_val_delta = (ball_vel_per_collision - ball_vel_post_collision)
    # n_des = np.array([-ball_val_delta / np.linalg.norm(ball_val_delta)])
    # # uav_data_planning[3:6] = np.dot(v_mid,n_des.T)*n_des

    # # n_des = (sinpcosr,-sinr,cospcosr)
    # a_f = np.dot(f,n_des)
    # a_real = np.array([0.,0.,0.])
    # a_cal = []
    # for i in range(len(f)):
    #     a = np.linalg.norm(a_f[i] + np.array([0.,0.,-9.81]))
    #     a_cal.append(a)
    # a_real = a_f[a_cal.index(min(a_cal))] + np.array([0.,0.,-9.81])
    # # for i in range(len(f)):
    # #     if (a_f[i][2]-9.81) > -2 and (a_f[i][2]-9.81) <= 0:
    # #         a_real = a_f[i] + np.array([0.,0.,-9.81])
    # #         break
    # uav_data_planning[6] = a_real[0]
    # uav_data_planning[7] = a_real[1]
    # uav_data_planning[8] = a_real[2]
    # time_left = time_max
    # return uav_data_planning


def get_uav_collision_data(
    ball_pose: np.ndarray,
    ball_vel_per_collision: np.ndarray,
    beta: float,
    ball_vel_post_collision: np.ndarray,
):
    """
    求取无人机在碰撞点的位置、速度及姿态
    :param collision_height: 碰撞高度
    :param ball_pose_poly_par: 球的位置轨迹参数
    :param ball_vel_poly_par: 球的速度轨迹参数
    :param beta: 球拍的恢复系数 (应为0.7左右)
    :param ball_vel_post_collision: 球碰撞后的速度
    :return: 无人机在碰撞点的位置、速度及姿态，以及剩余的可供无人机规划和执行的时间
    也即返回（x,y,z,vx,vy,vz,roll,pitch,yaw）
    """
    # print(collision_height)
    f = np.array(
        [
            [
                6.5,
                7.0,
                7.5,
                8.0,
                8.5,
                9.0,
                9.5,
                10.0,
                10.5,
                11.0,
                11.5,
                12.0,
                12.5,
                13.0,
                13.5,
            ]
        ]
    ).T
    uav_data_planning = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    uav_data_planning[0:3] = ball_pose
    uav_data_planning[3:6] = (
        1 / (1 + beta) * (beta * ball_vel_per_collision + ball_vel_post_collision)
    )

    ball_val_delta = ball_vel_per_collision - ball_vel_post_collision
    n_des = np.array([-ball_val_delta / np.linalg.norm(ball_val_delta)])
    # uav_data_planning[3:6] = np.dot(v_mid,n_des.T)*n_des

    # n_des = (sinpcosr,-sinr,cospcosr)
    a_f = np.dot(f, n_des)
    a_real = np.array([0.0, 0.0, 0.0])
    a_cal = []
    for i in range(len(f)):
        a = np.linalg.norm(a_f[i] + np.array([0.0, 0.0, -9.81]))
        a_cal.append(a)
    a_real = a_f[a_cal.index(min(a_cal))] + np.array([0.0, 0.0, -9.81])
    # for i in range(len(f)):
    #     if (a_f[i][2]-9.81) > -2 and (a_f[i][2]-9.81) <= 0:
    #         a_real = a_f[i] + np.array([0.,0.,-9.81])
    #         break
    uav_data_planning[6] = a_real[0]
    uav_data_planning[7] = a_real[1]
    uav_data_planning[8] = a_real[2]
    # time_left = time_max
    return uav_data_planning


def f_x(x, dt):
    """
    :param x: 球的质心位置及速度([x, x', y, y', z, z'])
    :param dt: 时间间隔
    :return: 球的运动矩阵
    """
    v = np.array([x[3:6]])
    v_norm = np.linalg.norm(v)
    # print(v_norm)
    F = np.array(
        [
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt - 0.5 * 9.81 * dt**2 / x[5]],
            [0, 0, 0, 1 - 0.06 * v_norm * dt, 0, 0],
            [0, 0, 0, 0, 1 - 0.06 * v_norm * dt, 0],
            [0, 0, 0, 0, 0, 1 - (0.06 * v_norm + 9.81 / x[5]) * dt],
        ]
    )
    # F = np.array([[1, 0, 0, dt, 0, 0],
    #               [0, 1, 0, 0, dt, 0],
    #               [0, 0, 1, 0, 0, dt],
    #               [0, 0, 0, 1, 0, 0],
    #               [0, 0, 0, 0, 1, 0],
    #               [0, 0, 0, 0, 0, 1]])
    return F @ x


def f_x_uav(x, dt):
    F = np.array(
        [
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    return F @ x



def h_cv(x):
    return x[[0, 1, 2]]


def get_ukf_data(ukf, msg):
    """
    :return: 空气阻力系数
    """
    ukf.predict()
    ukf.update(msg)
    data_ukf = ukf.x.copy()
    return data_ukf


def make_ukf(dt: float):
    sigmas = MerweScaledSigmaPoints(6, alpha=0.0001, beta=2.0, kappa=-3.0)
    ukf_p = UKF(dim_x=6, dim_z=3, fx=f_x, hx=h_cv, dt=dt, points=sigmas)

    ukf_p.x = np.array([0.0, 0.0, 1.5, 0.0, 0.0, 7.0])

    ukf_p.R = np.diag([0.005**2, 0.005**2, 0.005**2])

    ukf_p.Q[0:3, 0:3] = np.diag(
        [0.0, 0.0, 0.0]
    )  # )Q_discrete_white_noise(3, dt=dt, var=0.)
    ukf_p.Q[3:6, 3:6] = np.diag(
        [10.0, 10.0, 10.0]
    )  # Q_discrete_white_noise(3, dt=dt, var=10)
    return ukf_p


def make_ukf_uav(dt: float):
    sigmas = MerweScaledSigmaPoints(6, alpha=0.0001, beta=2.0, kappa=-3.0)
    ukf_p = UKF(dim_x=6, dim_z=3, fx=f_x_uav, hx=h_cv, dt=dt, points=sigmas)

    ukf_p.x = np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.1])

    ukf_p.R = np.diag([0.005**2, 0.005**2, 0.005**2])

    ukf_p.Q[0:3, 0:3] = np.diag(
        [0.0, 0.0, 0.0]
    )  # Q_discrete_white_noise(3, dt=dt, var=0.)
    ukf_p.Q[3:6, 3:6] = np.diag(
        [10.0, 10.0, 10.0]
    )  # Q_discrete_white_noise(3, dt=dt, var=10)
    return ukf_p


def get_kd_est(
    ukf_v_post: np.ndarray,
    ukf_v_pre: np.ndarray,
    ukf_height_post: Union[float, np.ndarray],
    ukf_height_pre: Union[float, np.ndarray],
    dt: float,
    Kds: List[np.ndarray],
    Pds: List[np.ndarray],
):
    ukf_v_post_norm = np.linalg.norm(ukf_v_post)
    ukf_v_pre_norm = np.linalg.norm(ukf_v_pre)
    e_ball_ukf_post = (
        0.5 * ukf_v_post_norm**2 + 9.81 * ukf_height_post
    )
    e_ball_ukf_pre = 0.5 * ukf_v_pre_norm**2 + 9.81 * ukf_height_pre
    v_delta = (0.5 * (ukf_v_post_norm + ukf_v_pre_norm)) ** 3

    kd_cal = np.abs((e_ball_ukf_post - e_ball_ukf_pre) / dt / v_delta)

    rd = np.var(np.array([ukf_v_post_norm, ukf_v_pre_norm])) / v_delta
    cd = Pds[-1] / (Pds[-1] + rd)
    kd = Kds[-1] + cd * (kd_cal - Kds[-1])
    Kds.append(kd)
    pd = Pds[-1] * rd / (Pds[-1] + rd)
    # print(kd,pd)
    Pds.append(pd)


if __name__ == "__main__":
    b_p_p, b_v_p, t = get_ball_traj_without_kd(
        p00=torch.tensor([2.1, -0.5, 5]),
        v00=torch.tensor([-10, -2, -6]),
        target_height=1.1,
        device="cpu",
        dtype=float,
    )
    print("b_p_p,b_v_p, t", b_p_p, b_v_p, t)
    b_v_v = ball_post_vel_without_kd(
        ball_collision_pose=b_p_p,
        ball_target_pose=torch.tensor([2.0, 1.0, 1]),
        device="cpu",
        dtype=float,
    )
    print(b_v_v)
    uav_data = get_uav_collision_data_without_kd(
        b_p_p, b_v_p, 0.77, b_v_v, "cpu", float
    )
    print(uav_data)
    pose = uav_data[0:3]
    print(pose)
