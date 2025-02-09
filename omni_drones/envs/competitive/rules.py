import pdb

import torch


def in_half_court_1(pos: torch.Tensor, L: float, W: float) -> torch.Tensor:
    return (
        (pos[..., 0] >= (-L / 2))
        & (pos[..., 0] < 0)
        & (pos[..., 1] >= (-W / 2))
        & (pos[..., 1] <= (W / 2))
    )


def in_half_court_0(pos: torch.Tensor, L: float, W: float) -> torch.Tensor:
    return (
        (pos[..., 0] > 0)
        & (pos[..., 0] <= (L / 2))
        & (pos[..., 1] >= (-W / 2))
        & (pos[..., 1] <= (W / 2))
    )


def in_half_court(pos: torch.Tensor, L: float, W: float) -> torch.Tensor:
    tmp = torch.concat([in_half_court_0(pos, L, W), in_half_court_1(pos, L, W)], dim=-1)
    return tmp


def _not_in_bounds(pos: torch.Tensor, L: float, W: float) -> torch.Tensor:
    """_summary_

    Args:
        pos (torch.Tensor): (*,3)
        L (float): _description_
        W (float): _description_

    Returns:
        torch.Tensor: (*,1)
    """
    boundary_xy = (
        torch.tensor([L / 2, W / 2], device=pos.device).unsqueeze(0).unsqueeze(0)
    )  # (1,1,2)
    pos_xy = pos[..., :2].abs()  # (*,2)

    tmp = (pos_xy > boundary_xy).any(-1)  # (*,1)
    return tmp


def calculate_ball_hit_net(
    ball_pos: torch.Tensor, r: float, W: float, H_NET: float
) -> torch.Tensor:
    """The function is kinematically incorrect and only applicable to non-boundary cases.
    But it's efficient and very easy to implement

    Args:
        ball_pos (torch.Tensor): (E,1,3)
        r (float): radius of the ball
        W (float): width of the imaginary net
        H_NET (float): height of the imaginary net

    Returns:
        torch.Tensor: (E,1)
    """
    tmp = (
        (
            ball_pos[:, :, 0].abs() < 3 * r
        )  # * 3 is to avoid the case where the ball hits the net without being reported due to simulation steps
        & (ball_pos[:, :, 1].abs() < W / 2)
        & (ball_pos[:, :, 2] < H_NET)
    )  # (E,1)
    return tmp


def calculate_drone_hit_net(
    drone_pos: torch.Tensor, anchor: torch.Tensor, L: float, W: float
) -> torch.Tensor:
    """_summary_

    Args:
        drone_pos (torch.Tensor): (E,N,3)
        anchor (torch.Tensor): (N,3)
        L (float): _description_
        W (float): _description_

    Returns:
        torch.Tensor: _description_
    """

    tmp = anchor * drone_pos
    return tmp[:, :, 0] < 0


def determine_game_result(
    L: float,
    W: float,
    H_NET: float,
    ball_radius: float,
    ball_pos: torch.Tensor,
    turn: torch.Tensor,
    wrong_hit_turn: torch.Tensor,
    wrong_hit_racket: torch.Tensor,
    drone_pos: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    """_summary_

    Args:
        L (float): _description_
        W (float): _description_
        turn (torch.Tensor): (E,)
        ball_pos (torch.Tensor): (E,3)
        ball_hit_net (torch.Tensor): (E,1)
        wrong_hit (torch.Tensor): (E,2)
        drone_pos (torch.Tensor): (E,2)
        anchor (torch.Tensor): (2,3)

    Returns:
        torch.Tensor: (E,)
    """
    ball_hit_net = calculate_ball_hit_net(ball_pos, ball_radius, W, H_NET)
    ball_hit_ground = ball_pos[..., 2] < 0.2  # (E,1)
    drone_hit_net = calculate_drone_hit_net(drone_pos, anchor, L, W)
    turn_one_hot = torch.nn.functional.one_hot(turn, 2)  # (E,2)
    wrong_hit = wrong_hit_turn | wrong_hit_racket  # (E,2)

    # 击球犯规1: 击球的人不是击球方
    case_1 = wrong_hit_turn  # (E,2)
    # 击球犯规2: 使用非球拍部位击球
    case_2 = wrong_hit_racket  # (E,2)
    # 无人机越网
    case_3 = drone_hit_net  # (E,2)
    # 球落地
    # 不用判断是哪方击的球
    case_4 = ball_hit_ground & in_half_court(ball_pos, L, W)  # (E,2)
    # 球出界
    # 需要判断是哪方击的球：1）如果存在wrong_hit，那么wrong_hit的一方就是击球方；2）如果不存在wrong_hit，那么turn就是击球方
    case_5 = _not_in_bounds(ball_pos, L, W) & (
        wrong_hit | (~wrong_hit.any(-1, keepdim=True) & ~turn_one_hot)
    )  # (E,2)
    # 球触网
    # 需要判断是哪方击的球，同上
    case_6 = ball_hit_net & (
        wrong_hit | (~wrong_hit.any(-1, keepdim=True) & ~turn_one_hot)
    )  # (E,2)

    cases = torch.stack(
        [case_1, case_2, case_3, case_4, case_5, case_6], dim=-1
    )  # (E,2,6)
    draw = cases.any(-1).all(-1, keepdim=True)  # (E,1)
    result = (cases.any(-1) & ~draw).int()  # (E,2)
    result = (result[:, 1] - result[:, 0]).unsqueeze(-1)  # (E,1)
    return draw, result, cases


def game_result_to_matrix(game_result: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(
        [[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]], device=game_result.device
    )
    return table[game_result.squeeze(-1) + 1]


def determine_game_result_3v3(
    L: float,
    W: float,
    H_NET: float,
    ball_radius: float,
    ball_pos: torch.Tensor,
    last_hit_team: torch.Tensor,
    wrong_hit_turn: torch.Tensor,
    wrong_hit_racket: torch.Tensor,
    drone_pos: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    """_summary_

    Args:
        L (float): _description_
        W (float): _description_
        H_NET (float): _description_
        ball_radius (float): _description_

        ball_pos (torch.Tensor): (E,1,3)
        last_hit_team (torch.Tensor): (E,)
        wrong_hit_turn (torch.Tensor): (E,6)
        wrong_hit_racket (torch.Tensor): (E,6)
        drone_pos (torch.Tensor): (E,6,3)
        anchor (torch.Tensor): (6,3)

    Returns:
        torch.Tensor: (E,)
    """

    ball_hit_net = calculate_ball_hit_net(ball_pos, ball_radius, W, H_NET)  # (E,1)
    ball_hit_ground = ball_pos[..., 2] < 0.2  # (E,1)
    drone_hit_net = calculate_drone_hit_net(drone_pos, anchor, L, W)  # (E,6)
    turn_one_hot = torch.nn.functional.one_hot(last_hit_team, 2)  # (E,2)
    extended_turn_one_hot = torch.stack(
        [
            turn_one_hot[:, 0],
            turn_one_hot[:, 0],
            turn_one_hot[:, 0],
            turn_one_hot[:, 1],
            turn_one_hot[:, 1],
            turn_one_hot[:, 1],
        ],
        dim=1,
    )  # (E,6)
    wrong_hit = wrong_hit_turn | wrong_hit_racket  # (E,6)

    # 击球犯规1: 击球的人不是击球方
    case_1 = wrong_hit_turn  # (E,6)
    case_1 = torch.stack(
        [case_1[:, :3].any(-1), case_1[:, 3:].any(-1)], dim=-1
    )  # (E,2)
    # 击球犯规2: 使用非球拍部位击球
    case_2 = wrong_hit_racket  # (E,6)
    case_2 = torch.stack(
        [case_2[:, :3].any(-1), case_2[:, 3:].any(-1)], dim=-1
    )  # (E,2)
    # 无人机越网
    case_3 = drone_hit_net  # (E,6)
    case_3 = torch.stack(
        [case_3[:, :3].any(-1), case_3[:, 3:].any(-1)], dim=-1
    )  # (E,2)
    # 球落地
    # 不用判断是哪方击的球
    case_4 = ball_hit_ground & in_half_court(ball_pos, L, W)  # (E,2)
    # 球出界
    # 需要判断是哪方击的球：1）如果存在wrong_hit，那么wrong_hit的一方就是击球方；2）如果不存在wrong_hit，那么turn就是击球方
    case_5 = (
        _not_in_bounds(ball_pos, L, W)
        & ball_hit_ground  # (E,1)
        & extended_turn_one_hot
    )  # (E,6)
    case_5 = torch.stack(
        [case_5[:, :3].any(-1), case_5[:, 3:].any(-1)], dim=-1
    )  # (E,2)
    # 球触网
    # 需要判断是哪方击的球，同上
    case_6 = ball_hit_net & extended_turn_one_hot  # (E,1)  # (E,6)
    case_6 = torch.stack(
        [case_6[:, :3].any(-1), case_6[:, 3:].any(-1)], dim=-1
    )  # (E,2)

    cases = torch.stack(
        [case_1, case_2, case_3, case_4, case_5, case_6], dim=-1
    )  # (E,2,6)

    draw = cases.any(-1).all(1, keepdim=True)  # (E,1)
    result = (cases.any(-1) & ~draw).int()  # (E,2)
    result = (result[:, 1] - result[:, 0]).unsqueeze(-1)  # (E,1)

    return draw, result, cases
