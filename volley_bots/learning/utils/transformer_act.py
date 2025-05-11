import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


def discrete_autoregreesive_act(
    decoder, tensordict, available_actions=None, deterministic=False
):
    print("call")
    encoder_input = tensordict.select(*self.encoder_in_keys, strict=False)
    print("tensordict", tensordict)
    print("encoder_input", encoder_input)
    print("self.encoder_in_keys", self.encoder_in_keys)
    print("output", self.encoder(encoder_input))

    # 使用 Encoder 生成价值估计和编码后的特征
    values = self.encoder(encoder_input)["state_value"]
    obs_rep = self.encoder(encoder_input)["obs_rep"]
    obs = tensordict.get(self.obs_name)

    print("obs_rep", obs_rep)
    print("obs", obs)
    batch_size = encoder_input.batch_size[0]  # [env_num]

    # 初始化shifted_action，大小为(batch_size, n_agent, action_dim + 1)，其中+1是为了额外存储一个状态标志位
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(device)
    shifted_action[:, 0, 0] = 1  # 第一个智能体的初始标志位设为1
    output_action = torch.zeros(
        (batch_size, n_agent, 1), dtype=torch.long
    )  # 初始化输出动作
    output_action_log = torch.zeros_like(
        output_action, dtype=torch.float32
    )  # 初始化输出动作的log概率

    for i in range(n_agent):  # 对每个智能体进行迭代
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]  # 解码器生成logits
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = (
                -1e10
            )  # 对于无效的动作，将logit设置为-1e10

        # 使用Categorical分布根据logit选择动作
        distri = Categorical(logits=logit)
        action = (
            distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        )  # 确定性或随机选择动作
        action_log = distri.log_prob(action)  # 计算选中动作的log概率

        # 更新输出的动作和log概率
        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)

        # 更新shifted_action，为下一个智能体提供信息
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)

    # return output_action, output_action_log
    return tensordict


def discrete_parallel_act(
    decoder,
    obs_rep,
    obs,
    action,
    batch_size,
    n_agent,
    action_dim,
    device,
    available_actions=None,
):
    # 将动作转换为one-hot编码
    one_hot_action = F.one_hot(
        action.squeeze(-1), num_classes=action_dim
    )  # (batch, n_agent, action_dim)

    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(device)
    shifted_action[:, 0, 0] = 1  # 设置初始状态标志位
    shifted_action[:, 1:, 1:] = one_hot_action[
        :, :-1, :
    ]  # 设置从第1个智能体到最后一个的动作信息

    # 生成动作的logits
    logit = decoder(shifted_action, obs_rep, obs)

    if available_actions is not None:
        logit[available_actions == 0] = -1e10  # 对于无效的动作，logit设置为负值

    # 使用Categorical分布计算动作的log概率和熵
    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)  # 计算log概率
    entropy = distri.entropy().unsqueeze(-1)  # 计算熵，表示动作的随机性

    return action_log, entropy


def continuous_autoregreesive_act(decoder, tensordict, deterministic=False):

    print("continuous_autoregreesive_act")
    encoder_input = tensordict.select(*self.encoder_in_keys, strict=False)
    print("tensordict", tensordict)
    print("encoder_input", encoder_input)
    print("self.encoder_in_keys", self.encoder_in_keys)
    print("output", self.encoder(encoder_input))

    # 使用 Encoder 生成价值估计和编码后的特征
    values = self.encoder(encoder_input)["state_value"]
    obs_rep = self.encoder(encoder_input)["obs_rep"]
    obs = tensordict.get(self.obs_name)

    print("obs_rep", obs_rep)
    print("obs", obs)
    batch_size = encoder_input.batch_size[0]  # [env_num]
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(device)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):  # 对每个智能体进行迭代
        print("shifted_action", shifted_action.shape)
        print("obs_rep", obs_rep.shape)
        print("obs", obs.shape)
        act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :]  # 解码器生成动作均值
        action_std = (
            torch.sigmoid(decoder.log_std) * 0.5
        )  # 计算标准差（通过sigmoid限制）

        # 使用Normal分布生成动作
        distri = Normal(act_mean, action_std)
        action = (
            act_mean if deterministic else distri.sample()
        )  # 如果是确定性策略，直接选择均值
        action_log = distri.log_prob(action)  # 计算选中动作的log概率

        output_action[:, i, :] = action  # 存储动作
        output_action_log[:, i, :] = action_log  # 存储log概率
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = (
                action  # 更新shifted_action，为下一个智能体提供信息
            )

    # return output_action, output_action_log
    tensordict.update(("agents", "action"), output_action)
    tensordict.update("drone.action_logp", output_action_log)
    tensordict.update("state_value", values)
    print("updated_tensordict", updated_tensordict)
    return tensordict


def continuous_parallel_act(decoder, tensordict, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(device)
    shifted_action[:, 1:, :] = action[:, :-1, :]  # 从第二个智能体开始，传递动作信息

    # 使用decoder生成动作均值
    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = torch.sigmoid(decoder.log_std) * 0.5  # 计算标准差
    distri = Normal(act_mean, action_std)

    # 计算动作的log概率和熵
    action_log = distri.log_prob(action)
    entropy = distri.entropy()

    return action_log, entropy
