from collections import defaultdict

import torch
from tensordict import TensorDict

from .mappo import MAPPOPolicy, make_dataset_naive
from .utils.gae import compute_gae


class HAPPOPolicy(MAPPOPolicy):

    def update_actor(self, batch: TensorDict, factor: torch.Tensor, agent_id: int):
        advantages = batch["advantages"]
        actor_input = batch.select(*self.actor_in_keys)
        actor_params = self.actor_params[agent_id]

        log_probs_old = batch[self.act_logps_name]
        actor_output = self.actor(actor_input, actor_params, eval_action=True)

        log_probs_new = actor_output[self.act_logps_name]
        dist_entropy = actor_output[f"{self.agent_spec.name}.action_entropy"]

        assert advantages.shape == log_probs_new.shape == dist_entropy.shape

        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        policy_loss = -torch.mean(factor * torch.min(surr1, surr2) * self.act_dim)
        entropy_loss = -torch.mean(dist_entropy)

        self.actor_opt.zero_grad()
        (policy_loss - entropy_loss * self.cfg.entropy_coef).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
        )
        self.actor_opt.step()

        ess = (
            2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)
        ).exp().mean() / ratio.shape[0]
        info = TensorDict(
            {
                "policy_loss": policy_loss.item(),
                "actor_grad_norm": grad_norm.item(),
                "entropy": -entropy_loss.item(),
                "ESS": ess.item(),
            },
            [],
        )
        return info, factor * ratio.detach()

    def train_op(self, tensordict: TensorDict):
        # 从输入的 `tensordict` 中选择需要的键以构建训练所需的数据，`strict=False` 表示未找到某些键不会报错。
        tensordict = tensordict.select(*self.train_in_keys, strict=False)

        # 提取下一时刻的最后一个数据点，`next_tensordict` 包含用于计算目标值的信息。
        next_tensordict = tensordict["next"][:, -1]

        # 禁用梯度计算，利用值网络预测下一状态的状态值。
        with torch.no_grad():
            value_output = self.value_op(next_tensordict)

        # 获取奖励数据，支持多种 reward 键的命名方式。
        rewards = tensordict.get(("next", *self.reward_name))  # [E, T, A, 1]
        if rewards.shape[-1] != 1:
            # 如果奖励是多维度的，按最后一维求和（例如多个奖励来源）。
            rewards = rewards.sum(-1, keepdim=True)

        # 提取当前状态的值函数和下一状态的值函数。
        values = tensordict["state_value"]
        next_value = value_output["state_value"].squeeze(0)

        # 如果存在值函数标准化器，对值进行去标准化处理。
        if hasattr(self, "value_normalizer"):
            values = self.value_normalizer.denormalize(values)
            next_value = self.value_normalizer.denormalize(next_value)

        # 获取终止状态标志，表明哪些时间步为终止状态。
        dones = self._get_dones(tensordict)

        # 使用 GAE（Generalized Advantage Estimation）计算优势函数和回报。
        tensordict["advantages"], tensordict["returns"] = compute_gae(
            rewards,  # 奖励序列
            dones,  # 终止标志
            values,  # 当前状态值
            next_value,  # 下一状态值
            gamma=self.gae_gamma,  # 折扣因子
            lmbda=self.gae_lambda,  # GAE 的 lambda 参数
        )

        # 计算优势函数的均值和标准差，用于标准化。
        advantages_mean = tensordict["advantages"].mean()
        advantages_std = tensordict["advantages"].std()

        # 如果启用了标准化优势，将优势函数标准化为零均值和单位方差。
        if self.normalize_advantages:
            tensordict["advantages"] = (tensordict["advantages"] - advantages_mean) / (
                advantages_std + 1e-8
            )

        # 如果存在值函数标准化器，更新标准化器并对回报进行标准化。
        if hasattr(self, "value_normalizer"):
            self.value_normalizer.update(tensordict["returns"])
            tensordict["returns"] = self.value_normalizer.normalize(
                tensordict["returns"]
            )

        # 存储训练过程中的信息。
        train_info = []

        # 进行多次 PPO 迭代（例如多回合更新）。
        for ppo_epoch in range(self.ppo_epoch):
            # 创建小批量数据集，用于每次迭代的梯度计算。
            dataset = make_dataset_naive(
                tensordict,
                int(self.cfg.num_minibatches),  # 小批量数
                (
                    self.minibatch_seq_len if hasattr(self, "minibatch_seq_len") else 1
                ),  # 时间序列长度
            )
            # 逐个小批量更新actor和critic
            for minibatch in dataset:
                # factor对应论文中的M的系数:修正因子，初始化因子为全 1
                factor = torch.ones(
                    minibatch[self.act_logps_name].shape[0], 1, device=minibatch.device
                )

                actor_batch = minibatch.select(
                    *self.actor_in_keys, "advantages", self.act_logps_name
                )
                actor_batch.batch_size = [*minibatch.shape, self.agent_spec.n]
                critic_batch = minibatch.select(
                    *self.critic_in_keys, "returns", "state_value"
                )

                agent_info = []

                for agent_id in torch.randperm(self.agent_spec.n):
                    # 更新agent_id 策略
                    info, factor = self.update_actor(
                        actor_batch[:, agent_id], factor, agent_id.item()
                    )
                    agent_info.append(info)

                # 将所有agent的训练信息合并，并更新 critic 的训练信息
                train_info.append(
                    TensorDict(
                        {
                            **torch.stack(agent_info).apply(torch.mean, batch_size=[]),
                            **self.update_critic(critic_batch),
                        },
                        [],
                    )
                )

        # 对所有训练信息进行统计，取均值
        train_info = {k: v.mean().item() for k, v in torch.stack(train_info).items()}
        train_info["advantages_mean"] = advantages_mean
        train_info["advantages_std"] = advantages_std

        # 记录动作的范数信息。
        train_info["action_norm"] = (
            tensordict[self.act_name].float().norm(dim=-1).mean()
        )

        # 如果有值归一化器，记录其运行均值
        if hasattr(self, "value_normalizer"):
            train_info["value_running_mean"] = self.value_normalizer.running_mean.mean()

        # 返回训练信息，包含agent名称的前缀
        return {f"{self.agent_spec.name}/{k}": v for k, v in train_info.items()}
