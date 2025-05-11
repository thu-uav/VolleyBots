import time
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictParams,
    make_functional,
)
from tensordict.utils import expand_right
from torch import vmap
from torch.optim import lr_scheduler
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    TensorSpec,
)
from torchrl.data import UnboundedContinuousTensorSpec as UnboundedTensorSpec

from volley_bots.utils.tensordict import print_td_shape
from volley_bots.utils.torchrl.env import AgentSpec

from .utils import valuenorm
from .utils.gae import compute_gae

LR_SCHEDULER = lr_scheduler._LRScheduler
import pdb

from torchrl.modules import IndependentNormal, TanhNormal


class MAPPOPolicy(object):
    def __init__(self, cfg, agent_spec: AgentSpec, device="cuda") -> None:
        super().__init__()

        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        # pdb.set_trace()

        print(self.agent_spec.observation_spec)
        print(self.agent_spec.action_spec)

        self.clip_param = cfg.clip_param
        self.ppo_epoch = int(cfg.ppo_epochs)
        self.num_minibatches = int(cfg.num_minibatches)
        self.normalize_advantages = cfg.normalize_advantages

        self.entropy_coef = cfg.entropy_coef
        self.gae_gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda

        self.act_dim = agent_spec.action_spec.shape[-1]

        self.obs_name = ("agents", "observation")
        self.state_name = ("agents", "state")
        self.act_name = ("agents", "action")
        self.reward_name = ("agents", "reward")

        self.make_actor()
        self.make_critic()

        self.train_in_keys = list(
            set(
                self.actor_in_keys
                + self.actor_out_keys
                + self.critic_in_keys
                + self.critic_out_keys
                + [
                    "next",
                    self.act_logps_name,
                    ("reward", self.reward_name),
                    "state_value",
                ]
                + ["progress", ("collector", "traj_ids")]
            )
        )

        self.n_updates = 0

    @property
    def act_logps_name(self):
        return f"{self.agent_spec.name}.action_logp"

    def make_actor(self):
        cfg = self.cfg.actor

        self.actor_in_keys = [self.obs_name, self.act_name]
        self.actor_out_keys = [
            self.act_name,
            self.act_logps_name,
            f"{self.agent_spec.name}.action_entropy",
        ]

        if cfg.get("rnn", None):
            self.actor_in_keys.extend(
                [f"{self.agent_spec.name}.actor_rnn_state", "is_init"]
            )
            self.actor_out_keys.append(f"{self.agent_spec.name}.actor_rnn_state")
            self.minibatch_seq_len = self.cfg.actor.rnn.train_seq_len
            assert self.minibatch_seq_len <= self.cfg.train_every

        if cfg.get("output_dist_params", False):
            self.actor_out_keys.append(("debug", "action_loc"))
            self.actor_out_keys.append(("debug", "action_scale"))

        create_actor_fn = lambda: TensorDictModule(
            make_ppo_actor(
                cfg, self.agent_spec.observation_spec, self.agent_spec.action_spec
            ),
            in_keys=self.actor_in_keys,
            out_keys=self.actor_out_keys,
        ).to(self.device)

        if self.cfg.share_actor:  # different agents share the same actor
            self.actor = create_actor_fn()
            self.actor_params = TensorDictParams(make_functional(self.actor))
        else:  # each agent has its own actor
            actors = nn.ModuleList(
                [create_actor_fn() for _ in range(self.agent_spec.n)]
            )
            # 创建一个 ModuleList，里面有n个actor，目的是初始化n个参数，结构实际上只用到actor[0]，加载不同的参数而已
            self.actor = actors[0]
            stacked_params = torch.stack([make_functional(actor) for actor in actors])
            self.actor_params = TensorDictParams(stacked_params.to_tensordict())

        self.actor_opt = torch.optim.Adam(self.actor_params.parameters(), lr=cfg.lr)

    def make_critic(self):
        cfg = self.cfg.critic

        if cfg.use_huber_loss:
            self.critic_loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
        else:
            self.critic_loss_fn = nn.MSELoss()

        assert self.cfg.critic_input in ("state", "obs")
        if (
            self.cfg.critic_input == "state" and self.agent_spec.state_spec is not None
        ):  # self.cfg.critic_input="state": centralized critic
            self.critic_in_keys = [self.state_name]
            self.critic_out_keys = ["state_value"]
            if cfg.get("rnn", None):
                self.critic_in_keys.extend(
                    [f"{self.agent_spec.name}.critic_rnn_state", "is_init"]
                )
                self.critic_out_keys.append(f"{self.agent_spec.name}.critic_rnn_state")
            reward_spec = self.agent_spec.reward_spec  # [E,A,1]
            # reward_spec = reward_spec.expand(
            #     self.agent_spec.n, *reward_spec.shape
            # )  # [A,E,A,1]
            critic = make_critic(
                cfg, self.agent_spec.state_spec, reward_spec, centralized=True
            )
            self.critic = TensorDictModule(
                critic,
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = self.critic
        else:  # self.cfg.critic_input="obs": shared decentralized critic
            self.critic_in_keys = [self.obs_name]
            self.critic_out_keys = ["state_value"]
            if cfg.get("rnn", None):
                self.critic_in_keys.extend(
                    [f"{self.agent_spec.name}.critic_rnn_state", "is_init"]
                )
                self.critic_out_keys.append(f"{self.agent_spec.name}.critic_rnn_state")
            critic = make_critic(
                cfg,
                self.agent_spec.observation_spec,  # [4096, 2, 37]
                self.agent_spec.reward_spec,  # [4096, 2, 1]
                centralized=False,  # shared decentralized critic
            )
            self.critic = TensorDictModule(
                critic,
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = self.critic
            # self.value_func = vmap(self.critic, in_dims=1, out_dims=1) # 忽略 agents 那维，最后再拼起来
            # import pdb; pdb.set_trace()

        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = cfg.lr_scheduler
        if scheduler is not None:
            scheduler = eval(scheduler)
            self.critic_opt_scheduler: LR_SCHEDULER = scheduler(
                self.critic_opt, **cfg.lr_scheduler_kwargs
            )

        if hasattr(cfg, "value_norm") and cfg.value_norm is not None:
            # The original MAPPO implementation uses ValueNorm1 with a very large beta,
            # and normalizes advantages at batch level.
            # Tianshou (https://github.com/thu-ml/tianshou) uses ValueNorm2 with subtract_mean=False,
            # and normalizes advantages at mini-batch level.
            # Empirically the performance is similar on most of the tasks.
            cls = getattr(valuenorm, cfg.value_norm["class"])
            self.value_normalizer: valuenorm.Normalizer = cls(
                input_shape=self.agent_spec.reward_spec.shape[-2:],
                **cfg.value_norm["kwargs"],
            ).to(self.device)

    def value_op(self, tensordict: TensorDict) -> TensorDict:
        critic_input = tensordict.select(*self.critic_in_keys, strict=False)
        if "is_init" in critic_input.keys():
            critic_input["is_init"] = expand_right(
                critic_input["is_init"], (*critic_input.batch_size, self.agent_spec.n)
            )
        if self.cfg.critic_input == "obs":
            critic_input.batch_size = [*critic_input.batch_size, self.agent_spec.n]
        elif (
            "is_init" in critic_input.keys() and critic_input["is_init"].shape[-1] != 1
        ):
            critic_input["is_init"] = critic_input["is_init"].all(-1, keepdim=True)
        # import pdb; pdb.set_trace()
        tensordict = self.value_func(critic_input)
        return tensordict

    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        actor_input = tensordict.select(*self.actor_in_keys, strict=False)
        if "is_init" in actor_input.keys():
            actor_input["is_init"] = expand_right(
                actor_input["is_init"], (*actor_input.batch_size, self.agent_spec.n)
            )
        actor_input.batch_size = [
            *actor_input.batch_size,
            self.agent_spec.n,
        ]  # [env_num, drone_num]
        if self.cfg.share_actor:
            actor_output = self.actor(
                actor_input, self.actor_params, deterministic=deterministic
            )
        else:
            actor_output = vmap(
                self.actor, in_dims=(1, 0), out_dims=1, randomness="different"
            )(actor_input, self.actor_params, deterministic=deterministic)
        # action_output [E,A]

        tensordict.update(actor_output)
        tensordict.update(self.value_op(tensordict))
        return tensordict

    def update_actor(self, batch: TensorDict) -> Dict[str, Any]:
        advantages = batch["advantages"]
        actor_input = batch.select(*self.actor_in_keys)
        if "is_init" in actor_input.keys():
            actor_input["is_init"] = expand_right(
                actor_input["is_init"], (*actor_input.batch_size, self.agent_spec.n)
            )
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]

        log_probs_old = batch[self.act_logps_name]
        if hasattr(self, "minibatch_seq_len"):  # [N, T, A, *]
            actor_output = vmap(self.actor, in_dims=(2, 0), out_dims=2)(
                actor_input, self.actor_params, eval_action=True
            )
        else:  # [N, A, *]
            if self.cfg.share_actor:
                actor_output = self.actor(
                    actor_input, self.actor_params, eval_action=True
                )
            else:
                actor_output = vmap(self.actor, in_dims=(1, 0), out_dims=1)(
                    actor_input, self.actor_params, eval_action=True
                )

            # actor_output [N, A]

        log_probs_new = actor_output[self.act_logps_name]

        dist_entropy = actor_output[f"{self.agent_spec.name}.action_entropy"]

        assert advantages.shape == log_probs_new.shape == dist_entropy.shape

        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        policy_loss = -torch.mean(torch.min(surr1, surr2) * self.act_dim)
        entropy_loss = -torch.mean(dist_entropy)

        self.actor_opt.zero_grad()
        # pdb.set_trace()
        (
            policy_loss + entropy_loss * self.cfg.entropy_coef
        ).backward()  # combine policy loss and entropy loss as the total loss
        grad_norm = torch.nn.utils.clip_grad_norm_(  # clip the gradient
            self.actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
        )
        self.actor_opt.step()

        ess = (
            2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)
        ).exp().mean() / ratio.shape[0]
        return {
            "policy_loss": policy_loss.item(),
            "actor_grad_norm": grad_norm.item(),
            "entropy": -entropy_loss.item(),
            "ESS": ess.item(),
        }

    def update_critic(self, batch: TensorDict) -> Dict[str, Any]:
        critic_input = batch.select(*self.critic_in_keys)
        values = self.value_op(critic_input)["state_value"]
        b_values = batch["state_value"]
        b_returns = batch["returns"]
        assert values.shape == b_values.shape == b_returns.shape
        value_pred_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )

        value_loss_clipped = self.critic_loss_fn(b_returns, value_pred_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)

        value_loss = torch.max(
            value_loss_original, value_loss_clipped
        )  # clip the value loss

        value_loss.backward()  # do not multiply weights here
        grad_norm = nn.utils.clip_grad_norm_(  # clip the gradient
            self.critic.parameters(), self.cfg.max_grad_norm
        )
        self.critic_opt.step()
        self.critic_opt.zero_grad(set_to_none=True)
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return {
            "value_loss": value_loss.mean(),
            "critic_grad_norm": grad_norm.item(),
            "explained_var": explained_var.item(),
        }

    def _get_dones(self, tensordict: TensorDict):
        env_done = tensordict[("next", "done")].unsqueeze(-1)
        agent_done = tensordict.get(
            ("next", f"{self.agent_spec.name}.done"),
            env_done.expand(*env_done.shape[:-2], self.agent_spec.n, 1),
        )
        done = agent_done | env_done
        return done

    def train_op(self, tensordict: TensorDict):
        """
        Training procedure.
        input:
            tensordict: batchsize=[env_nums, train_every]
        """
        tensordict = tensordict.select(*self.train_in_keys, strict=False)
        next_tensordict = tensordict["next"][:, -1]
        with torch.no_grad():
            value_output = self.value_op(next_tensordict)

        rewards = tensordict.get(("next", *self.reward_name))  # [E,T,A,1]
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)

        values = tensordict["state_value"]
        next_value = value_output["state_value"].squeeze(0)

        if hasattr(
            self, "value_normalizer"
        ):  # denormalize the normalized values to compute GAE
            values = self.value_normalizer.denormalize(values)
            next_value = self.value_normalizer.denormalize(next_value)

        dones = self._get_dones(tensordict)

        tensordict["advantages"], tensordict["returns"] = compute_gae(
            rewards,
            dones,
            values,
            next_value,
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
        )

        advantages_mean = tensordict["advantages"].mean()
        advantages_std = tensordict["advantages"].std()
        if self.normalize_advantages:  # normalize the advantages
            tensordict["advantages"] = (tensordict["advantages"] - advantages_mean) / (
                advantages_std + 1e-8
            )

        if hasattr(self, "value_normalizer"):
            self.value_normalizer.update(
                tensordict["returns"]
            )  # compute the running mean and variance of the returns
            tensordict["returns"] = (
                self.value_normalizer.normalize(  # normalize the returns
                    tensordict["returns"]
                )
            )

        train_info = []
        for ppo_epoch in range(
            self.ppo_epoch
        ):  # update the actor and critic for ppo_epoch times
            dataset = (
                make_dataset_naive(  # make shuffled minibatch dataset for training
                    tensordict,
                    int(self.cfg.num_minibatches),
                    self.minibatch_seq_len if hasattr(self, "minibatch_seq_len") else 1,
                )
            )
            for minibatch in dataset:
                train_info.append(
                    TensorDict(
                        {
                            **self.update_actor(minibatch),
                            **self.update_critic(minibatch),
                        },
                        batch_size=[],
                    )
                )

        train_info = {k: v.mean().item() for k, v in torch.stack(train_info).items()}
        train_info["advantages_mean"] = advantages_mean.item()
        train_info["advantages_std"] = advantages_std.item()
        if isinstance(
            self.agent_spec.action_spec, (BoundedTensorSpec, UnboundedTensorSpec)
        ):
            train_info["action_norm"] = (
                tensordict[self.act_name].norm(dim=-1).mean().item()
            )
        if hasattr(self, "value_normalizer"):
            train_info["value_running_mean"] = (
                self.value_normalizer.running_mean.mean().item()
            )

        self.n_updates += 1
        return {f"{self.agent_spec.name}/{k}": v for k, v in train_info.items()}

    def state_dict(self):
        state_dict = {
            "critic": self.critic.state_dict(),
            "actor_params": self.actor_params,
            "value_normalizer": self.value_normalizer.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.actor_params = state_dict["actor_params"]
        self.actor_opt = torch.optim.Adam(
            self.actor_params.parameters(), lr=self.cfg.actor.lr
        )
        self.critic.load_state_dict(state_dict["critic"])
        self.value_normalizer.load_state_dict(state_dict["value_normalizer"])

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()


def make_dataset_naive(
    tensordict: TensorDict, num_minibatches: int = 4, seq_len: int = 1
):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]


import torch.distributions as D

from .common import make_encoder
from .modules.distributions import (
    CustomDiagGaussian,
    MultiCategoricalModule,
    TanhIndependentNormalModule,
    TanhNormalWithEntropy,
)
from .modules.rnn import GRU


class TanhThrustDistribution(torch.distributions.Distribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        assert loc.shape == scale.shape and len(loc.shape) == 3
        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[-1:]
        self.dist_rate = D.Independent(D.Normal(loc[..., :-1], scale[..., :-1]), 1)
        self.dist_thrust = TanhNormalWithEntropy(
            loc[..., [-1]], scale[..., [-1]], tanh_loc=True
        )
        super().__init__(batch_shape, event_shape)

    def rsample(self, sample_shape=None):
        sample_rate = self.dist_rate.rsample(sample_shape)
        sample_thrust = self.dist_thrust.rsample(sample_shape)
        return torch.concat([sample_rate, sample_thrust], dim=-1)

    def log_prob(self, value):
        return self.dist_rate.log_prob(value[..., :-1]) + self.dist_thrust.log_prob(
            value[..., [-1]]
        )

    def entropy(self):
        return self.dist_rate.entropy() + self.dist_thrust.entropy()

    @property
    def mode(self):
        mode_rate = self.dist_rate.mode
        mode_thrust = self.dist_thrust.mode
        return torch.concat([mode_rate, mode_thrust], dim=-1)


class TanhThrustDistribution2(torch.distributions.Distribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        assert loc.shape == scale.shape and len(loc.shape) == 3
        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[-1:]
        self.dist_rate = D.Independent(D.Normal(loc[..., :-1], scale[..., :-1]), 1)
        self.dist_thrust = TanhNormalWithEntropy(
            torch.tanh(loc[..., [-1]]), scale[..., [-1]], tanh_loc=True
        )
        super().__init__(batch_shape, event_shape)

    def rsample(self, sample_shape=None):
        sample_rate = self.dist_rate.rsample(sample_shape)
        sample_thrust = self.dist_thrust.rsample(sample_shape)
        return torch.concat([sample_rate, sample_thrust], dim=-1)

    def log_prob(self, value):
        return self.dist_rate.log_prob(value[..., :-1]) + self.dist_thrust.log_prob(
            value[..., [-1]]
        )

    def entropy(self):
        return self.dist_rate.entropy() + self.dist_thrust.entropy()

    @property
    def mode(self):
        mode_rate = self.dist_rate.mode
        mode_thrust = self.dist_thrust.mode
        return torch.concat([mode_rate, mode_thrust], dim=-1)


class IndependentTanhDistribution(TanhNormalWithEntropy):
    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ):
        min = torch.tensor([-20.0, -20.0, -50.0, -1.0], device=loc.device)
        max = torch.tensor([20.0, 20.0, 50.0, 1.0], device=loc.device)
        super().__init__(loc, scale, min=min, max=max, tanh_loc=True)


def make_ppo_actor(cfg, observation_spec: TensorSpec, action_spec: TensorSpec):
    """
    Create an actor network for PPO (control only one agent).
    """

    encoder = make_encoder(cfg, observation_spec)

    if isinstance(action_spec, MultiDiscreteTensorSpec):
        act_dist = MultiCategoricalModule(
            encoder.output_shape.numel(),
            torch.as_tensor(action_spec.nvec.storage().float()).long(),
        )
    elif isinstance(action_spec, DiscreteTensorSpec):
        act_dist = MultiCategoricalModule(
            encoder.output_shape.numel(), [action_spec.space.n]
        )
    elif isinstance(action_spec, (UnboundedTensorSpec, BoundedTensorSpec)):
        action_dim = action_spec.shape[-1]

        create_dist_func = cfg.get("create_dist_func", "default")
        if create_dist_func == "default":
            create_dist_func = None
        elif create_dist_func == "tanh":
            create_dist_func = lambda loc, scale: TanhNormalWithEntropy(
                loc, scale, tanh_loc=True
            )

        elif create_dist_func == "tanh_loc_rescale":

            def create_dist_func(loc: torch.Tensor, scale: torch.Tensor):
                loc = torch.tanh(loc)
                return TanhNormalWithEntropy(loc, scale)

        elif create_dist_func == "tanh_3.0":

            def create_dist_func(loc, scale):
                return TanhNormalWithEntropy(
                    loc, scale, min=-3.0, max=3.0, tanh_loc=True
                )

        elif create_dist_func == "tanh_thrust":

            def create_dist_func(loc, scale):
                return TanhThrustDistribution(loc, scale)

        elif create_dist_func == "tanh_thrust_2":

            def create_dist_func(loc, scale):
                return TanhThrustDistribution(loc, scale)

        elif create_dist_func == "independent_tanh":

            def create_dist_func(loc, scale):
                return IndependentTanhDistribution(loc, scale)

        else:
            raise NotImplementedError()

        act_dist = CustomDiagGaussian(
            encoder.output_shape.numel(),
            action_dim,
            False,
            0.01,
            create_dist_func=create_dist_func,
        )
    else:
        raise NotImplementedError(action_spec)

    if cfg.get("rnn", None):
        rnn_cls = {"gru": GRU}[cfg.rnn.cls.lower()]
        rnn = rnn_cls(input_size=encoder.output_shape.numel(), **cfg.rnn.kwargs)
    else:
        rnn = None

    return Actor(
        encoder, act_dist, rnn, output_dist_params=cfg.get("output_dist_params", False)
    )


def make_critic(
    cfg, state_spec: TensorSpec, reward_spec: TensorSpec, centralized=False
):
    assert isinstance(reward_spec, (UnboundedTensorSpec, BoundedTensorSpec))
    encoder = make_encoder(
        cfg, state_spec
    )  # 根据最后一个维度进行MLP，前几个维度会自动flatten

    if cfg.get("rnn", None):
        rnn_cls = {"gru": GRU}[cfg.rnn.cls.lower()]
        rnn = rnn_cls(input_size=encoder.output_shape.numel(), **cfg.rnn.kwargs)
    else:
        rnn = None

    if centralized:  # centralized
        v_out = nn.Linear(
            encoder.output_shape.numel(), reward_spec.shape[-2:].numel()
        )  # reward_spec.shape[-2:].numel(): 2
        nn.init.orthogonal_(v_out.weight, cfg.gain)
        return Critic(
            encoder, rnn, v_out, reward_spec.shape[-2:]
        )  # reward_spec.shape[-2:]: [2, 1]
    else:  # decentralized
        v_out = nn.Linear(
            encoder.output_shape.numel(), reward_spec.shape[-1]
        )  # reward_spec.shape[-1]: 1
        nn.init.orthogonal_(v_out.weight, cfg.gain)
        return Critic(
            encoder, rnn, v_out, reward_spec.shape[-1:]
        )  # reward_spec.shape[-1]: [1]


def _is_independent_normal(dist: torch.distributions.Distribution) -> bool:
    return isinstance(dist, torch.distributions.Independent) and isinstance(
        dist.base_dist, torch.distributions.Normal
    )


class Actor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        act_dist: nn.Module,
        rnn: Optional[nn.Module] = None,
        output_dist_params: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.act_dist = act_dist
        self.rnn = rnn
        self.output_dist_params = output_dist_params

    def forward(
        self,
        obs: Union[torch.Tensor, TensorDict],
        action: torch.Tensor = None,
        rnn_state=None,
        is_init=None,
        deterministic=False,
        eval_action=False,
    ):
        actor_features = self.encoder(obs)
        if self.rnn is not None:
            actor_features, rnn_state = self.rnn(actor_features, rnn_state, is_init)
        else:
            rnn_state = None
        action_dist: torch.distributions.Distribution = self.act_dist(actor_features)
        # print(f"action_dist.batch_shape:{action_dist.batch_shape}")  # [E,]

        if self.output_dist_params and _is_independent_normal(action_dist):
            loc = action_dist.base_dist.loc
            scale = action_dist.base_dist.scale
        else:
            loc = None
            scale = None

        if eval_action:
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)  # (*,A,1)
            dist_entropy = action_dist.entropy().unsqueeze(-1)  # (*,A,1)
            if self.rnn is None:
                return action, action_log_probs, dist_entropy, loc, scale
            else:
                return action, action_log_probs, dist_entropy, None, loc, scale
        else:
            if deterministic:
                action = action_dist.mode
            else:
                action = action_dist.sample()
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            dist_entropy = action_dist.entropy().unsqueeze(-1)
            if self.rnn is None:
                return action, action_log_probs, dist_entropy, loc, scale
            else:
                return (
                    action,
                    action_log_probs,
                    dist_entropy,
                    None,
                    rnn_state,
                    loc,
                    scale,
                )


class Critic(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        rnn: nn.Module,
        v_out: nn.Module,
        output_shape: torch.Size = torch.Size((-1,)),
    ):
        super().__init__()
        self.base = base
        self.rnn = rnn
        self.v_out = v_out
        self.output_shape = output_shape

    def forward(
        self,
        critic_input: torch.Tensor,
        rnn_state: torch.Tensor = None,
        is_init: torch.Tensor = None,
    ):
        critic_features = self.base(critic_input)
        if self.rnn is not None:
            critic_features, rnn_state = self.rnn(critic_features, rnn_state, is_init)
        else:
            rnn_state = None

        values = self.v_out(critic_features)

        if len(self.output_shape) > 1:
            values = values.unflatten(-1, self.output_shape)
        return values, rnn_state
