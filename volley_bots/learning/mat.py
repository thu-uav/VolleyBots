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
from torch.distributions import Categorical, Normal
from torch.nn import ModuleList
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


class MATPolicy(object):
    def __init__(self, cfg, agent_spec: AgentSpec, device="cuda") -> None:
        super().__init__()

        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        print("self.agent_spec.observation_spec", self.agent_spec.observation_spec)
        print("self.agent_spec.action_spec", self.agent_spec.action_spec)
        print("self.agent_spec.reward_spec", self.agent_spec.reward_spec)
        print("self.agent_spec", self.agent_spec)

        self.clip_param = cfg.clip_param
        self.ppo_epoch = int(cfg.ppo_epochs)
        self.num_minibatches = int(cfg.num_minibatches)
        self.normalize_advantages = cfg.normalize_advantages

        self.entropy_coef = cfg.entropy_coef
        self.value_loss_coef = cfg.value_loss_coef
        self.gae_gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda

        self.act_dim = self.agent_spec.action_spec.shape[-1]

        self.obs_name = ("agents", "observation")
        self.state_name = ("agents", "state")
        self.act_name = ("agents", "action")
        self.reward_name = ("agents", "reward")

        self.init_mat_networks()

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.train_in_keys = list(
            set(
                self.encoder_in_keys
                + self.encoder_out_keys
                + self.decoder_in_keys
                + self.decoder_out_keys
                + [
                    "next",
                    self.act_logps_name,
                    ("reward", self.reward_name),
                    "state_value",
                ]
                + ["progress", ("collector", "traj_ids")]
            )
        )
        # print("self.train_in_keys",self.train_in_keys)
        self.n_updates = 0

    @property
    def act_logps_name(self):
        return f"{self.agent_spec.name}.action_logp"

    def init_mat_networks(self):
        cfg = self.cfg
        n_block = self.cfg.get("n_block", 3)
        n_embd = self.cfg.get("n_embd", 256)
        n_head = self.cfg.get("n_head", 8)
        encode_state = self.cfg.get("encode_state", False)
        dec_actor = self.cfg.get("dec_actor", False)
        share_actor = self.cfg.get("share_actor", False)
        action_type = (
            "Discrete"
            if isinstance(self.agent_spec.action_spec, DiscreteTensorSpec)
            else "Continuous"
        )
        self.encoder_in_keys = [("agents", "state"), self.obs_name]
        self.encoder_out_keys = ["state_value", "obs_rep"]

        self.decoder_in_keys = [self.act_name, self.obs_name, "obs_rep"]
        self.decoder_out_keys = [
            self.act_name,
            self.act_logps_name,
            f"{self.agent_spec.name}.action_entropy",
        ]

        encoder = Encoder(
            state_dim=self.agent_spec.observation_spec.shape[-1],
            obs_dim=self.agent_spec.observation_spec.shape[-1],
            n_block=n_block,
            n_embd=n_embd,
            n_head=n_head,
            n_agent=self.agent_spec.n,
            encode_state=encode_state,
        ).to(self.device)

        self.encoder = TensorDictModule(
            encoder,
            in_keys=self.encoder_in_keys,
            out_keys=self.encoder_out_keys,
        ).to(self.device)

        if self.cfg.use_huber_loss:
            self.encoder_loss_fn = nn.HuberLoss(delta=self.cfg.huber_delta)
        else:
            self.encoder_loss_fn = nn.MSELoss()

        self.decoder_network = Decoder(
            obs_dim=self.agent_spec.observation_spec.shape[-1],
            action_dim=self.agent_spec.action_spec.shape[-1],
            n_block=n_block,
            n_embd=n_embd,
            n_head=n_head,
            n_agent=self.agent_spec.n,
            action_type=action_type,
            dec_actor=dec_actor,
            share_actor=share_actor,
        ).to(self.device)

        self.decoder = TensorDictModule(
            self.decoder_network,
            in_keys=self.decoder_in_keys,
            out_keys=self.decoder_out_keys,
        ).to(self.device)

        print("self.encoder", self.encoder)
        print("self.decoder", self.decoder)

        if cfg.get("output_dist_params", False):
            self.decoder_out_keys.append(("debug", "action_loc"))
            self.decoder_out_keys.append(("debug", "action_scale"))

        if hasattr(cfg, "value_norm") and cfg.value_norm is not None:
            cls = getattr(valuenorm, cfg.value_norm["class"])
            self.value_normalizer: valuenorm.Normalizer = cls(
                input_shape=self.agent_spec.reward_spec.shape[-2:],
                **cfg.value_norm["kwargs"],
            ).to(self.device)

    def value_op(self, tensordict: TensorDict) -> TensorDict:
        encoder_input = tensordict.select(*self.encoder_in_keys, strict=False)

        if "is_init" in encoder_input.keys():
            encoder_input["is_init"] = expand_right(
                encoder_input["is_init"], (*encoder_input.batch_size, self.agent_spec.n)
            )

        encoder_input.batch_size = [*encoder_input.batch_size, self.agent_spec.n]
        values = self.encoder(encoder_input)["state_value"]
        tensordict["state_value"] = values

        return tensordict

    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        tensordict = self.continuous_autoregreesive_act(
            self.decoder_network, tensordict, deterministic
        )
        return tensordict

    def continuous_autoregreesive_act(self, decoder, tensordict, deterministic=False):
        encoder_input = tensordict.select(*self.encoder_in_keys, strict=False)
        values = self.encoder(encoder_input)["state_value"]
        obs_rep = self.encoder(encoder_input)["obs_rep"]
        obs = tensordict.get(self.obs_name)

        batch_size = encoder_input.batch_size[0]  # [env_num]
        shifted_action = torch.zeros(
            (batch_size, self.agent_spec.n, self.agent_spec.action_spec.shape[-1])
        ).to(self.device)
        output_action = torch.zeros(
            (batch_size, self.agent_spec.n, self.agent_spec.action_spec.shape[-1]),
            dtype=torch.float32,
        )
        output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

        for i in range(self.agent_spec.n):
            act_mean = decoder(shifted_action, obs, obs_rep)[:, i, :]
            action_std = torch.sigmoid(decoder.log_std) * 0.5

            distri = Normal(act_mean, action_std)
            action = act_mean if deterministic else distri.sample()
            action_log = distri.log_prob(action)

            output_action[:, i, :] = action
            output_action_log[:, i, :] = action_log
            if i + 1 < self.agent_spec.n:
                shifted_action[:, i + 1, :] = action

        tensordict.update({"agents": {"action": output_action}})
        tensordict.update({"drone.action_logp": output_action_log})
        tensordict.update({"state_value": values})
        tensordict.update({"obs_rep": obs_rep})
        return tensordict

    def continuous_parallel_act(self, decoder, tensordict, deterministic=False):

        encoder_input = tensordict.select(*self.encoder_in_keys, strict=False)

        obs_rep = self.encoder(encoder_input)["obs_rep"]
        obs = tensordict.get(self.obs_name)
        action = tensordict.get(self.act_name)

        batch_size = encoder_input.batch_size[0]  # [env_num]
        shifted_action = torch.zeros(
            (batch_size, self.agent_spec.n, self.agent_spec.action_spec.shape[-1])
        ).to(self.device)
        shifted_action[:, 1:, :] = action[:, :-1, :]

        act_mean = decoder(shifted_action, obs, obs_rep)
        action_std = torch.sigmoid(decoder.log_std) * 0.5
        distri = Normal(act_mean, action_std)

        action_log = distri.log_prob(action)
        entropy = distri.entropy()

        return action_log, entropy

    def update_op(self, batch: TensorDict) -> Dict[str, Any]:
        obs = batch[self.obs_name]
        actions = batch[self.act_name]
        old_action_log_probs = batch[self.act_logps_name]
        advantages = batch["advantages"]
        returns = batch["returns"]
        value_preds = batch["state_value"]

        encoder_input = batch.select(*self.encoder_in_keys, strict=False)

        values = self.encoder(encoder_input)["state_value"]  # [4*E, 2, 1]

        batch_size = obs.shape[0]

        action_log_probs, dist_entropy = self.continuous_parallel_act(
            self.decoder_network, batch, deterministic=False
        )

        ratio = torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        if self.cfg._use_clipped_value_loss:
            value_pred_clipped = value_preds + (values - value_preds).clamp(
                -self.clip_param, self.clip_param
            )
            value_loss_clipped = self.encoder_loss_fn(returns, value_pred_clipped)
            value_loss_original = self.encoder_loss_fn(returns, values)
            value_loss = torch.max(value_loss_original, value_loss_clipped).mean()
        else:
            value_loss = self.encoder_loss_fn(returns, values)

        entropy_loss = -self.entropy_coef * torch.mean(dist_entropy)
        decoder_loss = policy_loss + entropy_loss
        encoder_loss = self.value_loss_coef * value_loss
        loss = encoder_loss + decoder_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()

        ess = (
            2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)
        ).exp().mean() / ratio.shape[0]

        # return policy_loss, value_loss, dist_entropy

        return {
            "encoder_loss": decoder_loss.item(),
            "decoder_loss": encoder_loss.item(),
            "total_loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "entropy": entropy_loss.item(),
            "ESS": ess.item(),
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

        tensordict = tensordict.select(*self.train_in_keys, strict=False)
        next_tensordict = tensordict["next"][:, -1]  # batchsize=[env_nums]

        with torch.no_grad():
            value_output = self.value_op(next_tensordict)  # [E,A]

        rewards = tensordict.get(("next", *self.reward_name))  # [E,T,A,1]
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)


        values = tensordict["state_value"]  # [E,T,A,1]
        next_value = value_output["state_value"].squeeze(
            0
        )

        if hasattr(self, "value_normalizer"):
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
        )  #  [E,T,A,1]

        advantages_mean = tensordict["advantages"].mean()
        advantages_std = tensordict["advantages"].std()

        if self.normalize_advantages:
            tensordict["advantages"] = (tensordict["advantages"] - advantages_mean) / (
                advantages_std + 1e-8
            )

        if hasattr(self, "value_normalizer"):
            self.value_normalizer.update(tensordict["returns"])
            tensordict["returns"] = self.value_normalizer.normalize(
                tensordict["returns"]
            )

        train_info = []

        for ppo_epoch in range(self.ppo_epoch):
            dataset = make_dataset_naive(
                tensordict,  # batchsize=[env_nums,train_every]
                int(self.cfg.num_minibatches),
                self.minibatch_seq_len if hasattr(self, "minibatch_seq_len") else 1,
            )

            for minibatch in dataset:
                train_info.append(
                    TensorDict(
                        {
                            **self.update_op(minibatch),
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
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "value_normalizer": self.value_normalizer.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.cfg.lr,
        )

        self.value_normalizer.load_state_dict(state_dict["value_normalizer"])

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def train(self):
        self.encoder.train()
        self.decoder.train()


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


import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

from .utils.transformer_act import discrete_autoregreesive_act, discrete_parallel_act

# from .utils.transformer_act import continuous_autoregreesive_act
# from .utils.transformer_act import continuous_parallel_act


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain("relu")
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head

        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))

        self.proj = init_(nn.Linear(n_embd, n_embd))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(n_agent + 1, n_agent + 1)).view(
                1, 1, n_agent + 1, n_agent + 1
            ),
        )
        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()
        k = (
            self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        )  # (B, n_head, L, head_dim)
        q = (
            self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        )
        v = (
            self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        )

        att = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(k.size(-1))
        )  # (B, n_head, L, L)
        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.proj(y)

        return y


class EncodeBlock(nn.Module):

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd)),
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd)),
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(
        self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state
    ):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state

        if self.encode_state:
            self.state_encoder = nn.Sequential(
                nn.LayerNorm(state_dim),
                init_(nn.Linear(state_dim, n_embd), activate=True),
                nn.GELU(),
            )

        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            init_(nn.Linear(obs_dim, n_embd), activate=True),
            nn.GELU(),
        )

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(
            *[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)]
        )
        self.head = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, 1)),
        )

    def forward(
        self,
        state: Union[torch.Tensor, TensorDict],
        obs: Union[torch.Tensor, TensorDict],
    ):
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        n_block,
        n_embd,
        n_head,
        n_agent,
        action_type="Continuous",
        dec_actor=False,
        share_actor=False,
    ):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type != "Discrete":
            log_std = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)

        if self.dec_actor:
            print("MAT-Dec!!!!!")
            if self.share_actor:
                self.mlp = nn.Sequential(
                    nn.LayerNorm(obs_dim),
                    init_(nn.Linear(obs_dim, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, action_dim)),
                )
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(
                        nn.LayerNorm(obs_dim),
                        init_(nn.Linear(obs_dim, n_embd), activate=True),
                        nn.GELU(),
                        nn.LayerNorm(n_embd),
                        init_(nn.Linear(n_embd, n_embd), activate=True),
                        nn.GELU(),
                        nn.LayerNorm(n_embd),
                        init_(nn.Linear(n_embd, action_dim)),
                    )
                    self.mlp.append(actor)

        else:
            if action_type == "Discrete":
                self.action_encoder = nn.Sequential(
                    init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                    nn.GELU(),
                )
            else:
                self.action_encoder = nn.Sequential(
                    init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU()
                )
            self.obs_encoder = nn.Sequential(
                nn.LayerNorm(obs_dim),
                init_(nn.Linear(obs_dim, n_embd), activate=True),
                nn.GELU(),
            )
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(
                *[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)]
            )
            self.head = nn.Sequential(
                init_(nn.Linear(n_embd, n_embd), activate=True),
                nn.GELU(),
                nn.LayerNorm(n_embd),
                init_(nn.Linear(n_embd, action_dim)),
            )

    def zero_std(self, device):
        if self.action_type != "Discrete":
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, action, obs, obs_rep):
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)
        return logit
