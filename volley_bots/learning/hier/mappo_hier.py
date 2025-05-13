from numbers import Number
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from torch import vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.utils import expand_right
from tensordict.nn import make_functional, TensorDictModule, TensorDictParams, TensorDictModuleBase
from torch.optim import lr_scheduler

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    MultiDiscreteTensorSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec as UnboundedTensorSpec,
)

from volley_bots.utils.torchrl.env import AgentSpec
from volley_bots.utils.tensordict import print_td_shape


LR_SCHEDULER = lr_scheduler._LRScheduler
from torchrl.modules import TanhNormal, IndependentNormal

from . import MAPPOPolicy2

import copy

class MAPPOPolicy_hier(object):

    def __init__(self, cfg, agent_spec_dict: Dict[str, AgentSpec], device="cuda") -> None:
        super().__init__()

        Opp_agent_spec = agent_spec_dict["Opp"]

        FirstPass_serve_agent_spec = agent_spec_dict["FirstPass_serve"]
        FirstPass_serve_hover_agent_spec = agent_spec_dict["FirstPass_serve_hover"]
        FirstPass_goto_agent_spec = agent_spec_dict["FirstPass_goto"]
        FirstPass_agent_spec = agent_spec_dict["FirstPass"]
        FirstPass_hover_agent_spec = agent_spec_dict["FirstPass_hover"]

        SecPass_goto_agent_spec = agent_spec_dict["SecPass_goto"]
        SecPass_agent_spec = agent_spec_dict["SecPass"]
        SecPass_hover_agent_spec = agent_spec_dict["SecPass_hover"]

        Att_goto_agent_spec = agent_spec_dict["Att_goto"]
        Att_agent_spec = agent_spec_dict["Att"]
        Att_hover_agent_spec = agent_spec_dict["Att_hover"]

        cfg.share_actor = False
        cfg.agent_name = "FirstPass_serve"
        self.policy_FirstPass_serve = MAPPOPolicy2(cfg=cfg, agent_spec=FirstPass_serve_agent_spec, device=device)
        cfg.agent_name = "FirstPass_serve_hover"
        self.policy_FirstPass_serve_hover = MAPPOPolicy2(cfg=cfg, agent_spec=FirstPass_serve_hover_agent_spec, device=device)
        cfg.agent_name = "FirstPass_goto"
        self.policy_FirstPass_goto = MAPPOPolicy2(cfg=cfg, agent_spec=FirstPass_goto_agent_spec, device=device)
        cfg.agent_name = "FirstPass"
        self.policy_FirstPass = MAPPOPolicy2(cfg=cfg, agent_spec=FirstPass_agent_spec, device=device)
        cfg.agent_name = "FirstPass_hover"
        self.policy_FirstPass_hover = MAPPOPolicy2(cfg=cfg, agent_spec=FirstPass_hover_agent_spec, device=device)

        cfg.agent_name = "SecPass_goto"
        self.policy_SecPass_goto = MAPPOPolicy2(cfg=cfg, agent_spec=SecPass_goto_agent_spec, device=device)
        cfg.agent_name = "SecPass"
        self.policy_SecPass = MAPPOPolicy2(cfg=cfg, agent_spec=SecPass_agent_spec, device=device)
        cfg.agent_name = "SecPass_hover"
        self.policy_SecPass_hover = MAPPOPolicy2(cfg=cfg, agent_spec=SecPass_hover_agent_spec, device=device)

        cfg.agent_name = "Att_goto"
        self.policy_Att_goto = MAPPOPolicy2(cfg=cfg, agent_spec=Att_goto_agent_spec, device=device)
        cfg.agent_name = "Att"
        self.policy_Att = MAPPOPolicy2(cfg=cfg, agent_spec=Att_agent_spec, device=device)
        cfg.agent_name = "Att_hover"
        self.policy_Att_hover = MAPPOPolicy2(cfg=cfg, agent_spec=Att_hover_agent_spec, device=device)
   
        cfg_Opp = copy.deepcopy(cfg)
        cfg_Opp.agent_name = "Opp"
        cfg_Opp.share_actor = True
        self.policy_Opp = MAPPOPolicy2(cfg=cfg_Opp, agent_spec=Opp_agent_spec, device=device)


    def load_state_dict(self, state_dict, player: str):
        if player == "Opp":
            self.policy_Opp.load_state_dict(state_dict)
        
        elif player == "FirstPass_serve":
            self.policy_FirstPass_serve.load_state_dict(state_dict)
        elif player == "FirstPass_serve_hover":
            self.policy_FirstPass_serve_hover.load_state_dict(state_dict)
        elif player == "FirstPass_goto":
            self.policy_FirstPass_goto.load_state_dict(state_dict)
        elif player == "FirstPass":
            self.policy_FirstPass.load_state_dict(state_dict)
        elif player == "FirstPass_hover":
            self.policy_FirstPass_hover.load_state_dict(state_dict)

        elif player == "SecPass_goto":
            self.policy_SecPass_goto.load_state_dict(state_dict)
        elif player == "SecPass":
            self.policy_SecPass.load_state_dict(state_dict)
        elif player == "SecPass_hover":
            self.policy_SecPass_hover.load_state_dict(state_dict)

        elif player == "Att_goto":
            self.policy_Att_goto.load_state_dict(state_dict)
        elif player == "Att":
            self.policy_Att.load_state_dict(state_dict)
        elif player == "Att_hover":
            self.policy_Att_hover.load_state_dict(state_dict)
        
        else:
            raise ValueError("player not found")
        
    def state_dict(self, player: str):
        if player == "Opp":
            return self.policy_Opp.state_dict()        

        elif player == "FirstPass_serve":
            return self.policy_FirstPass_serve.state_dict()
        elif player == "FirstPass_serve_hover":
            return self.policy_FirstPass_serve_hover.state_dict()
        elif player == "FirstPass_goto":
            return self.policy_FirstPass_goto.state_dict()
        elif player == "FirstPass":
            return self.policy_FirstPass.state_dict()
        elif player == "FirstPass_hover":
            return self.policy_FirstPass_hover.state_dict()
        
        elif player == "SecPass_goto":
            return self.policy_SecPass_goto.state_dict()
        elif player == "SecPass":
            return self.policy_SecPass.state_dict()
        elif player == "SecPass_hover":
            return self.policy_SecPass_hover.state_dict()

        elif player == "Att_goto":
            return self.policy_Att_goto.state_dict()
        elif player == "Att":
            return self.policy_Att.state_dict()
        elif player == "Att_hover":
            return self.policy_Att_hover.state_dict()
        
        else:
            raise ValueError("player not found")

    def train_op(self, tensordict: TensorDict):
        pass

    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        self.policy_Opp(tensordict, deterministic)

        self.policy_FirstPass_serve(tensordict, deterministic)
        self.policy_FirstPass_serve_hover(tensordict, deterministic)
        self.policy_FirstPass_goto(tensordict, deterministic)
        self.policy_FirstPass(tensordict, deterministic)
        self.policy_FirstPass_hover(tensordict, deterministic)

        self.policy_SecPass_goto(tensordict, deterministic)
        self.policy_SecPass(tensordict, deterministic)
        self.policy_SecPass_hover(tensordict, deterministic)

        self.policy_Att_goto(tensordict, deterministic)
        self.policy_Att(tensordict, deterministic)
        self.policy_Att_hover(tensordict, deterministic)

        return tensordict