from dataclasses import dataclass
from typing import Optional

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs import EnvBase


@dataclass
class AgentSpec:
    name: str
    n: int
    observation_key: Optional[str] = "observation"
    action_key: Optional[str] = None
    state_key: Optional[str] = None
    reward_key: Optional[str] = None
    done_key: Optional[str] = None

    _env: Optional[EnvBase] = None

    @property
    def observation_spec(self) -> TensorSpec:
        return self._env.observation_spec[self.observation_key]

    @property
    def action_spec(self) -> TensorSpec:
        if self.action_key is None:
            return self._env.action_spec
        try:
            return self._env.input_spec["full_action_spec"][self.action_key]
        except:
            return self._env.action_spec[self.action_key]

    @property
    def state_spec(self) -> TensorSpec:
        if self.state_key is None:
            raise ValueError()
        return self._env.observation_spec[self.state_key]

    @property
    def reward_spec(self) -> TensorSpec:
        if self.reward_key is None:
            return self._env.reward_spec
        try:
            return self._env.output_spec["full_reward_spec"][self.reward_key]
        except:
            return self._env.reward_spec[self.reward_key]

    @property
    def done_spec(self) -> TensorSpec:
        if self.done_key is None:
            return self._env.done_spec
        try:
            return self._env.output_spec["full_done_spec"][self.done_key]
        except:
            return self._env.done_spec[self.done_key]


# for debugging
@dataclass
class DummyEnv:
    observation_spec: CompositeSpec
    action_spec: CompositeSpec
    reward_spec: CompositeSpec
