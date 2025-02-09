import torch
from omni.isaac.core.prims import RigidPrimView
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH


class Iris_racket(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/iris_batVisual_white.usd"
    param_path: str = ASSET_PATH + "/usd/iris.yaml"
