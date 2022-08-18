from typing import List, Generator

import numpy as np
import torch
from torch import Tensor, nn

from curiosity.base import Curiosity
from envs import Converter


class NoCuriosity(Curiosity):
    """
    Placeholder class to be used when agent does not need curiosity. For example in environments that has dense reward.
    """

    # noinspection PyMissingConstructor
    def __init__(self, *args):
        pass

    def reward(self, rewards: np.ndarray, *args) -> np.ndarray:
        return rewards

    def loss(self, policy_loss: Tensor, *args) -> Tensor:
        return policy_loss

    def parameters(self) -> Generator[nn.Parameter, None, None]:
        yield from ()
