from typing import Union, Sequence, Any, Dict, Tuple

import numpy as np
import torch
from torch import nn


class NoPolicyNet(nn.Module):
    """
    Policy to do nothing

    Args:
        state_shape: int or a sequence of int of the shape of state.
        action_shape: int or a sequence of int of the shape of action.
    """

    def __init__(self,
                 state_shape: Union[int, Sequence[int]],
                 action_shape: Union[int, Sequence[int]] = 0,
                 ) -> None:
        super().__init__()
        self.input_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

    def forward(self,
                obs: Union[np.ndarray, torch.Tensor],
                state: Any = None,
                info: Dict[str, Any] = {},
                ) -> Tuple[torch.Tensor, Any]:
        logits = torch.zeros(self.action_dim)

        # only the first action (do nothing) is one
        logits[0] = 1

        return logits, state


class GreedyPolicyNet(nn.Module):
    """
    Policy to always change to a new price if it is lower then the current price.

    Args:
        state_shape: int or a sequence of int of the shape of state.
        action_shape: int or a sequence of int of the shape of action.
    """

    def __init__(self,
                 state_shape: Union[int, Sequence[int]],
                 action_shape: Union[int, Sequence[int]] = 0,
                 ) -> None:
        super().__init__()
        self.input_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

    def forward(self,
                obs: Union[np.ndarray, torch.Tensor],
                state: Any = None,
                info: Dict[str, Any] = {},
                ) -> Tuple[torch.Tensor, Any]:
        logits = torch.zeros(self.action_dim)

        # if underbooking solve it with the lowest current price
        if obs[0, -2] == -1:
            logits[torch.argmin(obs[0, 1:3]) + 3] = 1
        # if overbooking solve it with the lowest future price
        elif obs[0, -2] == 1:
            logits[torch.argmin(obs[0, 3:5]) + 3] = 1
        else:
            # change if the current price is lower
            logits[torch.argmin(obs[0, 0:3])] = 1

        return logits, state
