import abc
import functools
from numbers import Number
import math
from typing import Sequence, Union

import dask.distributed
import numpy as np
import scipy.optimize

from . import interpolation


class DynamicGame(abc.ABC):
    n_players: int
    n_actions: Sequence[int]
    beta: Sequence[float]

    def __init__(
        self,
        n_players: int,
        n_actions: Union[Sequence[int], int],
        beta: Union[Sequence[float], float],
        cost_att: Union[Sequence[Number], Number]
    ):
        if n_players < 1:
            raise ValueError('Number of players must be at least 1')
        self.n_players = n_players

        if isinstance(n_actions, int):
            if n_actions >= 1:
                n_actions = [n_actions] * n_players
            else:
                raise ValueError('Number of actions must be at least 1')
        else:
            if len(n_actions) != n_players:
                raise ValueError(
                    'Provide the number of actions for every player')
            for x in n_actions:
                if x < 1:
                    raise ValueError('Number of actions must be at least 1')
        self.n_actions = n_actions

        if isinstance(beta, float):
            if 0 < beta < 1:
                beta = [beta] * n_players
            else:
                raise ValueError('Discount factor must be in (0, 1)')
        else:
            if len(beta) != n_players:
                raise ValueError(
                    'Provide a discount factor for every player')
            for x in beta:
                if not 0 < x < 1 or math.isnan(x):
                    raise ValueError('Discount factor must be in (0, 1)')
        self.beta = beta

        if isinstance(cost_att, Number):
            if cost_att >= 0:
                cost_att = [cost_att] * n_players
            else:
                raise ValueError('Cost of attention must be non-negative')
        else:
            if len(cost_att) != n_players:
                raise ValueError(
                    'Provide a cost of attention for every player')
            for x in cost_att:
                if x < 0 or math.isnan(x):
                    raise ValueError('Cost of attention must be non-negative')
        self.cost_att = cost_att

    @abc.abstractmethod
    def static_profits(self, player_ind, state, actions):
        pass

    @abc.abstractmethod
    def state_evolution(self, state, actions):
        pass
