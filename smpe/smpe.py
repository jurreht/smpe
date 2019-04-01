import abc
import functools
from numbers import Number
import math
from typing import Sequence, Tuple, Union

import dask.array
import numpy as np
import scipy.optimize

from . import interpolation


NullableFloat = Union[float, None]


class DynamicGame(abc.ABC):
    n_players: int
    n_actions: Sequence[int]
    beta: Sequence[float]

    def __init__(
        self,
        n_players: int,
        n_actions: Union[Sequence[int], int],
        beta: Union[Sequence[float], float],
        cost_att: Union[Sequence[Number], Number],
        action_bounds: Sequence[Tuple[NullableFloat, NullableFloat]] = None
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
        if any(cost > 0 for cost in cost_att) and (
            not hasattr(self, 'sparse_state') or
            not callable(self.sparse_state)
        ):
            raise ValueError(
                'When the cost of attention is non-negative, a DynamicGame'
                'subclass must implement the sparse_state() method.')
        self.cost_att = cost_att

        if action_bounds is None:
            # Set default action bound
            self.action_bounds = []
            for i in range(self.n_players):
                self.action_bounds.append([(None, None)] * self.n_actions[i])
        else:
            if len(action_bounds) != self.n_players:
                raise ValueError('Provide action bounds for every player')
            for i, bounds in enumerate(action_bounds):
                if len(action_bounds[i]) != self.n_actions[i]:
                    raise ValueError(
                        f'Incorrect length of action bounds for player {i}')
            self.action_bounds = action_bounds

    def compute(
        self,
        value_functions: interpolation.MultivariateInterpolatedFunction,
        eps=1e-4,
        chunk_size=50
    ):
        if len(value_functions.vf) != self.n_players:
            raise ValueError('Provide a value function for every player')

        if (any(cost > 0 for cost in self.cost_att) and
            not isinstance(value_functions.vf[0],
                           interpolation.DifferentiableInterpolatedFunction)):
            raise ValueError(
                'Can only calculate sparsity for differentiable value'
                'functions'
            )

        # Players may have different numbers of actions. Allocate the maximum
        # number of actions amongst all of them for every player. This might
        # be wasteful when the number of actions is very asymmetric, but in the
        # more common case that every agent has the same number of actions we
        # gain a lot of effiency since we can pass around a NumPy array instead
        # of a Python list containing NumPy arrays.
        max_actions = np.max(self.n_actions)
        init_actions = dask.array.from_array(
            np.zeros((value_functions.shape[0], self.n_players * max_actions)),
            (chunk_size, -1)
        )

        # Combine the nodes and the (flattened) actions into one array.
        # A line is a combintion of a node and the actions players use in
        # that node. Combine them into one so that the node and the related
        # actions are always in the same chunk, so that we can map_blocks()
        # over it efficiently.
        nodes = dask.array.from_array(value_functions, (chunk_size, -1))
        lines = dask.array.concatenate((nodes, init_actions), axis=1)
        lines = lines.rechunk((chunk_size, -1))
        lines = lines.persist()

        converged = [False] * self.n_players
        prev_value_function = [None] * self.n_players
        prev_optimal = [False] * self.n_players
        while not all(converged):
            for i in range(self.n_players):
                lines, calc_value_function = self._inner_loop(
                    lines, i, value_functions, eps, chunk_size, prev_optimal[i]
                )

                if prev_value_function[i] is not None:
                    vf_norm = self.value_function_norm(
                        calc_value_function, prev_value_function)
                    print('outer: ', vf_norm)
                    if vf_norm <= eps:
                        converged[i] = True
                    else:
                        # The actions for one player have changed. As a
                        # result, we can no longer assume that the
                        # actions calculated for other players are optimal.
                        converged = [False] * self.n_players

                prev_optimal[i] = True

                prev_value_function[i] = calc_value_function

        lines = lines.compute()
        optimal_actions = [
            np.zeros((nodes.shape[0], self.n_actions[i]))
            for i in range(self.n_players)]
        for i in range(nodes.shape[0]):
            _, actions = self._from_line(lines[i])
            for j in range(self.n_players):
                optimal_actions[j][i] = actions[j]
        return dask.compute(*optimal_actions)

    def _to_line(self, node, actions):
        return np.concatenate((
            node, actions.flatten()
        ))

    def _from_line(self, line):
        max_n_actions = np.max(self.n_actions)
        n_actions = self.n_players * max_n_actions
        node = line[:-n_actions]
        actions = line[-n_actions:].reshape((self.n_players, max_n_actions))
        return node, actions

    def _inner_loop(
        self, lines, player_ind, value_functions, eps, chunk_size,
        prev_optimal=False
    ):
        prev_value_function = None
        while True:
            lines = self.optimal_actions(
                lines, player_ind, value_functions, chunk_size, prev_optimal)
            calc_value_function = self.calculate_value_function(
                lines, player_ind, value_functions, chunk_size, eps)
            if prev_value_function is not None:
                vf_norm = self.value_function_norm(
                    calc_value_function, prev_value_function
                )
                if vf_norm <= eps:
                    break
            prev_value_function = calc_value_function
            prev_optimal = True

        return lines, calc_value_function

    def optimal_actions(
        self, lines, player_ind, value_functions, chunk_size, prev_optimal=None
    ):
        return dask.array.core.map_blocks(
            self._optimal_action,
            lines,
            dtype=lines.dtype,
            player_ind=player_ind,
            value_function=value_functions.vf[player_ind],
            prev_optimal=None
        ).persist()

    def _optimal_action(self, lines, prev_optimal, player_ind, value_function):
        ret = np.zeros(lines.shape)
        for i in range(lines.shape[0]):
            state, actions = self._from_line(lines[i])
            actions_others = np.delete(actions, player_ind, axis=0)
            prev = None if prev_optimal is None else prev_optimal[i]
            optimal_action = self.calculate_optimal_action(
                state, prev, player_ind, value_function, actions_others
            )
            actions = np.insert(
                actions_others, player_ind, optimal_action, axis=0)
            ret[i] = self._to_line(state, actions)

        return ret

    def calculate_optimal_action(
        self, state, prev_optimal, player_ind, value_function, actions_others
    ):
        x0 = self.compute_optimization_x0(
            state, player_ind, value_function, actions_others,
            prev_optimal
        )
        bounds = self.compute_action_bounds(
            state, player_ind, value_function, actions_others)
        res = self._maximize_value(
            state, player_ind, value_function, actions_others, x0, bounds)

        if not res.success:
            # TODO
            pass
        else:
            return res.x

    def _maximize_value(
        self, state, player_ind, value_functions, actions_others, x0, bounds
    ):
        return scipy.optimize.minimize(
            functools.partial(
                self._neg_value, state, player_ind, value_functions,
                actions_others),
            x0,
            bounds=bounds
        )

    def _value(self, state, player_ind, value_function, actions):
        next_state = self.state_evolution(state, actions)
        cont_value = value_function(next_state)
        return (self.static_profits(player_ind, state, actions) +
                self.beta[player_ind] * cont_value)

    def _neg_value(self, state, player_ind, value_function, actions):
        return -1 * self._value(
            state, player_ind, value_function, actions)

    def _value_from_blocks(self, lines, player_ind, value_function):
        ret = np.zeros(lines.shape[0])
        for i in range(lines.shape[0]):
            state, actions = self._from_line(lines[i])
            ret[i] = self._value(state, player_ind, value_function, actions)
        return ret

    def calculate_value_function(
        self, lines, player_ind, value_functions, chunk_size, eps
    ):
        # Because _value() is a lot faster than calculate_optimal_action(),
        # we use a larger chunk size here.
        x_chunk = min(lines.shape[0], chunk_size * 100)
        lines = lines.rechunk((x_chunk, None)).persist()

        prev_value_function = None
        while True:
            calc_value_function = dask.array.core.map_blocks(
                self._value_from_blocks,
                lines,
                dtype=lines.dtype,
                chunks=(lines.shape[0],),
                drop_axis=1,
                player_ind=player_ind,
                value_function=value_functions.vf[player_ind],
            ).compute()

            if prev_value_function is not None:
                vf_diff = self.value_function_norm(
                    calc_value_function, prev_value_function)
                if vf_diff <= eps:
                    break
            prev_value_function = calc_value_function
            value_functions.vf[player_ind].update(calc_value_function)

        return calc_value_function

    def value_function_norm(self, new_value, old_value):
        # Convergence criterion as in Doraszelski & Pakes (2007)
        return np.max(np.abs((new_value - old_value) / (1 + new_value)))

    def compute_optimization_x0(
        self, state, player_ind, value_functions, actions_others, prev_optimal
    ):
        if prev_optimal is None:
            return np.zeros(self.n_actions[player_ind])
        else:
            return prev_optimal

    def compute_action_bounds(
        self, state, player_ind, value_functions, actions_others
    ):
        return self.action_bounds[player_ind]

    @abc.abstractmethod
    def static_profits(self, player_ind, state, actions):
        pass

    @abc.abstractmethod
    def state_evolution(self, state, actions):
        pass


class DynamicGameDifferentiable(DynamicGame):
    @abc.abstractclassmethod
    def static_profits_gradient(self, player_ind, state, actions):
        pass

    @abc.abstractclassmethod
    def state_evolution_gradient(self, state, actions):
        pass

    def compute(
        self,
        value_functions: interpolation.MultivariateInterpolatedFunction,
        *args,
        **kwargs
    ):
        if not isinstance(
            value_functions.vf[0],
            interpolation.DifferentiableInterpolatedFunction
        ):
            raise ValueError(
                'Provide a DifferentiableInterpolatedFunction when computing'
                'a DynamicGameDifferentiable')

        return super().compute(value_functions, *args, **kwargs)

    def _maximize_value(
        self, state, player_ind, value_function, actions_others, x0, bounds
    ):
        return scipy.optimize.minimize(
            lambda x: self._neg_value_deriv(
                state, player_ind, value_function,
                np.insert(actions_others, player_ind, x, axis=0)),
            x0,
            bounds=bounds,
            jac=True
        )

    def _neg_value_deriv(self, state, player_ind, value_function, actions):
        val = self._neg_value(state, player_ind, value_function, actions)
        next_state = self.state_evolution(state, actions)
        grad = -1 * self.static_profits_gradient(player_ind, state, actions)
        vf_grad = value_function.derivative(np.array(next_state))
        evol_grad = np.array(self.state_evolution_gradient(state, actions))
        grad -= self.beta[player_ind] * vf_grad @ evol_grad

        if self.n_actions[player_ind] == 1:
            # When the player has only one action, grad will be a float instead
            # of an ndarray. This creates issues in the LFBGS code so make it
            # an ndarray again,
            grad = np.array([grad])

        return val, grad
