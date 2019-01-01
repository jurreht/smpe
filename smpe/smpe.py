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
        if len(value_functions) != self.n_players:
            raise ValueError('Provide a value function for every player')

        dask_nodes = dask.array.from_array(
            value_functions.numpy_nodes(),
            (chunk_size, self._dim_state(value_functions)))
        dask_nodes = dask_nodes.persist()

        converged = [False] * self.n_players
        prev_value_function = [None] * self.n_players
        optimal_actions = [None] * self.n_players
        while not all(converged):
            for i in range(self.n_players):
                actions_others = (optimal_actions[:i] +
                                  optimal_actions[i + 1:])
                optimal_actions[i], calc_value_function = self._inner_loop(
                    dask_nodes, i, value_functions, actions_others, eps,
                    chunk_size, optimal_actions[i]
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

                prev_value_function[i] = calc_value_function

        return dask.compute(*optimal_actions)

    def _dim_state(self, value_functions):
        for x in value_functions.nodes():
            return len(x)

    def _inner_loop(
        self, dask_nodes, player_ind, value_functions, actions_others, eps,
        chunk_size, prev_optimal=None
    ):
        prev_value_function = None
        while True:
            optimal_actions = self.optimal_actions(
                dask_nodes, player_ind, value_functions, actions_others,
                chunk_size, prev_optimal)
            calc_value_function = self.calculate_value_function(
                dask_nodes, player_ind, value_functions, optimal_actions,
                actions_others, eps)
            if prev_value_function is not None:
                vf_norm = self.value_function_norm(
                    calc_value_function, prev_value_function
                )
                if vf_norm <= eps:
                    break
            prev_value_function = calc_value_function
            prev_optimal = optimal_actions

        return optimal_actions, calc_value_function

    def optimal_actions(
        self, dask_nodes, player_ind, value_functions, actions_others,
        chunk_size, prev_optimal=None
    ):
        if prev_optimal is None:
            return dask_nodes.map_blocks(
                self.calculate_optimal_action,
                dtype=np.float_,
                chunks=(chunk_size, self.n_actions[player_ind]),
                player_ind=player_ind,
                value_functions=value_functions,
                actions_others=actions_others,
                prev_optimal=None
            ).persist()
        else:
            return dask.array.core.map_blocks(
                self.calculate_optimal_action,
                dask_nodes,
                prev_optimal,
                dtype=np.float_,
                chunks=(chunk_size, self.n_actions[player_ind]),
                player_ind=player_ind,
                value_functions=value_functions,
                actions_others=actions_others
            ).persist()

    def calculate_optimal_action(
        self, states, prev_optimal, player_ind, value_functions, actions_others
    ):
        ret = np.zeros((states.shape[0], self.n_actions[player_ind]))
        for i in range(states.shape[0]):
            state = states[i]
            prev = None if prev_optimal is None else prev_optimal[i]
            x0 = self.compute_optimization_x0(
                state, player_ind, value_functions, actions_others,
                prev
            )
            bounds = self.compute_action_bounds(
                state, player_ind, value_functions, actions_others)
            res = self._maximize_value(
                state, player_ind, value_functions, actions_others, x0, bounds)

            if not res.success:
                # TODO
                pass
            else:
                ret[i] = res.x

        return ret

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

    def _actions(self, player_ind, actions_player, actions_others):
        if isinstance(actions_player, np.ndarray):
            actions_player = actions_player.tolist()
        return (actions_others[:player_ind] +
                [actions_player] +
                actions_others[player_ind:])

    def _value(
        self, state, player_ind, value_functions,
        actions_player, actions_others
    ):
        actions = self._actions(player_ind, actions_player, actions_others)
        next_state = self.state_evolution(state, actions)
        cont_value = value_functions[player_ind](next_state)
        return (self.static_profits(player_ind, state, actions) +
                self.beta[player_ind] * cont_value)

    def _neg_value(
        self, state, player_ind, value_functions, actions_others,
        actions_player
    ):
        return -1 * self._value(
            state, player_ind, value_functions, actions_player,
            actions_others)

    def _value_from_blocks(
        self, states, actions_player, player_ind, value_functions,
        actions_others
    ):
        ret = np.zeros((states.shape[0], 1))
        for i in range(states.shape[0]):
            ret[i] = self._value(
                states[i], player_ind, value_functions, actions_player[i],
                actions_others)
        return ret

    def calculate_value_function(
        self, dask_nodes, player_ind, value_functions, actions_player,
        actions_others, eps
    ):
        # Because _value() is a lot faster than calculate_optimal_action(),
        # we use a larger chunk size here.
        x_chunk = min(dask_nodes.shape[0], 1000)
        actions_player = actions_player.rechunk((x_chunk, -1)).persist()
        dask_nodes = dask_nodes.rechunk((x_chunk, -1)).persist()

        prev_value_function = None
        while True:
            calc_value_function = dask.array.core.map_blocks(
                self._value_from_blocks,
                dask_nodes,
                actions_player,
                dtype=np.float_,
                chunks=(x_chunk, 1),
                player_ind=player_ind,
                value_functions=value_functions,
                actions_others=actions_others
            ).compute()
            # map_blocks() adds an extra dimension which .update() below
            # cannot swallow => remove it
            calc_value_function = calc_value_function[:, 0]

            if prev_value_function is not None:
                vf_diff = self.value_function_norm(
                    calc_value_function, prev_value_function)
                if vf_diff <= eps:
                    break
            prev_value_function = calc_value_function
            value_functions[player_ind].update(calc_value_function)

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
            value_functions[0],
            interpolation.DifferentiableInterpolatedFunction
        ):
            raise ValueError(
                'Provide a DifferentiableInterpolatedFunction when computing'
                'a DynamicGameDifferentiable')

        return super().compute(value_functions, *args, **kwargs)

    def _maximize_value(
        self, state, player_ind, value_functions, actions_others, x0, bounds
    ):
        return scipy.optimize.minimize(
            functools.partial(
                self._neg_value_deriv, state, player_ind, value_functions,
                actions_others),
            x0,
            bounds=bounds,
            jac=True
        )

    def _neg_value_deriv(
        self, state, player_ind, value_functions, actions_others,
        actions_player
    ):
        val = self._neg_value(
            state, player_ind, value_functions, actions_others,
            actions_player)
        actions = self._actions(player_ind, actions_player, actions_others)
        next_state = self.state_evolution(state, actions)
        grad = -1 * self.static_profits_gradient(player_ind, state, actions)
        vf_grad = value_functions[player_ind].derivative(np.array(next_state))
        evol_grad = np.array(self.state_evolution_gradient(state, actions))
        grad -= self.beta[player_ind] * vf_grad @ evol_grad

        if self.n_actions[player_ind] == 1:
            # When the player has only one action, grad will be a float instead
            # of an ndarray. This creates issues in the LFBGS code so make it
            # an ndarray again,
            grad = np.array([grad])

        return val, grad
