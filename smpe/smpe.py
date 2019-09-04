import abc
import functools
import logging
from numbers import Number
import math
from typing import Sequence, Tuple, Union

import dask.array
import numpy as np
import scipy.optimize

from . import interpolation


NullableFloat = Union[float, None]


class ComputationLimitError(Exception):
    pass


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
        action_bounds: Sequence[Tuple[NullableFloat, NullableFloat]] = None,
    ):
        if n_players < 1:
            raise ValueError("Number of players must be at least 1")
        self.n_players = n_players

        if isinstance(n_actions, int):
            if n_actions >= 1:
                n_actions = [n_actions] * n_players
            else:
                raise ValueError("Number of actions must be at least 1")
        else:
            if len(n_actions) != n_players:
                raise ValueError("Provide the number of actions for every player")
            for x in n_actions:
                if x < 1:
                    raise ValueError("Number of actions must be at least 1")
        self.n_actions = n_actions

        if isinstance(beta, float):
            if 0 < beta < 1:
                beta = [beta] * n_players
            else:
                raise ValueError("Discount factor must be in (0, 1)")
        else:
            if len(beta) != n_players:
                raise ValueError("Provide a discount factor for every player")
            for x in beta:
                if not 0 < x < 1 or math.isnan(x):
                    raise ValueError("Discount factor must be in (0, 1)")
        self.beta = beta

        if isinstance(cost_att, Number):
            if cost_att >= 0:
                cost_att = [cost_att] * n_players
            else:
                raise ValueError("Cost of attention must be non-negative")
        else:
            if len(cost_att) != n_players:
                raise ValueError("Provide a cost of attention for every player")
            for x in cost_att:
                if x < 0 or math.isnan(x):
                    raise ValueError("Cost of attention must be non-negative")
        self.cost_att = np.array(cost_att)

        if action_bounds is None:
            # Set default action bound
            self.action_bounds = []
            for i in range(self.n_players):
                self.action_bounds.append([(None, None)] * self.n_actions[i])
        else:
            if len(action_bounds) != self.n_players:
                raise ValueError("Provide action bounds for every player")
            for i, bounds in enumerate(action_bounds):
                if len(action_bounds[i]) != self.n_actions[i]:
                    raise ValueError(
                        f"Incorrect length of action bounds for player {i}"
                    )
            self.action_bounds = action_bounds

    def compute(
        self,
        interp_base: "interpolation.InterpolatedFunction",
        x0=None,
        att0=None,
        eps=1e-4,
        chunk_size=50,
        max_iter_outer=None,
        max_iter_inner=None,
        interpolate_pf=False,
    ):
        if any(cost > 0 for cost in self.cost_att) and not isinstance(
            interp_base, interpolation.DifferentiableInterpolatedFunction
        ):
            raise ValueError(
                "Can only calculate sparsity for differentiable value" "functions"
            )

        if any(cost > 0 for cost in self.cost_att):
            interpolate_pf = True

        interp_funcs = interpolation.DynamicGameInterpolatedFunctions(interp_base, self)

        # Players may have different numbers of actions. Allocate the maximum
        # number of actions amongst all of them for every player. This might
        # be wasteful when the number of actions is very asymmetric, but in the
        # more common case that every agent has the same number of actions we
        # gain a lot of effiency since we can pass around a NumPy array instead
        # of a Python list containing NumPy arrays.
        max_actions = np.max(self.n_actions)
        init_actions = np.zeros((interp_funcs.shape[0], self.n_players * max_actions))
        if isinstance(x0, Sequence):
            if len(x0) != self.n_players:
                raise ValueError("Provide initial actions for every player.")

            if any(not isinstance(x0_i, np.ndarray) for x0_i in x0):
                raise TypeError("Initial actions must be ndarray's")

            if any(x0[i].shape[1] != self.n_actions[i] for i in range(self.n_players)):
                raise ValueError(
                    "Provide the correct number of initial" " actions for every player."
                )

            if any(x0_i.shape[0] != interp_funcs.shape[0] for x0_i in x0):
                raise ValueError("Provide initial actions for every node.")

            for i in range(self.n_players):
                mi = i * max_actions
                init_actions[:, mi : (mi + self.n_actions[i])] = x0[i]
        elif isinstance(x0, np.ndarray):
            if x0.shape[0] != interp_funcs.shape[0]:
                raise ValueError("Provide initial actions for every node.")

            if x0.shape[1] != self.n_players:
                raise ValueError("Provide initial actions for every player.")

            if x0.shape[2] != max_actions:
                raise ValueError(f"Provide {max_actions} initial actions.")

            for i in range(self.n_players):
                mi = i * max_actions
                init_actions[:, mi : (mi + max_actions)] = x0[:, i]
        elif x0 is not None:
            raise TypeError("x0 must be a sequence or ndarray.")
        init_actions = dask.array.from_array(init_actions, (chunk_size, -1))

        # Combine the nodes and the (flattened) actions into one array.
        # A line is a combintion of a node and the actions players use in
        # that node. Combine them into one so that the node and the related
        # actions are always in the same chunk, so that we can map_blocks()
        # over it efficiently.
        nodes = dask.array.from_array(interp_funcs, (chunk_size, -1))
        lines = dask.array.concatenate((nodes, init_actions), axis=1)
        lines = lines.rechunk((chunk_size, -1))
        lines = lines.persist()

        if att0 is None:
            attention = np.ones((self.n_players, interp_funcs.shape[1]), dtype=np.bool_)
            default_state = np.zeros(interp_funcs.shape[1])
        else:
            if att0.shape[0] != self.n_players:
                raise ValueError("Provide initial attentions for every" " player.")
            if att0.shape[1] != interp_funcs.shape[1]:
                raise ValueError("Provide initial attentions for state" " variable.")
            if att0.dtype != np.bool_:
                raise ValueError("Provide intiial attentions with a bool" " dtype.")
            attention = att0
            default_state, _ = self.compute_default_state(
                0, interp_funcs, nodes[0].compute()
            )
        logging.debug(f"Starting default state: {default_state}")

        if x0 is not None:
            # If starting actions are passed, initialize value actions
            # from the values resulting from these actions.
            prev_value_function = []
            for i in range(self.n_players):
                prev_value_function.append(
                    self.calculate_value_function(
                        lines,
                        i,
                        interp_funcs,
                        attention[i],
                        default_state,
                        chunk_size,
                        eps,
                    )
                )
            prev_optimal = [True] * self.n_players

            if interpolate_pf:
                for i in range(self.n_players):
                    self._update_policy_functions(lines, i, interp_funcs)
        else:
            prev_value_function = [None] * self.n_players
            prev_optimal = [False] * self.n_players

        converged = [False] * self.n_players
        n_iters = 0
        while not all(converged):
            for i in range(self.n_players):
                logging.info(f"Outer loop step for player {i}")

                attention[i], lines, calc_value_function = self._inner_loop(
                    lines,
                    i,
                    interp_funcs,
                    eps,
                    chunk_size,
                    prev_optimal[i],
                    prev_value_function[i],
                    max_iter_inner,
                    interpolate_pf,
                )

                if prev_value_function[i] is not None:
                    vf_norm = self.value_function_norm(
                        calc_value_function, prev_value_function[i]
                    )

                    logging.info(f"Outer loop player {i}, step = {vf_norm}")

                    if vf_norm <= eps:
                        logging.info(f"Outer loop player {i} converged")
                        converged[i] = True
                    else:
                        # The actions for one player have changed. As a
                        # result, we can no longer assume that the
                        # actions calculated for other players are optimal.
                        converged = [False] * self.n_players

                prev_optimal[i] = True

                prev_value_function[i] = calc_value_function

            n_iters += 1
            if max_iter_outer is not None and n_iters >= max_iter_outer:
                raise ComputationLimitError(
                    f"Maximum of {max_iter_outer} iterations in outer loop "
                    "reached without convergence."
                )

        lines = lines.compute()
        optimal_actions = [
            np.zeros((nodes.shape[0], self.n_actions[i])) for i in range(self.n_players)
        ]
        for i in range(nodes.shape[0]):
            _, actions = self._from_line(lines[i])
            for j in range(self.n_players):
                optimal_actions[j][i] = actions[j]
        return attention, optimal_actions, interp_funcs

    def _to_line(self, node, actions):
        return np.concatenate((node, actions.flatten()))

    def _from_line(self, line):
        max_n_actions = np.max(self.n_actions)
        n_actions = self.n_players * max_n_actions
        node = line[:-n_actions]
        actions = line[-n_actions:].reshape((self.n_players, max_n_actions))
        return node, actions

    def _update_policy_functions(self, lines, player_ind, interp_funcs):
        mi = interp_funcs.shape[1] + player_ind * np.max(self.n_actions)
        for i in range(self.n_actions[player_ind]):
            actions = lines[:, mi + i].compute()
            interp_funcs.pf[player_ind][i].update(actions)

    def _inner_loop(
        self,
        lines,
        player_ind,
        interp_funcs,
        eps,
        chunk_size,
        prev_optimal=False,
        prev_value_function=None,
        max_iter_inner=None,
        interpolate_pf=False,
    ):
        n_iters = 0
        while True:
            attention, default_state, lines = self.optimal_actions(
                lines, player_ind, interp_funcs, chunk_size, prev_optimal
            )

            if interpolate_pf:
                self._update_policy_functions(lines, player_ind, interp_funcs)

            calc_value_function = self.calculate_value_function(
                lines,
                player_ind,
                interp_funcs,
                attention,
                default_state,
                chunk_size,
                eps,
            )
            if prev_value_function is not None:
                vf_norm = self.value_function_norm(
                    calc_value_function, prev_value_function
                )

                logging.info(f"Inner loop step {vf_norm}")

                if vf_norm <= eps:
                    break
            prev_value_function = calc_value_function
            prev_optimal = True

            n_iters += 1
            if max_iter_inner is not None and n_iters >= max_iter_inner:
                raise ComputationLimitError(
                    f"Maximum of {max_iter_inner} iterations in inner loop "
                    "reached without convergence."
                )

        return attention, lines, calc_value_function

    def optimal_actions(
        self, lines, player_ind, interp_funcs, chunk_size, prev_optimal=False
    ):
        dim_state = interp_funcs.shape[1]
        if self.cost_att[player_ind] > 0:
            # Potentially, we have sparsity for this player. Figure out
            # which states it pays attention to.
            default_state, default_state_var = self.compute_default_state(
                player_ind, interp_funcs, lines[0, :dim_state].compute()
            )

            actions_others_default = np.zeros(
                (self.n_players - 1, np.max(self.n_actions))
            )
            for i in range(self.n_players):
                if i == player_ind:
                    continue
                for j in range(self.n_actions[i]):
                    ind = i if i < player_ind else i - 1
                    actions_others_default[ind, j] = interp_funcs.pf[i][j](
                        default_state
                    )

            vf = interp_funcs.vf[player_ind]
            start_actions = [
                interp_funcs.pf[player_ind][i](default_state)
                for i in range(self.n_actions[player_ind])
            ]
            start_actions = np.array(start_actions)
            default_action = self.calculate_optimal_action(
                default_state, start_actions, player_ind, vf, actions_others_default
            )
            logging.debug(
                f"Default state = {default_state},"
                f" default action = {default_action}"
            )

            actions = np.insert(
                actions_others_default, player_ind, default_action, axis=0
            )
            deriv_action_state = self.derivative_action_state(
                default_state, player_ind, vf, actions_others_default, default_action
            )
            state_evol = self._normalize_next_state(
                self.state_evolution(player_ind, default_state, actions)
            )
            hess_profits_action = self.static_profits_hessian(
                player_ind, default_state, actions
            )
            if state_evol[2] is not None:
                if state_evol[4] is None:
                    raise ValueError(
                        "When the cost of attention is positive and a gradient for next state's "
                        "elements is provided, also provide a Hessian."
                    )
                vf_grad = np.apply_along_axis(vf.derivative, 1, state_evol[0])
                vf_hess = np.apply_along_axis(vf.second_derivative, 1, state_evol[0])
                hess_profits_action += (
                    self.beta[player_ind]
                    * state_evol[1]
                    @ (np.transpose(state_evol[2], (0, 2, 1)) @ vf_hess @ state_evol[2])
                )
                for i in range(dim_state):
                    hess_profits_action += (
                        self.beta[player_ind]
                        * state_evol[1]
                        @ (vf_grad[:, i] * state_evol[4][:, i])
                    )
            if state_evol[4] is not None:
                # TODO
                pass
            # hess_profits_action = profits_hess + state_grad.T @ vf_hess @ state_grad
            benefit_att = default_state_var * np.diag(
                deriv_action_state.T @ hess_profits_action @ deriv_action_state
            )
            attention = benefit_att >= self.cost_att

            if np.any(attention):
                # TODO
                pass
            else:
                # dask.array.unique() does not work when we pass it an empty
                # array. In this case, the only relevant state is the default
                # state.
                relevant_lines = np.concatenate([default_state, actions.flatten()])
                relevant_lines = dask.array.from_array(
                    relevant_lines[np.newaxis], chunks=(1, -1)
                )
        else:
            attention = np.ones(dim_state, dtype=np.bool_)
            default_state = np.zeros(dim_state)  # Is irrelevant in this case
            relevant_lines = lines

        relevant_lines = dask.array.core.map_blocks(
            self._optimal_action,
            relevant_lines,
            dtype=lines.dtype,
            player_ind=player_ind,
            value_function=interp_funcs.vf[player_ind],
            prev_optimal=prev_optimal,
        )

        if self.cost_att[player_ind] > 0:
            if np.any(attention):
                # TODO
                pass
            else:
                actions = relevant_lines[0, dim_state:].compute().flatten()
                lines = dask.array.map_blocks(
                    lambda x: np.concatenate(
                        (
                            x[:, :dim_state],
                            np.tile(actions[np.newaxis], (x.shape[0], 1)),
                        ),
                        axis=1,
                    ),
                    lines,
                    dtype=lines.dtype,
                )
        else:
            lines = relevant_lines

        lines = lines.persist()
        return attention, default_state, lines

    def _optimal_action(self, lines, prev_optimal, player_ind, value_function):
        ret = np.zeros(lines.shape)
        for i in range(lines.shape[0]):
            state, actions = self._from_line(lines[i])
            actions_others = np.delete(actions, player_ind, axis=0)
            prev = actions[player_ind] if prev_optimal else None
            optimal_action = self.calculate_optimal_action(
                state, prev, player_ind, value_function, actions_others
            )
            actions = np.insert(actions_others, player_ind, optimal_action, axis=0)
            ret[i] = self._to_line(state, actions)

        return ret

    def calculate_optimal_action(
        self, state, prev_optimal, player_ind, value_function, actions_others
    ):
        x0 = self.compute_optimization_x0(
            state, player_ind, value_function, actions_others, prev_optimal
        )
        bounds = self.compute_action_bounds(
            state, player_ind, value_function, actions_others
        )
        res = self._maximize_value(
            state, player_ind, value_function, actions_others, x0, bounds
        )

        if not res.success:
            raise ValueError("Optimization did not converge")
        else:
            return res.x

    def _maximize_value(
        self, state, player_ind, interp_funcs, actions_others, x0, bounds
    ):
        return scipy.optimize.minimize(
            lambda x: self._neg_value(
                state,
                player_ind,
                interp_funcs,
                np.insert(actions_others, player_ind, x, axis=0),
            ),
            x0,
            bounds=bounds,
        )

    def _value(
        self, state, player_ind, value_function, attention, default_state, actions
    ):
        (next_state, next_state_p, *_) = self._normalize_next_state(
            self.state_evolution(player_ind, state, actions)
        )
        if np.any(~attention):
            # There is inattention => calculte the perceived value function
            next_state = np.apply_along_axis(
                lambda x: self.sparse_state(x, attention, default_state), 1, next_state
            )
        cont_value = np.apply_along_axis(value_function, 1, next_state)
        return self.static_profits(player_ind, state, actions) + self.beta[
            player_ind
        ] * (next_state_p @ cont_value)

    def _neg_value(self, state, player_ind, value_function, actions):
        return -1 * self._value(
            state,
            player_ind,
            value_function,
            np.ones(state.shape, dtype=np.bool_),
            np.zeros(state.shape),
            actions,
        )

    def _value_from_blocks(
        self, lines, player_ind, value_function, attention, default_state
    ):
        ret = np.zeros(lines.shape[0])
        for i in range(lines.shape[0]):
            state, actions = self._from_line(lines[i])
            # TODO: loop over only the sparse states in calculate_value_function()
            # => this will be more efficient!
            if np.any(~attention):
                state = self.sparse_state(state, attention, default_state)
            ret[i] = self._value(
                state, player_ind, value_function, attention, default_state, actions
            )
        return ret

    def calculate_value_function(
        self, lines, player_ind, interp_funcs, attention, default_state, chunk_size, eps
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
                value_function=interp_funcs.vf[player_ind],
                attention=attention,
                default_state=default_state,
            ).compute()
            interp_funcs.vf[player_ind].update(calc_value_function)

            if prev_value_function is not None:
                vf_diff = self.value_function_norm(
                    calc_value_function, prev_value_function
                )
                logging.debug(f'Inner loop value function diff = {vf_diff}')
                if vf_diff <= eps:
                    break

            prev_value_function = calc_value_function

        return calc_value_function

    def value_function_norm(self, new_value, old_value):
        # Convergence criterion as in Doraszelski & Pakes (2007)
        return np.max(np.abs((new_value - old_value) / (1 + new_value)))

    def compute_optimization_x0(
        self, state, player_ind, interp_funcs, actions_others, prev_optimal
    ):
        if prev_optimal is None:
            return np.zeros(self.n_actions[player_ind])
        else:
            return prev_optimal

    def compute_action_bounds(self, state, player_ind, interp_funcs, actions_others):
        return self.action_bounds[player_ind]

    def compute_default_state(
        self, player_ind, interp_funcs, start_state, n_sims=10000, sim_length=100
    ):
        states = np.empty((n_sims, start_state.shape[0]))
        state = start_state
        for state_ind in range(n_sims):
            states[state_ind] = state
            actions = np.empty((self.n_players, np.max(self.n_actions)))
            for i in range(self.n_players):
                for j in range(self.n_actions[i]):
                    actions[i, j] = interp_funcs.pf[i][j](state)
            if state_ind > 0 and state_ind % sim_length == 0:
                state = start_state
            else:
                state_evol = self._normalize_next_state(
                    self.state_evolution(player_ind, state, actions)
                )
                state_ind = np.random.choice(state_evol[0].shape[0], p=state_evol[1])
                state = state_evol[0][state_ind]

        default_state = np.mean(states, axis=0)
        default_state_var = np.var(states, axis=0)
        return default_state, default_state_var

    def sparse_state(self, state, att, default_state):
        sparse_state = state.copy()
        sparse_state[~att] = default_state[~att]
        return sparse_state

    def derivative_action_state(
        self,
        state,
        player_ind,
        value_function,
        actions_others,
        action=None,
        eps=np.sqrt(np.finfo(float).eps),
    ):
        # TODO: Perhaps see if we can parallelize this also
        # (is this worth it?)
        deriv = np.empty((self.n_actions[player_ind], state.shape[0]))
        for i in range(self.n_actions[player_ind]):
            deriv[i] = scipy.optimize.approx_fprime(
                state,
                functools.partial(
                    self.calculate_optimal_action,
                    prev_optimal=action,
                    player_ind=player_ind,
                    value_function=value_function,
                    actions_others=actions_others,
                ),
                eps,
            )

        return deriv

    @abc.abstractmethod
    def static_profits(self, player_ind, state, actions):
        pass

    @abc.abstractmethod
    def state_evolution(self, player_ind, state, actions):
        pass

    def _normalize_next_state(self, next_state):
        if isinstance(next_state, Number):
            ret = (np.array([[next_state]]), np.ones(1), None, None, None, None)
        elif len(next_state) == 2:
            ret = (next_state[0], next_state[1], None, None, None, None)
        elif len(next_state) == 3:
            raise ValueError(
                "A tuple of length 3 has an ambiguous interpretation."
                "Use a 4-length tuple with the unused element set to None."
            )
        elif len(next_state) == 4:
            ret = tuple(next_state) + (None, None)
        elif len(next_state) == 5:
            raise ValueError(
                "A tuple of length 5 has an ambiguous interpretation."
                "Use a 6-length tuple with the unused element set to None."
            )
        elif len(next_state) == 6:
            ret = tuple(next_state)
        else:
            raise ValueError("Provide at most 6 elements for the state evolution.")

        if ret[0].shape[0] != len(ret[1]):
            raise ValueError("Mismatch between length of states and probabilities")
        if not np.isclose(np.sum(ret[1]), 1):
            raise ValueError("State probabilities must sum to one")

        return ret


class DynamicGameDifferentiable(DynamicGame):
    @abc.abstractclassmethod
    def static_profits_gradient(self, player_ind, state, actions):
        pass

    def compute(
        self,
        interp_base: "interpolation.DifferentiableInterpolatedFunction",
        *args,
        **kwargs,
    ):
        if not isinstance(
            interp_base, interpolation.DifferentiableInterpolatedFunction
        ):
            raise ValueError(
                "Provide a DifferentiableInterpolatedFunction when computing"
                "a DynamicGameDifferentiable"
            )

        return super().compute(interp_base, *args, **kwargs)

    def _maximize_value(
        self, state, player_ind, value_function, actions_others, x0, bounds
    ):
        return scipy.optimize.minimize(
            lambda x: self._neg_value_deriv(
                state,
                player_ind,
                value_function,
                np.insert(actions_others, player_ind, x, axis=0),
            ),
            x0,
            bounds=bounds,
            jac=True,
        )

    def _neg_value_deriv(self, state, player_ind, value_function, actions):
        val = self._neg_value(state, player_ind, value_function, actions)
        state_evol = self.state_evolution(player_ind, state, actions)
        grad = -1 * self.static_profits_gradient(player_ind, state, actions)
        vf = np.apply_along_axis(value_function, 1, state_evol[0])
        vf_grad = np.apply_along_axis(value_function.derivative, 1, state_evol[0])
        if state_evol[2] is not None:
            # vf_grad @ state_evol[2] has dimension (n_samples, 1, n_actions). Remove
            # middle axis so that multiplication with state_evol (dimension n_samples)
            # returns a vector with dimension n_actions.
            grad -= (
                self.beta[player_ind] * state_evol[1] @ (vf_grad @ state_evol[2])[:, 0]
            )
        if state_evol[3] is not None:
            grad -= state_evol[3] @ vf

        if self.n_actions[player_ind] == 1:
            # When the player has only one action, grad will be a float instead
            # of an ndarray. This creates issues in the LFBGS code so make it
            # an ndarray again,
            grad = np.array([grad])

        return val, grad
