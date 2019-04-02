import numpy as np
from numpy.testing import assert_allclose
import pytest

from smpe.interpolation import (
    MultivariateInterpolatedFunction,
    ShareableInterpolatedFunction,
    ChebyshevInterpolatedFunction
)
from smpe.smpe import (DynamicGame,
                       DynamicGameDifferentiable)


class MockGame(DynamicGame):
    def static_profits(self, player_ind, state, actions):
        pass

    def state_evolution(self, state, actions):
        pass

    def sparse_state(self, state, attention):
        pass


def test_compute_value_functions_wrong_length():
    """
    compute() should raise an exception when the number of value
    functions supplied differs from the number of players.
    """
    game = MockGame(1, 1, .98, 0)
    vf = MultivariateInterpolatedFunction(
        ChebyshevInterpolatedFunction(10, 2, np.zeros(1), np.ones(1)),
        2
    )
    with pytest.raises(ValueError):
        game.compute(vf)


class NotDifferentiableValueFunction(ShareableInterpolatedFunction):
    def nodes(self):
        pass

    def __call__(self, x):
        pass

    def update(self, values):
        pass

    @property
    def num_nodes(self):
        pass

    @property
    def dim_state(self):
        pass

    def __getitem__(self, key):
        pass

    def share(self):
        return type(self)()


def test_compute_cost_att_not_differentiable():
    """
    compute() should raise an exception when att_cost > 0 for any player
    and the value function provided is not differentiable.
    """
    game = MockGame(2, 1, .98, [0, .5])
    vf = MultivariateInterpolatedFunction(
        NotDifferentiableValueFunction(), 2
    )

    with pytest.raises(ValueError):
        game.compute(vf)


class CapitalAccumulationProblem(DynamicGameDifferentiable):
    """
    A standard, one player, capital accumulation problem. This has
    closed form solutions, therefore it is easy to verify our numerical
    results. This is based on Example 6.4 from Acemoglu, "An Introduction
    to Modern Economic growth."
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        super().__init__(1, 1, beta, 0)

    def static_profits(self, player_ind, state, actions):
        return np.log(state[0]**self.alpha - actions[0][0])

    def state_evolution(self, state, actions):
        return [actions[0][0]]

    def static_profits_gradient(self, player_ind, state, actions):
        return -1 / (state[0]**self.alpha - actions[0][0])

    def state_evolution_gradient(self, state, actions):
        return [1.]

    def compute_action_bounds(
        self, state, player_ind, value_function, actions_others
    ):
        lower = next(iter(value_function.nodes()))[0]
        return [(lower, state[0]**self.alpha)]

    def compute_optimization_x0(
        self, state, player_ind, value_functions, actions_others, prev_optimal
    ):
        if prev_optimal is None:
            return np.array([state[0]**self.alpha / 2])
        else:
            return prev_optimal


def test_compute_capital_accumulation(dask_client):
    """
    compute() should be able to approximately find the optimal solution to
    the capital accumulation problem.
    """
    alpha = .5
    beta = .95
    n_nodes = 500
    game = CapitalAccumulationProblem(alpha, beta)
    k_steady_state = (alpha * beta)**(1 / (1 - alpha))
    cheb = ChebyshevInterpolatedFunction(
        n_nodes, 20, [0.], [2 * k_steady_state])
    nodes = np.array(list(cheb.nodes()))[:, 0]
    cheb.update(np.log(nodes**alpha) / (1 - beta))
    vf = MultivariateInterpolatedFunction(
        cheb,
        1
    )
    policy_calc = game.compute(vf, eps=0.01, max_loop_inner=50,
                               max_loop_outer=10)
    policy_exact = alpha * beta * vf.numpy_nodes()**alpha
    assert_allclose(
        policy_calc, policy_exact[np.newaxis], atol=1e-2, rtol=5e-1)
