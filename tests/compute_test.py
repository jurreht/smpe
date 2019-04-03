from collections import namedtuple
import itertools

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


@pytest.fixture
def game_vf():
    game = MockGame(2, [1, 2], .98, 0)
    vf = MultivariateInterpolatedFunction(
        ChebyshevInterpolatedFunction(10, 2, np.zeros(1), np.ones(1)),
        2
    )
    return game, vf


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


def init_test_x0_incorrect_type(game_vf):
    """
    compute() should raise an exception when x0 is neither a sequence nor
    an ndarray.
    """
    game, vf = game_vf
    with pytest.raises(TypeError):
        game.compute(vf, x0=1)


def test_init_x0_list_incorrect_length(game_vf):
    """
    compute should raise an exception when the x0 is a list of the
    incorrect length.
    """
    game, vf = game_vf
    with pytest.raises(ValueError):
        game.compute(vf, x0=[np.array([1.])])


def test_init_x0_list_incorrect_member_type(game_vf):
    """
    compute should raise an exception when the x0 is a list containing
    something other than ndarray's.
    """
    game, vf = game_vf
    with pytest.raises(TypeError):
        game.compute(vf, x0=[[1.], [2.]])


def test_init_x0_list_incorrect_num_actions(game_vf):
    """
    compute should raise an exception when the x0 is a list where the
    ndarrays contain the wrong number of actions.
    """
    game, vf = game_vf
    # Wrong number of actions for player 1 (should be 1)
    with pytest.raises(ValueError):
        game.compute(vf, x0=[np.tile(np.array([1, 2]), (10, 1)),
                             np.tile(np.array([1, 2]), (10, 1))])

    # Wrong number of actions for player 2 (should be 2)
    with pytest.raises(ValueError):
        game.compute(vf, x0=[np.tile(np.array([1]), (10, 1)),
                             np.tile(np.array([1]), (10, 1))])


def test_init_x0_list_incorrect_num_nodes(game_vf):
    """
    compute should raise an exception when the x0 is a list where the
    ndarrays contain the wrong number states.
    """
    game, vf = game_vf
    # Wrong number of actions for player 1 (should be 10)
    with pytest.raises(ValueError):
        game.compute(vf, x0=[np.tile(np.array([1]), (2, 1)),
                             np.tile(np.array([1, 2]), (10, 1))])

    # Wrong number of actions for player 2 (should be 10)
    with pytest.raises(ValueError):
        game.compute(vf, x0=[np.tile(np.array([1]), (10, 1)),
                             np.tile(np.array([1, 2]), (2, 1))])


def test_init_x0_ndarray_incorrect_num_actions(game_vf):
    """
    compute() should raise an exception when x0 is an ndarray with
    the incorrect number of actions.
    """
    game, vf = game_vf
    with pytest.raises(ValueError):
        game.compute(vf, np.tile(np.array([[1]]), (10, 1)))


def test_init_x0_ndarray_incorrect_num_nodes(game_vf):
    """
    compute() should raise an exception when x0 is an ndarray with
    the incorrect number of nodes.
    """
    game, vf = game_vf
    with pytest.raises(ValueError):
        game.compute(vf, np.tile(np.array([[1]]), (9, 2)))


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

    def state_evolution_gradient(self, player_ind, state, actions):
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
    n_nodes = 100
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
    policy_calc = game.compute(vf, eps=0.01, max_iter_inner=50,
                               max_iter_outer=10)
    policy_exact = alpha * beta * vf.numpy_nodes()**alpha
    assert_allclose(
        policy_calc, policy_exact[np.newaxis], atol=1e-2, rtol=5e-1)


class SwitchingModel(DynamicGameDifferentiable):
    """
    A model of competition with switching costs (Einav and Somaini, 2013).

    Somaini, Paulo and Einav, Liran. 2013. "A Model of Market Power in
    Customer Markets." Journal of Industral Economics 61 (4): 938-986.
    """

    def __init__(
        self, n_firms, marginal_cost, switch_cost, new_consumers, df_firms,
        df_consumers
    ):
        assert len(marginal_cost) == n_firms
        self.marginal_cost = marginal_cost

        assert switch_cost >= 0
        self.switch_cost = switch_cost

        assert new_consumers > 0
        self.new_consumers = new_consumers

        # df_firms = r_f * g in the paper's notation is firms' effective
        # discount rate
        assert 0 <= df_firms < 1

        # For now, the code only works when consumers are not forward-looking.
        # In this case, consumers' choices do not depend on equilibrium policy
        # function. To accommodate df_consumers > 0, we would have to add
        # the consumers as an additional player.
        assert df_consumers == 0
        self.df_consumers = df_consumers

        super().__init__(n_firms, 1, df_firms, 0, [[(0, None)]] * n_firms)

    def static_profits(self, player_ind, state, actions):
        pi = actions[player_ind][0]
        p_other = np.delete(actions, player_ind).mean()
        xi = state[player_ind]
        ci = self.marginal_cost[player_ind]

        # eq. 10
        demand_old = .5 * (self.n_players - 1) * (
            (1 - pi + p_other) +
            (self.n_players * xi - 1) * self.switch_cost)

        # eq. 12
        demand_young = (
            .5 * (self.n_players - 1) * (1 - pi + p_other) *
            self.new_consumers)

        # first part of eq. 22
        return (pi - ci) * (demand_young + demand_old)

    def state_evolution(self, state, actions):
        next_state = np.zeros(self.n_players)
        for i in range(self.n_players):
            pi = actions[i][0]
            p_other = np.delete(actions, i).mean()

            next_state[i] = (
                .5 * (self.n_players - 1) * (1 - pi + p_other) *
                self.new_consumers)

        return next_state

    def static_profits_gradient(self, player_ind, state, actions):
        pi = actions[player_ind][0]
        p_other = np.delete(actions, player_ind).mean()
        xi = state[player_ind]
        ci = self.marginal_cost[player_ind]

        # eq. 10
        demand_old = .5 * (self.n_players - 1) * (
            (1 - pi + p_other) +
            (self.n_players * xi - 1) * self.switch_cost)

        # eq. 12
        demand_young = (
            .5 * (self.n_players - 1) * (1 - pi + p_other) *
            self.new_consumers)

        return np.array([
            demand_old + demand_young -
            (pi - ci) * (self.n_players - 1) * .5 * (1 + self.new_consumers)])

    def state_evolution_gradient(self, player_ind, state, actions):
        state_diff = np.full((self.n_players, 1), .5 * self.new_consumers)
        state_diff[player_ind] = -.5 * (self.n_players - 1) * self.new_consumers
        return state_diff


# Correct equilibria for SwitchingModel. Computed from the source code as
# written by Einav & Somaini (available on soma.people.stanford.edu).
# s, df, N and g are the parameters of the model. beta, mu0, mu1 are the
# parametrization of the equilibrium policy function. We have that
# p_i = mean(c) + mu0 + mu1 (c_i - mean(c)) + beta * x_i.
SwitchingEq = namedtuple(
    'SwitchingEq',
    ['s', 'df', 'N', 'g', 'beta', 'mu0', 'mu1'])
switching_eqs = (
    SwitchingEq(0, .9, 2, 1, 0, 1, .3333),
    SwitchingEq(.5, .9, 2, 1, .1681, .7940, .3656),
)


@pytest.mark.parametrize(
    'eq, mc',
    [(eq, np.insert(np.zeros(eq.N - 1), 0, c))
     for eq, c in itertools.product(switching_eqs, (0, .5))]
)
def test_switching_model(dask_client, eq, mc):
    n_nodes = 20

    game = SwitchingModel(eq.N, mc, eq.s, eq.g, eq.df, 0)
    cheb = ChebyshevInterpolatedFunction(
        n_nodes, 4, np.zeros(eq.N), np.ones(eq.N))

    nodes = cheb.numpy_nodes()
    # Einav and Somaini (2013, Theorem 1)
    policy_check = eq.mu0 + eq.mu1 * (mc - mc.mean()) + mc.mean() + eq.beta * nodes
    policy_check = policy_check[..., np.newaxis]

    vf = MultivariateInterpolatedFunction(cheb, eq.N)

    policy_calc = game.compute(vf, policy_check, max_iter_inner=50,
                               max_iter_outer=10, chunk_size=100)

    # We calculate the policies on a regular grid, so that in many of
    # the initial states the market shares do not sum to 1. However, the
    # policy functions from the paper are only valid when the market shares
    # *do* sum to 1, so compare only for these nodes.
    check_nodes = nodes.sum(axis=1) == 1
    for i in range(eq.N):
        assert_allclose(policy_calc[i][check_nodes],
                        policy_check[check_nodes, i],
                        atol=1e-2, rtol=1e-2)
