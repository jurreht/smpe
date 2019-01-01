import typing

from hypothesis import assume, given
import hypothesis.strategies as st
import pytest

from smpe.smpe import DynamicGame


class MockGame(DynamicGame):
    def static_profits(self, player_ind, state, actions):
        pass

    def state_evolution(self, state, actions):
        pass

    def sparse_state(self, state, attention):
        pass


def test_init_incorrect_n_players():
    """
    The constructor should raise an expection when the number of
    playes is smaller than 1.
    """
    with pytest.raises(ValueError):
        MockGame(0, 1, .98, 0)


# max_value for speed
@given(n_players=st.integers(min_value=1, max_value=20))
def test_init_correct_n_players(n_players):
    """
    The constructor should accept any number of players >= 1.
    """
    MockGame(n_players, 1, .98, 0)


@given(
    # max_value to prevent slowness
    n_players=st.integers(min_value=1, max_value=10),
    n_actions=st.integers(min_value=1, max_value=10))
def test_init_int_n_actions(n_players, n_actions):
    """
    The constructor should accept an int as an argument to n_actions.
    """
    game = MockGame(n_players, n_actions, .98, 0)
    assert isinstance(game.n_actions, typing.Sequence)
    assert len(game.n_actions) == n_players
    for x in game.n_actions:
        assert x == n_actions


@given(
    n_players=st.integers(min_value=1),
    n_actions=st.integers(max_value=0))
def test_init_incorrect_int_n_actions(n_players, n_actions):
    """
    When giving an int to n_actions smaller than 1, this raise an error.
    """
    with pytest.raises(ValueError):
        MockGame(n_players, n_actions, .98, 0)


@given(
    n_players=st.integers(min_value=1),
    n_actions=st.lists(elements=st.integers(min_value=1, max_value=10)))
def test_init_n_actions_list_length(n_players, n_actions):
    """
    The constructor should raise an expection when the size of the
    number of actions list is different from the number of players, and no
    expception otherwise.
    """
    if len(n_actions) != n_players:
        with pytest.raises(ValueError):
            MockGame(n_players, n_actions, .98, 0)
    else:
        MockGame(n_players, n_actions, .98, 0)


@given(n_actions=st.lists(
    elements=st.integers(max_value=0),
    min_size=1
))
def test_init_n_actions_list_negative_el(n_actions):
    """
    The constructor should raise an exception if the number of actions
    for one player is < 1.
    """
    with pytest.raises(ValueError):
        MockGame(len(n_actions), n_actions, .98, 0)


@given(n_actions=st.lists(
    elements=st.integers(min_value=1, max_value=10),
    min_size=1
))
def test_init_n_actions_list_positive_el(n_actions):
    """
    The constructor should raise no exception if the number of actions
    for one player is >= 1.
    """
    MockGame(len(n_actions), n_actions, .98, 0)


@given(
    # max_value to prevent slowness
    n_players=st.integers(min_value=1, max_value=10),
    beta=st.floats(min_value=0.01, max_value=0.99))
def test_init_beta_valid_float(n_players, beta):
    """
    The constructor should accept a single valid float as an argument for beta.
    """
    game = MockGame(n_players, 1, beta, 0)
    assert isinstance(game.beta, typing.Sequence)
    assert len(game.beta) == n_players
    for x in game.beta:
        assert x == beta


@given(beta=st.floats())
def test_init_beta_invalid_float(beta):
    """
    The constructor should raise an exception on a single invalid
    float for beta.
    """
    assume(not 0 < beta < 1)
    with pytest.raises(ValueError):
        MockGame(3, 1, beta, 0)


@given(
    n_players=st.integers(min_value=1, max_value=10),
    beta=st.lists(st.floats(min_value=0.01, max_value=0.99))
)
def test_init_beta_length(n_players, beta):
    """
    When beta is passed as a list, raise an exception when its length
    is not n_players, otherwise not.
    """
    if len(beta) != n_players:
        with pytest.raises(ValueError):
            MockGame(n_players, 1, beta, 0)
    else:
        MockGame(n_players, 1, beta, 0)


@given(beta=st.lists(
    elements=st.floats(min_value=0.01, max_value=0.99),
    min_size=1
))
def test_init_beta_correct_list(beta):
    """
    When beta is a list with correct elements, raise no exception.
    """
    MockGame(len(beta), 1, beta, 0)


@given(beta=st.lists(
    elements=st.floats(),
    min_size=1
))
def test_init_beta_incorrect_list(beta):
    """
    When beta is passed as a list with at least one element not in (0, 1),
    raise an exception.
    """
    assume(not all(0 < x < 1 for x in beta))
    with pytest.raises(ValueError):
        MockGame(len(beta), 1, beta, 0)


@given(
    # max_value to prevent slowness
    n_players=st.integers(min_value=1, max_value=10),
    cost_att=st.floats(min_value=0))
def test_init_cost_att_valid_float(n_players, cost_att):
    """
    The constructor should accept a single valid float as an
    argument for cost_att.
    """
    game = MockGame(n_players, 1, .98, cost_att)
    assert isinstance(game.cost_att, typing.Sequence)
    assert len(game.cost_att) == n_players
    for x in game.cost_att:
        assert x == cost_att


@given(cost_att=st.floats())
def test_init_cost_att_invalid_float(cost_att):
    """
    The constructor should raise an exception on a single invalid
    float for cost_att.
    """
    assume(cost_att < 0)
    with pytest.raises(ValueError):
        MockGame(3, 1, .98, cost_att)


@given(
    n_players=st.integers(min_value=1, max_value=10),
    cost_att=st.lists(st.floats(min_value=0))
)
def test_init_cost_att_length(n_players, cost_att):
    """
    When cost_att is passed as a list, raise an exception when its length
    is not n_players, otherwise not.
    """
    if len(cost_att) != n_players:
        with pytest.raises(ValueError):
            MockGame(n_players, 1, .98, cost_att)
    else:
        MockGame(n_players, 1, .98, cost_att)


@given(cost_att=st.lists(
    elements=st.floats(min_value=0),
    min_size=1
))
def test_init_cost_att_correct_list(cost_att):
    """
    When cost_att is a list with correct elements, raise no exception.
    """
    MockGame(len(cost_att), 1, .98, cost_att)


@given(cost_att=st.lists(
    elements=st.floats(),
    min_size=1
))
def test_init_cost_att_incorrect_list(cost_att):
    """
    When cost_att is passed as a list with at least one element < 0,
    raise an exception.
    """
    assume(not all(x >= 0 for x in cost_att))
    with pytest.raises(ValueError):
        MockGame(len(cost_att), 1, .98, cost_att)


class MockGameNoSparseStateMethod(DynamicGame):
    def static_profits(self, player_ind, state, actions):
        pass

    def state_evolution(self, state, actions):
        pass


@given(cost_att=st.lists(
    elements=st.floats(min_value=0),
    min_size=1
))
def test_init_cost_att_sparse_method_not_impl(cost_att):
    """
    When cost_att is non-zero for at least one player, the sublcass of
    DynamicGame must implement the sparse_state() method. If not, the
    constructor should raise an error.
    """
    assume(not all(c == 0 for c in cost_att))
    with pytest.raises(ValueError):
        MockGameNoSparseStateMethod(len(cost_att), 1, .98, cost_att)


@given(
    n_players=st.integers(min_value=1, max_value=10),
    n_actions=st.integers(min_value=1, max_value=10))
def test_init_action_bounds_none(n_players, n_actions):
    """
    When no action bounds are given, the constructor must initialize default
    bounds.
    """
    game = MockGame(n_players, n_actions, .98, 0)
    assert len(game.action_bounds) == n_players
    for bounds in game.action_bounds:
        assert bounds == [(None, None)] * n_actions


@given(
    n_players=st.integers(min_value=1, max_value=10),
    n_bounds=st.integers(min_value=1, max_value=10),
    n_actions=st.integers(min_value=1, max_value=10))
def test_init_action_bounds_wrong_number_players(
    n_players, n_bounds, n_actions
):
    """
    If no action bounds are provided for every player, the constructor
    should raise an exception.
    """
    assume(n_players != n_bounds)
    with pytest.raises(ValueError):
        MockGame(
            n_players, n_actions, .98, 0,
            [[(None, None)] * n_actions] * n_bounds)


@given(
    n_players=st.integers(min_value=1, max_value=10),
    n_actions=st.integers(min_value=1, max_value=10),
    n_bounds_per_player=st.integers(min_value=1, max_value=10))
def test_init_action_bounds_wrong_number_bounds(
    n_players, n_actions, n_bounds_per_player
):
    """
    The number of action bounds per player must equal the number of
    actions. Otherwise, raise an exception.
    """
    assume(n_actions != n_bounds_per_player)
    with pytest.raises(ValueError):
        MockGame(
            n_players, n_actions, .98, 0,
            [[(None, None)] * n_bounds_per_player] * n_players)


@given(
    n_players=st.integers(min_value=1, max_value=10),
    n_actions=st.integers(min_value=1, max_value=10))
def test_init_action_bounds_correct(n_players, n_actions):
    """
    The constructor should raise no exceptions when correct action bounds are
    given.
    """
    MockGame(n_players, n_actions, .98, 0, [[(0, 1)] * n_actions] * n_players)
