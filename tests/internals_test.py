from hypothesis import given
from hypothesis import assume
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import composite, floats, integers
import numpy as np
import pytest

from smpe.smpe import DynamicGame


class MockGame(DynamicGame):
    def __init__(self, n_players, n_actions):
        super().__init__(n_players, n_actions, 0.9, 0)

    def static_profits(self, player_ind, state, actions):
        pass

    def state_evolution(self, state, actions):
        pass


@pytest.fixture
def simple_game():
    return MockGame(1, 2)


@given(
    node=arrays(np.double, array_shapes(max_dims=1), floats(allow_nan=False)),
    actions=arrays(
        np.double, array_shapes(min_dims=2, max_dims=2), floats(allow_nan=False)
    ),
)
def test_to_from_line(node, actions):
    game = MockGame(actions.shape[0], actions.shape[1])
    node_out, actions_out = game._from_line(game._to_line(node, actions))
    assert np.all(node == node_out)
    assert np.all(actions == actions_out)


@given(next_state=floats(allow_nan=False))
def test_normalize_next_state_scalar(simple_game, next_state):
    assert simple_game._normalize_next_state(next_state) == (
        np.array([[next_state]]),
        np.array([1.0]),
        None,
        None,
    )


@composite
def draws_with_probabilities_array(
    draw,
    dim=2,
    normalize_probs=True,
    dim_state=integers(min_value=1, max_value=10),
    n_draws=integers(min_value=1, max_value=10),
):
    dim_state = draw(dim_state)
    n_draws = draw(n_draws)
    ret = []
    for i in range(dim):
        ret.append(
            draw(arrays(np.double, (n_draws, dim_state), floats(allow_nan=False)))
        )
    if normalize_probs:
        ret[1] = draw(
            arrays(np.double, n_draws - 1, floats(min_value=0, max_value=1))
            .map(lambda x: np.concatenate([x, [1 - x.sum()]]))
            .filter(lambda x: x[-1] >= 0)
        )
    return ret


@given(next_state=draws_with_probabilities_array())
def test_normalize_next_state_2d_array(simple_game, next_state):
    assert simple_game._normalize_next_state(next_state) == (
        next_state[0],
        next_state[1],
        None,
        None,
    )


@given(next_state=draws_with_probabilities_array())
def test_normalize_next_state_2d_array_wrong_length(simple_game, next_state):
    next_state[1] = next_state[1][:-1]
    with pytest.raises(ValueError):
        simple_game._normalize_next_state(next_state)


@given(next_state=draws_with_probabilities_array(normalize_probs=False))
def test_normalize_next_state_2d_array_wrong_probs(simple_game, next_state):
    assume(not np.isclose(np.sum(next_state[1]), 1))
    with pytest.raises(ValueError):
        simple_game._normalize_next_state(next_state)


@given(next_state=draws_with_probabilities_array(dim=3))
def test_normalize_next_state_3d_array_error(simple_game, next_state):
    with pytest.raises(ValueError):
        simple_game._normalize_next_state(next_state)


@given(next_state=draws_with_probabilities_array(dim=4))
def test_normalize_next_state_4d_array(simple_game, next_state):
    assert simple_game._normalize_next_state(next_state) == (
        next_state[0],
        next_state[1],
        next_state[2],
        next_state[3],
    )
