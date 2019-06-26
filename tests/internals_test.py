from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats
import numpy as np

from smpe.smpe import DynamicGame


class MockGame(DynamicGame):
    def __init__(self, n_players, n_actions):
        super().__init__(n_players, n_actions, 0.9, 0)

    def static_profits(self, player_ind, state, actions):
        pass

    def state_evolution(self, state, actions):
        pass


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
