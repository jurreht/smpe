from typing import Sequence, Union


class DynamicGame:
    n_players: int
    n_actions: Sequence[int]
    beta: Sequence[float]

    def __init__(
        self,
        n_players: int,
        n_actions: Union[Sequence[int], int],
        beta: Union[Sequence[float], float]
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
                if not 0 < x < 1:
                    raise ValueError('Discount factor must be in (0, 1)')
        self.beta = beta
