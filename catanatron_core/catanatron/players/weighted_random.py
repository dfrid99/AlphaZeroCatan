import random

from catanatron.models.player import Player
from catanatron.models.actions import ActionType
from catanatron.state_functions import (
    player_key,
)


WEIGHTS_BY_ACTION_TYPE = {
    ActionType.BUILD_CITY: 10000,
    ActionType.BUILD_SETTLEMENT: 1000,
    ActionType.BUY_DEVELOPMENT_CARD: 100,
}


class WeightedRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution
    to actions that are likely better (cities > settlements > dev cards).
    """

    def decide(self, game, playable_actions):
        WEIGHTS_BY_ACTION_TYPE[ActionType.END_TURN] = 1
        key = player_key(game.state, self.color)
        city_count = 4-game.state.player_state[f"{key}_CITIES_AVAILABLE"]
        bloated_actions = []
        if city_count == 0 and ActionType.BUY_DEVELOPMENT_CARD in playable_actions:
            WEIGHTS_BY_ACTION_TYPE[ActionType.END_TURN] = 1000
        for action in playable_actions:
            weight = WEIGHTS_BY_ACTION_TYPE.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)
        
        
        return random.choice(bloated_actions)
