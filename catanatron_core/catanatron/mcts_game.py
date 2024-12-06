from typing import Any
from catanatron.state import State, apply_action
from catanatron import game
from catanatron.models.player import Color
from catanatron.models import map
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.models.enums import ActionPrompt

from catanatron.mcts_simple import Game as MCTS_Game
from catanatron.mcts_simple import UCT
from catanatron.models.enums import ActionType, Action
from catanatron.models.map import build_dice_probas

import random

DICE_PROBS = build_dice_probas()

rolls = {2 :(1,1),
         3 :(1,2),
         4 :(1,3),
         5 :(1,4),
         6 :(1,5),
         7 :(1,6),
         8 :(2,6),
         9 :(3,6),
         10 :(4,6),
         11 :(5,6),
         12 :(6,6)}

numbers = list(rolls.keys())
probs_list = []
for number in numbers:
    probs_list.append(DICE_PROBS[number])

class CatanGame(MCTS_Game):
    def __init__(self, players, board=None):
        self.game = game.Game(players=players, catan_map=board, vps_to_win=5)

    def get_state(self) -> Any:
        return self.game.state
    
    def current_player(self) -> int:
        return self.get_state().current_player_index
    
    def possible_actions(self):
        #get playable actions given state
        actions = self.game.state.playable_actions
        #print(actions)
        action_prompt = self.get_state().current_prompt
        if action_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
            actions.sort(reverse=True, key=lambda x:self.get_state().board.map.node_production[x.value])    
            actions = actions[0:8]
        return actions
    
    def take_action(self, action) -> None:
        #take action update state
        val_action = True
        if action.action_type == ActionType.ROLL:
            val_action = False  
        self.game.execute(action, validate_action=val_action)

    def has_outcome(self) -> bool:
        if ((self.game.winning_color() is None) and 
            (self.game.state.num_turns < 1000)):
            return False
        return True
    
    def winner(self) -> game.List[int]:
        if self.game.winning_color() is not None:
            return [self.game.state.color_to_index[self.game.winning_color()]]
        return []
    
if __name__ == "__main__":
    board = map.CatanMap.from_template(map.MINI_MAP_TEMPLATE)
    init_game = CatanGame(players=[WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)], board=None)
    tree = UCT(game=init_game, allow_transpositions=False)
    tree.self_play(iterations=10000)
    #for key in tree.root.children.keys():
        #print(tree.root.children[key].n, tree.game.get_state().board.map.node_production[key.value])
    root = tree.root
    while root.children:
        action = root.choose_best_action(training=False)
        if (action.action_type == ActionType.ROLL):
                root = root.children[action]
                if root.is_expanded:
                    action = Action(action.color, action.action_type, 
                                rolls[random.choices(numbers, probs_list)[0]])
                else:
                    action = root.choose_random_action()
        print(root.children.keys())
        print(action)
        root = root.children[action]


