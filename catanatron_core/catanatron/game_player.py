from catanatron.mcts_simple import Game as MCTS_Game
from catanatron.mcts_simple import UCT
from catanatron.mcts_game import CatanGame
import multiprocessing as mp
from functools import partial
from catanatron.models.player import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.models import map
import random

def play_game(iterations=100, save=False,network=None):
    board = map.CatanMap.from_template(map.MINI_MAP_TEMPLATE)
    game = CatanGame(players=[WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)], board=board)
    tree = UCT(game=game, allow_transpositions=False)
    tree.self_play(iterations=iterations)
    while tree.root.children is not None:
        #print(tree.root.children.keys())
        print(tree.root.state.playable_actions)
        for action in tree.root.children:
            print(action, tree.root.children[action].n)
        if tree.root.n < iterations:
            tree.self_play(iterations=(iterations-tree.root.n))
        keys = list(tree.root.children.keys())
        weights = [tree.root.children[key].n for key in keys]
        random_weighted_action = random.choices(keys, weights=weights, k=1)[0]
        # print(tree.game.get_state().current_prompt)
        #print(random_weighted_action)
        tree.update_root(random_weighted_action)


if __name__ == "__main__":
    play_game()

