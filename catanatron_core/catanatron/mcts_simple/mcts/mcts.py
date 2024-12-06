from __future__ import annotations
import gc
import os
import pickle
import random
from copy import deepcopy
from tqdm.auto import tqdm
from typing import List, Dict, Union, Optional
from catanatron.models.enums import ActionType, Action, RESOURCES
from catanatron.models.map import build_dice_probas
from catanatron.state_functions import player_key

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

class Node:
    def __init__(self, player: int, state, prev_node: Optional[Node] = None, transposition_table: Optional[Dict[tuple, str]] = None):
        self.player = player # player that makes a move which leads to one of the child nodes
        self.state = state
        
        self.prev_node = prev_node
        self.transposition_table = transposition_table # {(player, state): Node}
        self.children = dict() # {action: Node}

        self.is_expanded = False
        self.has_outcome = False

        self.w = 0. # number of games won by previous player where node was traversed
        self.n = 0 # number of games played where node was traversed

    def eval(self, training: bool) -> float:
        return self.w / self.n if self.n > 0 else float("inf") if training else 0.

    def add_child(self, next_player: int, next_state, action: int) -> None:
        if action not in self.children:
            if self.transposition_table is not None:
                key = (next_player, next_state)
                if key in self.transposition_table:
                    self.children[action] = self.transposition_table[key]
                else:
                    self.children[action] = self.transposition_table[key] = Node(next_player, next_state, transposition_table = self.transposition_table)
            else:
                self.children[action] = Node(next_player, next_state, prev_node = self)

    def choose_best_action(self, training: bool) -> int:
        return max(self.children, key = lambda action: self.children[action].eval(training))

    def choose_random_action(self) -> int:
        return random.sample(list(self.children), 1)[0]

class MCTS:
    def __init__(self, game: Game, allow_transpositions: bool = False, training: bool = True):
        self.game = game
        self.copied_game = deepcopy(self.game)

        self.transposition_table = dict() if allow_transpositions is True else None
        self.root = Node(self.game.current_player(), self.game.get_state(), transposition_table = self.transposition_table)
        if self.transposition_table is not None:
            self.transposition_table[(self.game.current_player(), str(self.game.get_state()))] = self.root
        self.training = training

    def update_root(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.game.take_action(action)
            self.copied_game = deepcopy(self.game)
        else:
            raise ValueError("Action not found in current root's children")
    
        
    def selection(self, node: Node) -> List[Node]:
        path = [node]
        while path[-1].is_expanded is True and path[-1].has_outcome is False: # loop if not leaf node
            action = path[-1].choose_best_action(self.training)
            path.append(path[-1].children[action])
            if action.action_type == ActionType.ROLL:
                action = Action(action.color, action.action_type, 
                                rolls[random.choices(numbers, probs_list)[0]])
                path.append(path[-1].children[action])
            self.copied_game.take_action(action)
        return path

    def expansion(self, path: List[Node]) -> List[Node]:
        if path[-1].is_expanded is False and path[-1].has_outcome is False:
            for action in self.copied_game.possible_actions():
                if action.action_type == ActionType.ROLL:
                    path[-1].add_child(self.copied_game.current_player(), self.copied_game.get_state(), action)
                    for roll in numbers:
                        roll_game = deepcopy(self.copied_game)
                        copied_action = Action(action.color, action.action_type, rolls[roll])
                        roll_game.take_action(copied_action)
                        path[-1].children[action].add_child(roll_game.current_player(),roll_game.get_state(), copied_action)
                else:
                    expanded_game = deepcopy(self.copied_game)
                    expanded_game.take_action(action)
                    path[-1].add_child(expanded_game.current_player(), expanded_game.get_state(), action)

            assert len(path[-1].children) > 0
            
            path[-1].is_expanded = True
            action = path[-1].choose_random_action()
            path.append(path[-1].children[action])
            if action.action_type == ActionType.ROLL:
                path[-1].is_expanded = True
                action = Action(action.color, action.action_type, 
                                rolls[random.choices(numbers, probs_list)[0]])
                path.append(path[-1].children[action])
            self.copied_game.take_action(action)
        return path

    def simulation(self, path: List[Node]) -> List[Node]:
        while self.copied_game.has_outcome() is False:
            action = random.choice(self.copied_game.possible_actions())
            if action.action_type == ActionType.ROLL:
                path[-1].add_child(self.copied_game.current_player(), self.copied_game.get_state(), action)
                path.append(path[-1].children[action])
                action = Action(action.color, action.action_type, 
                                rolls[random.choices(numbers, probs_list)[0]])
            self.copied_game.take_action(action)
            path[-1].add_child(self.copied_game.current_player(), self.copied_game.get_state(), action)
            path.append(path[-1].children[action])
        return path

    def backpropagation(self, path: List[Node]) -> None:
        if self.copied_game.has_outcome() is True:
            winners = self.copied_game.winner()
            number_of_winners = len(winners)
            path[0].n += 1
            for i in range(1, len(path)):
                if number_of_winners > 0:
                    if path[i].player == winners[0]:
                        path[i].w += 1
                path[i].n += 1
            path[-1].has_outcome = True

    def step(self) -> None:
        if self.training is True:
            self.backpropagation(self.simulation(self.expansion(self.selection(self.root))))
        else:
            node = self.root
            while not self.copied_game.has_outcome():
                self.copied_game.render()
                if len(node.children) > 0:
                    action = node.choose_best_action(self.training)
                    node = node.children[action]
                else:
                    action = random.choice(self.copied_game.possible_actions())
                self.copied_game.take_action(action)
            self.copied_game.render()
            
        self.copied_game = deepcopy(self.game)
        #gc.collect()

    def self_play(self, iterations: int = 1) -> None:
        desc = "Training" if self.training is True else "Evaluating"
        for _ in tqdm(range(iterations), desc = desc):
            self.step()

    def save(self, file_path: Union[str, os.PathLike]) -> None:
        game, copied_game, training = self.game, self.copied_game, self.training
        del self.game, self.copied_game, self.training
        with open(file_path, "wb") as handle:
            pickle.dump(self, handle, protocol = -1)
            handle.close()
        self.game, self.copied_game, self.training = game, copied_game, training

    def load(self, file_path: Union[str, os.PathLike]) -> None:
        with open(file_path, "rb") as handle:
            self.__dict__.update(pickle.load(handle).__dict__)
            handle.close()
        if self.transposition_table is not None:
            self.root = self.transposition_table[(self.game.current_player(), str(self.game.get_state()))]
        assert self.game.current_player() == self.root.player and str(self.game.get_state()) == self.root.state
