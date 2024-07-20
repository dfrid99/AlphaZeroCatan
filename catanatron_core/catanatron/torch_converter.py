from catanatron.mcts_game import CatanGame
from typing import Any
from catanatron.state import State, apply_action
from catanatron import game
from catanatron.models.player import Color
from catanatron.models import map
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.models.enums import ActionPrompt

from catanatron.mcts_simple import Game as MCTS_Game
from catanatron.mcts_simple import UCT
from catanatron.models.enums import (ActionType, Action, WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE)
from catanatron.models.map import build_dice_probas

from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    NUM_NODES,
    CatanMap,
    NodeId,
)

import torch
from torch import nn

from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    MONOPOLY,
    RESOURCES,
    YEAR_OF_PLENTY,
    SETTLEMENT,
    CITY,
    Action,
    ActionPrompt,
    ActionType,
)

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

node_direction_embed = {"NORTH": (-1, 0), "SOUTH": (1,0),
                        "NORTHEAST":(-1,2), "NORTHWEST":(-1,-2),
                        "SOUTHEAST":(1,2), "SOUTHWEST":(1,-2)}

edge_direction_embed = {"EAST": (0, 2), "WEST": (0,-2),
                        "NORTHEAST":(-1,1), "NORTHWEST":(-1,-1),
                        "SOUTHEAST":(1,1), "SOUTHWEST":(1,-1)}

resources = {'ORE':0, 'WHEAT':1, 'SHEEP':2, 'WOOD':3, 'BRICK':4, None:5}

numbers = list(rolls.keys())
probs_list = []
for number in numbers:
    probs_list.append(DICE_PROBS[number])


def in_player_building(node, buildings):
    for building in buildings:
        if node == building:
            return True
    return False

def in_player_roads(edge, roads):
    for road in roads:
        if edge == road:
            return True
    return False

def state_to_tensor(state, board_embed, resources_embed):
    center = (board_embed.shape[1]//2, board_embed.shape[2]//2)
    board = state.board.map
    players = state.color_to_index
    current_player = state.current_player_index
    opposing_player = 0 if current_player == 1 else 1
    buildings = state.buildings_by_color
    for coords in board.land_tiles:
        # height = center[0] + coords[0]
        # width = center[1] + coords[1]
        # board[height, width, :4] = 1
        if state.board.robber_coordinate == coords:
            robber = True
        else:
            robber = False
        tile = board.land_tiles[coords]
        settles = {'current':[], 'opposing':[]}
        roads = {'current':[], 'opposing':[]}
        for color in players:
            if players[color] == current_player:
                for node in tile.nodes:
                    if in_player_building(tile.nodes[node], buildings[color]['SETTLEMENT']):
                        settles['current'].append(node_direction_embed[node.value])
                for edge in tile.edges:
                    if in_player_roads(tile.edges[edge], buildings[color]['SETTLEMENT']):
                        roads['current'].append(edge_direction_embed[edge.value])
            else:
                for node in tile.nodes:
                    if in_player_building(tile.nodes[node], buildings[color]['SETTLEMENT']):
                        settles['opposing'].append(node_direction_embed[node.value])

                for edge in tile.edges:
                    if in_player_roads(tile.edges[edge], buildings[color]['SETTLEMENT']):
                        roads['opposing'].append(edge_direction_embed[edge.value])

        if coords[0] == 0:
            height = center[0] + 2*coords[2]
            width = center[1] + 2*coords[2]
        if coords[1] == 0:
            height = center[0] + 2*coords[2]
            width = center[1] + 2*coords[0]
        if coords[2] == 0:
            height = center[0]
            width = center[1] + 4*coords[0]
        board_embed[0:4,height,width] = resources_embed(torch.tensor([resources[tile.resource]]))
        if robber:
            board_embed[4,height,width] = 1
        for settle in settles['current']:
            board_embed[0:4, height+settle[0], width+settle[1]] = 1
        for settle in settles['opposing']:
            board_embed[0:4, height+settle[0], width+settle[1]] = -1
        for road in roads['current']:
            board_embed[0:4, height+road[0], width+road[1]] = 1
        for road in roads['opposing']:
            board_embed[0:4, height+road[0], width+road[1]] = -1
        current_vps = state.player_state[f"P{current_player}_VICTORY_POINTS"]
        opponent_vps = state.player_state[f"P{opposing_player}_VICTORY_POINTS"]
        board_embed[5] = current_vps
        board_embed[6] = opponent_vps
        if state.player_state[f"P{current_player}_HAS_ROAD"]:
            board_embed[7] = 1
        if state.player_state[f"P{opposing_player}_HAS_ROAD"]:
            board_embed[7] = -1

        if state.player_state[f"P{current_player}_HAS_ARMY"]:
            board_embed[8] = 1
        if state.player_state[f"P{opposing_player}_HAS_ARMY"]:
            board_embed[8] = -1

    return board_embed


def actions_to_tensor(root):
    actions_embed = torch.zeros(10)
    actions_board_embed = torch.zeros(7,13)
    for action in root.children:
        prob = root.children[action].n/root.n
        if action.action_type == ActionType.END_TURN:
            actions_embed[0] += prob    
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            actions_embed[1] += prob
            node_id = action.value
        if action.action_type == ActionType.BUILD_CITY:
            actions_embed[2] += prob
        if action.action_type == ActionType.BUILD_ROAD:
            actions_embed[3] += prob
        if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            actions_embed[4] += prob
        if action.action_type == ActionType.PLAY_KNIGHT_CARD:
            actions_embed[5] += prob
        if action.action_type == ActionType.PLAY_MONOPOLY:
            actions_embed[6] += prob
        if action.action_type == ActionType.PLAY_ROAD_BUILDING:
            actions_embed[7] += prob
        if action.action_type == ActionType.PLAY_ROAD_BUILDING:
            actions_embed[8] += prob
        if action.action_type == ActionType.ROLL:
            actions_embed[9] += prob
    return actions_embed