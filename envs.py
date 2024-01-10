LOSS_REWARD = -10.0
TIE_REWARD = -1.0
WIN_REWARD = 1.0
NORMAL_REWARD = 0.0

import numpy as np
from connect4 import ConnectNBoard
from typing import List, Tuple
from gym import Env
from gym.spaces import Box, Dict, Discrete
from policies.RandomPolicy import RandomPolicy
from abc import ABC, abstractmethod

class AdvancedDiscreteEnv(Env):

    @property
    @abstractmethod
    def possible_actions(self) -> List:
        pass

    @property
    @abstractmethod
    def single_obs_size(self) -> Tuple:
        pass

    @property
    @abstractmethod
    def action_mask(self) -> np.array:
        pass

class ConnectNEnv(AdvancedDiscreteEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, board_w = 6, board_l = 7, connect = 4):
        self.board_w = board_w
        self.board_l = board_l
        self.window_size = 512  # The size of the window

        # Observations: the board with placed element
        self.observation_space = Dict(
            {
                "action_mask": Box(0,1,(board_l,), dtype=np.int8),
                "observations": Box(-1,1,(board_w, board_l), dtype=np.int8)
            }
        )

        # We have 7 actions, corresponding to the board_l slots in the grid
        self.action_space = Discrete(board_l)

        # Define the engine
        self.engine = ConnectNBoard(len = board_l, wid = board_w, connect= connect)

        # Turn
        self.turn = True
            

    @property
    def possible_actions(self):
        mask = self.engine.get_available_columns_mask()
        return [action for action in range(0, self.board_l) if mask[action] == 1]
    
    @property
    def action_mask(self):
        return np.array(self.engine.get_available_columns_mask(), dtype=np.int8)
    
    @property
    def single_obs_size(self):
        return (1, 1, self.engine.wid, self.engine.len,)

    def step(self, action):
        
        assert self.action_space.contains(action), "ACTION ERROR {}".format(action)

        # action invalid in current state
        if action not in self.possible_actions:
            raise ValueError("Invalid Action")

        # Current player action
        won = self.engine.place(action)

        state = self._get_state()

        if won is None:
            reward, done, info = TIE_REWARD, True, {}
        elif won:
            reward, done, info = WIN_REWARD, True, {}

        self.turn = not self.turn

        return state, NORMAL_REWARD, False, {}
    
    def reset(self, seed=None, options=None):
        self.engine.reset()
        return self._get_state(), {}

    def _get_state(self):
        p1, p2 = self.engine.get_custom_board()
        turn = "P1" if self.engine.turn else "P2"
        return {
            "observations": p1 + p2,
            "action_mask": self.action_mask,
            "player_1_board": p1,
            "player_2_board": p2,
            "current_player": turn
        }
    