LOSS_REWARD = -10.0
TIE_REWARD = -1.0
WIN_REWARD = 1.0
NORMAL_REWARD = 0.0

import numpy as np
from connect4 import ConnectNBoard

from gym import Env
from gym.spaces import Box, Dict, Discrete
from policies.RandomPolicy import RandomPolicy


class ConnectNEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, board_w = 6, board_l = 7, connect = 4, opponent_policy = "random"):
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

        # Set opponent policy
        if opponent_policy == "random": self.opponent_policy = RandomPolicy(self.action_space)
        else: self.opponent_policy = opponent_policy
            

    @property
    def possible_actions(self):
        mask = self.engine.get_available_columns_mask()
        return [action for action in range(0, self.board_l) if mask[action] == 1]
    
    @property
    def action_mask(self):
        return np.array(self.engine.get_available_columns_mask(), dtype=np.int8)

    def step(self, action):
        
        assert self.action_space.contains(action), "ACTION ERROR {}".format(action)

        # action invalid in current state
        if action not in self.possible_actions:
            raise ValueError("Invalid Action")

        # Current player action
        won = self.engine.place(action)

        if won is None:
            return self._get_state(), TIE_REWARD, True, {}
        elif won:
            return self._get_state(), WIN_REWARD, True, {}
        
        # Simulate
        opponent_action = self.opponent_policy.compute_actions(self._get_state())
        opp_won = self.engine.place(opponent_action)

        if opp_won is None:
            return self._get_state(), TIE_REWARD, True, {}
        elif opp_won:
            return self._get_state(), LOSS_REWARD, True, {}
        
        return self._get_state(), NORMAL_REWARD, False, {}
    
    def reset(self, seed=None, options=None):
        self.engine.reset()
        return self._get_state(), {}


    def _get_state(self):
        return {
            "observations": self.engine.get_custom_board(),
            "action_mask": self.action_mask
        }