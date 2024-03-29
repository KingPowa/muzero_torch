import torch
import numpy as np
from typing import List, Tuple
from torch import Tensor
from envs import AdvancedDiscreteEnv

class Target:

    def __init__(self, value: Tensor, reward: Tensor, action: int):
        self.value = value
        self.reward = reward
        self.action = action

    def unpack(self) -> Tuple[Tensor, Tensor, int]:
        return self.value, self.reward, self.action

# Class that acts as an adapter for an env for the Muzero
class Game:

    def __init__(self, game_env: AdvancedDiscreteEnv, history_len = 7, only_once_per_player = False):
        self.game_env = game_env
        # Parameter to describe if the game has ended
        self.is_done = False
        # How many version of the board should be encoded in the state
        self.history_len = history_len
        # History of states
        self.history = {
            "P1": [],
            "P2": []
        }
        # History of actions
        self.action_history = []
        # history of reward
        self.rewards = []
        # Current player
        self.current_player = "P1"
        # Parameter that tells if the state representation (and so, the history) will save the board also for the other player
        # when it's not its turn.
        self.only_once_per_player = only_once_per_player

        # Search statistics
        self.child_visits = []
        self.root_values = []
        
    def length_of_history(self):
        """ Return the length of the history
        """
        return len(self.history["P1"]) if not self.only_once_per_player else max(len(self.history["P1"]), len(self.history["P2"]))

    def terminal(self):
        """ Boolean which informs
        whether the game has ended
        """
        return len(self.game_env.possible_actions) == 0 or self.is_done
    
    def make_image(self, idx):
        """ Returns the observation at the given index
        where idx means how many steps in the past
        """
        sub_history_P1 = self._make_subhistory(idx, "P1")
        sub_history_P2 = self._make_subhistory(idx, "P2")
        
        return torch.cat((torch.cat(sub_history_P1, dim=1), torch.cat(sub_history_P2, dim=1),), dim=1)
    
    def sample_position(self):
        # TO-DO: case for only_once
        if self.only_once_per_player:
            pass
        else:
            return np.random.choice(np.arange(len(self.history["P1"])))
    
    def _make_subhistory(self, idx, player):
        # TO-DO: case for only_once
        idx = len(self.history[player]) + idx if idx < 0 else idx
        sub_history = self.history[player][max(0, idx - self.history_len + 1): idx + 1]
        left_history = self.history_len-len(sub_history)
        if left_history > 0:
            sub_history.append(torch.zeros((1, left_history, self.game_env.single_obs_size[2], self.game_env.single_obs_size[3],)))
        return sub_history
    
    def step(self, action):
        """ Make env advance of 1 step and understand if the game is ended
        """
        observation_dict, reward, self.is_done, _ = self.game_env.step(action)
        self.action_history.append(action)
        self.current_player = observation_dict["current_player"]
        if self.only_once_per_player:
            self.history[self.current_player].append(torch.Tensor(observation_dict[self._board_name(self.current_player)]).view(self.game_env.single_obs_size))
        else:
            self.history["P1"].append(torch.Tensor(observation_dict["player_1_board"]).view(self.game_env.single_obs_size))
            self.history["P2"].append(torch.Tensor(observation_dict["player_2_board"]).view(self.game_env.single_obs_size))
        self.rewards.append(reward)

        self.current_player = observation_dict["current_player"]

    def make_target(self, index: int, unroll_steps: int, td_steps: int, discount: float) -> List[Target]:
        targets = []
        # for each step, we calculate the value. It's the next N rewards (weighted) + the N step estimated value 
        for curr_idx in range(index, index + unroll_steps):
            # N-step value function estimate
            value = self.root_values[curr_idx + td_steps] * discount ** td_steps if (len(self.root_values) > curr_idx + td_steps) else 0
            # Sum of all the rewards from curr_idx to curr_idx + td_steps
            for i, reward in enumerate(self.rewards[curr_idx:curr_idx+td_steps]):
                # Sum to the value the reward multiplied by the discount
                value += reward * discount ** i
            # Value is now the estimated value function at step curr_idx. Target is this value.
            targets.append(Target(value, self.rewards[curr_idx], self.child_visits[curr_idx]) if curr_idx < len(self.root_values) else (0,0, []))
        return targets

    def _board_name(self, player):
        return "player_1_board" if player == "P1" else "player_2_board"

    def to_play(self):
        """ Return next current player 
        """
        return self.current_player
    
    def legal_actions(self):
        """ Return a list of legal actions
        """
        return self.game_env.possible_actions
    
    def action_mask(self):
        """ Return an action mask
        """
        action_mask = self.game_env.action_mask
        return torch.Tensor(action_mask)
    
    def action_space(self, mask=False):
        """ Return the action space
        """
        action_space_cardinality = self.game_env.action_space.n
        if mask: return np.ones((action_space_cardinality))
        else: return np.arange(0, action_space_cardinality)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in self.action_space(mask=False)
        ])
        self.root_values.append(root.value())

    def present_game(self):
        self.game_env.print_game()

    def reset(self):
        self.game_env.reset()