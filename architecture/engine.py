import torch
import numpy as np
from architecture.network import MuZeroNetwork, NetworkOutput
from architecture.game import Game
from typing import List, Tuple
from copy import deepcopy

MAXIMUM_FLOAT_VALUE = float("inf")

class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Tuple = None):
    assert known_bounds is None or len(known_bounds) == 2
    self.maximum = known_bounds[1] if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds[0] if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value

class Node:

    def __init__(self, prior: float):
        self.visit_count = 0
        self.player = None
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
           return 0
        return self.value_sum / self.visit_count

class MuZeroConfig:

    def __init__(self, game_class, game_configs: dict = {},
                 history_len = 7, max_moves = 30,
                 root_dirichlet_alpha = 0.3, root_exploration_factor = 0.25,
                 known_bounds = None, c1 = 1.25, c2 = 19.652, num_simulations = 100,
                 discount = 0.99):
        self.game_class = game_class
        self.game_configs = game_configs

        self.history_len = history_len
        self.max_moves = max_moves
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_factor = root_exploration_factor
        self.known_bounds = known_bounds
        self.c1 = c1
        self.c2 = c2
        self.num_simulations = num_simulations
        self.discount = discount

    def new_game(self):
        return Game(self.game_class(**self.game_configs), self.history_len) 
    
    def temperature_value(self, num_counts: int) -> float:
        if num_counts < self.max_moves:
            return 1.0
        else:
            return 0.0

def play_game(config: MuZeroConfig, network: MuZeroNetwork) -> Game:
    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:

        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.action_mask(),
                    network.from_output_to_scalar(network.initial_inference(current_observation), softmax=False))
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history, game.to_play(), game.action_space(mask=True), network)
        
        print("Simulation finished. Current game history {}.".format(game.length_of_history()))
        game.present_game()
        action = select_action(config, game.length_of_history(), root, network)
        game.step(action)
        game.store_search_statistics(root)
    return game
    
def expand_node(node: Node, player: str, action_mask: torch.Tensor, initial_inference: NetworkOutput):
    node.player = player
    node.hidden_state = initial_inference.hidden_state
    node.reward = initial_inference.reward

    with torch.no_grad():
        action_probabilities = initial_inference.policy_logits.clone()
        action_probabilities[action_mask.view(action_probabilities.shape[0], -1) == 0] = -MAXIMUM_FLOAT_VALUE
        action_probabilities = torch.nn.functional.softmax(action_probabilities, dim=1)

        for action in np.arange(0, action_mask.shape[0])[action_mask==1]:
            node.children[action] = Node(action_probabilities[:, action].item())

def add_exploration_noise(config: MuZeroConfig, node: Node):
    noise = np.random.dirichlet([config.root_dirichlet_alpha]*len(node.children)) * config.root_exploration_factor
    for i, child in enumerate(node.children.values()):
        child.prior = (1-config.root_exploration_factor) * child.prior + noise[i]

def run_mcts(config: MuZeroConfig, root: Node, action_history_global: List, player: str, full_action_mask: torch.Tensor, network: MuZeroNetwork):
    min_max = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        # Selection
        # Select the node based on a UCB formula

        action_history = deepcopy(action_history_global)
        node = root
        visited_childs: List[Node] = []
        while node.expanded():
            visited_childs.append(node)
            action, node = select_child(config, node, min_max)
            action_history.append(action)

        # Expansion and Simulation
        # Expand selected node
        last_hidden_state = visited_childs[-1].hidden_state
        action = torch.Tensor([action_history[-1]]).view(1,1)
        network_output = network.from_output_to_scalar(network.recurrent_inference(last_hidden_state, action), softmax=False)
        expand_node(node, player, torch.Tensor(full_action_mask), network_output)
        visited_childs.append(node)

        # Backpropagation
        backpropagate(config, visited_childs, player, network_output.value, min_max)
        
def select_child(config: MuZeroConfig, node: Node, min_max: MinMaxStats):
    # For each node, calculate ucb
    _, action, child = max((ucb_score(config, node, child, min_max), action, child) for action, child in node.children.items())
    return action, child

def ucb_score(config: MuZeroConfig, parent: Node, node: Node, min_max: MinMaxStats):
    probability_term = node.prior * np.sqrt(parent.visit_count) / (node.visit_count + 1)
    probability_term *= config.c1 + np.log((parent.visit_count + config.c2 + 1) / config.c2)
    return min_max.normalize(node.value()) + probability_term

def backpropagate(config: MuZeroConfig, visited_childs: List, player: str, value: float, min_max: MinMaxStats):
    for child in visited_childs[::-1]:
        child.value_sum += value if child.player == player else -value
        child.visit_count += 1
        min_max.update(child.value())

        value = child.reward + config.discount * value

def select_action(config: MuZeroConfig, num_actions: int, node: Node, network: MuZeroNetwork):
    temperature = config.temperature_value(num_actions)
    actions, counts = list(zip(*[(action, child.visit_count) for action, child in node.children.items()]))
    if temperature != 0.0: counts = np.array(counts) ** 1/temperature
    else: return actions[np.argmax(counts)]
    prob_distribution = counts / np.sum(counts)
    return np.random.choice(actions, p=prob_distribution)
