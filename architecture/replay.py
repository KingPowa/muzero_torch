import numpy as np
from typing import List, Tuple
from architecture.game import Game

class ReplayBuffer:

    def __init__(self, max_buffer_size, batch_size):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.buffer = [] # <- A list of games

    def sample_game(self):
        return np.random.choice(self.buffer)
    
    def sample_position(self, game: Game):
        return game.sample_position()

    def sample_batch(self, unroll_steps: int, td_steps: int, ):
        # Get all games
        games : List[Game] = [self.sample_game() for _ in range(self.batch_size)]
        game_position : List[Tuple[Game, int]] = [(game, self.sample_position(game)) for game in games]
        return [(game.make_image(position), 
                 game.action_history[position:position+unroll_steps],
                 game.make_target(position, unroll_steps, td_steps, 0.99))
                 for game, position in game_position]