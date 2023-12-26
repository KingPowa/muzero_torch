from ray.rllib.utils.typing import TensorStructType
from typing import (
    List,
    Union,
)

class RandomPolicy:

    def __init__(self, action_space):
        self.action_space = action_space

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType]
    ):
        if isinstance(obs_batch, List):
            return [self.compute_action(obs_sample) for obs_sample in obs_batch], [], {}
        else:
            return self.compute_action(obs_batch)
    
    def compute_action(self, obs):
        action_mask = obs['action_mask']
        return self.action_space.sample(mask=action_mask)
