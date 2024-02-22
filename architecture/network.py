import torch
import os
from itertools import chain
from networks.architecture import *
from typing import Tuple, NamedTuple, List, Dict
from torch import Tensor
import utils as ut

SUPPORT_SIZE_DEFAULT = 601
ENCODED_CHANNELS = 256

class NetworkOutput(NamedTuple):
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    hidden_state: torch.Tensor

    def unpack(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.value, self.reward, self.policy_logits, self.hidden_state

class MuZeroNetwork:

    def __init__(self, num_of_features, board_total_slots, n_possible_actions, configs=None):

        # Setup configs
        configs = self.default_configs(configs)


        self.representation_network = RepresentationNetwork(in_channels=num_of_features,
                                                            **configs['representation'])

        self.prediction_network = PredictionNetwork(in_channels=configs['representation']['n_channels'], 
                                                    board_total_slots=board_total_slots,
                                                    action_space_size=n_possible_actions,
                                                    **configs['prediction'])
        
        self.dynamics_network = DynamicsNetwork(in_channels=configs['representation']['n_channels']+1,
                                                  board_total_slots=board_total_slots,
                                                  **configs['dynamics'])
        
        self.prediction_support_size = configs['prediction']['support_size']
        self.dynamics_support_size = configs['dynamics']['support_size']
        self.network_path = configs['path']
        self.max_save = configs['max_save']
        self.action_space_size = n_possible_actions

    def parameters(self):
        return chain(self.prediction_network.parameters(), self.representation_network.parameters(), self.dynamics_network.parameters())

    def default_configs(self, configs):
        if configs is None:
            configs = {"prediction": {}, "representation": {}, "dynamics": {}}
        # Prediction Network
        prediction = {
            "n_convs": 2,
            "n_channels": ENCODED_CHANNELS,
            "n_residual_layers": 10,
            "kernel_size": (3,3),
            "support_size": SUPPORT_SIZE_DEFAULT
        }
        if "prediction" not in configs: 
            configs["prediction"] = prediction
        else:
            ut.fill_defaults(configs["prediction"], prediction)
            # Check if Support Size is ok
            if not (configs["prediction"]['support_size']-1) % 2 == 0: 
                print("[NETWORK - Prediction] Support Size invalid. Set to default = {}.".format(SUPPORT_SIZE_DEFAULT))
                configs["prediction"]['support_size'] = SUPPORT_SIZE_DEFAULT
        # Representation Network
        representation = {
            "n_channels": ENCODED_CHANNELS,
            "n_residual_layers": 10,
            "kernel_size": (3,3)
        }
        if "representation" not in configs: 
            configs["representation"] = representation
        else:
            ut.fill_defaults(configs["representation"], representation)
        # Dynamics Network
        dynamics = {
            "n_convs": 2,
            "n_channels": ENCODED_CHANNELS,
            "n_residual_layers": 10,
            "kernel_size": (3,3),
            "support_size": SUPPORT_SIZE_DEFAULT
        }
        if "dynamics" not in configs: 
            configs["dynamics"] = dynamics
        else:
            ut.fill_defaults(configs["dynamics"], dynamics)
            # Check if Support Size is ok
            if not (configs["dynamics"]['support_size']-1) % 2 == 0: 
                print("[NETWORK - Dynamics] Support Size invalid. Set to default = {}.".format(SUPPORT_SIZE_DEFAULT))
                configs["dynamics"]['support_size'] = SUPPORT_SIZE_DEFAULT

        if "path" not in configs:
            configs["path"] = "model_save"

        if "max_save" not in configs:
            configs["max_save"] = 3

        return configs

    def representation(self, image: torch.Tensor) -> torch.Tensor:
        state_representation = self.representation_network(image)
        orig_shape = state_representation.shape
        # Scale image along each channel
        max_per_channel = state_representation.view(
            orig_shape[0],
            orig_shape[1],
            -1
        ).max(2, keepdim=True)[0].unsqueeze(-1)
        min_per_channel = state_representation.view(
            orig_shape[0],
            orig_shape[1],
            -1
        ).min(2, keepdim=True)[0].unsqueeze(-1)
        scale = max_per_channel - min_per_channel
        scale[scale <= 0] += 1e-5
        return (state_representation - min_per_channel) / scale
    
    def prediction(self, encoded_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Predict via state the policy logits and value function
        return self.prediction_network(encoded_state)
    
    def dynamics(self, encoded_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode the action
        enc_state_shape = encoded_state.shape
        encoded_action = torch.zeros((enc_state_shape[0], 1, enc_state_shape[2], enc_state_shape[3])) / (self.action_space_size-action)
        encoded_action = encoded_action * action[:,:,None,None] / self.action_space_size
        encoded_action = encoded_action.to(action.device)
        encoded_full_state = torch.cat((encoded_state, encoded_action), dim=1)

        state_representation, logits_reward = self.dynamics_network(encoded_full_state)
        orig_shape = state_representation.shape
        # Scale image along each channel
        max_per_channel = state_representation.view(
            orig_shape[0],
            orig_shape[1],
            -1
        ).max(2, keepdim=True)[0].unsqueeze(-1)
        min_per_channel = state_representation.view(
            orig_shape[0],
            orig_shape[1],
            -1
        ).min(2, keepdim=True)[0].unsqueeze(-1)
        scale = max_per_channel - min_per_channel
        scale[scale <= 0] += 1e-5

        return (state_representation - min_per_channel) / scale, logits_reward
  
    def initial_inference(self, image: torch.Tensor) -> NetworkOutput:
        # representation + prediction function
        state_representation = self.representation(image)
        logits_value, logits_policy = self.prediction(state_representation)

        logits_reward = torch.ones(image.shape[0], self.prediction_support_size) * -float("inf")
        logits_reward[:, self.prediction_support_size//2] = 0.0

        return NetworkOutput(logits_value, logits_reward, logits_policy, state_representation)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        # dynamics + prediction function
        next_state, logits_reward = self.dynamics(hidden_state, action)
        logits_value, logits_policy = self.prediction(next_state)

        return NetworkOutput(logits_value, logits_reward, logits_policy, next_state)

    def get_weights(self):
        # Returns the weights of this network.
        return []

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0
    
    def from_output_to_scalar(self, network_output: NetworkOutput, softmax=False, type_output="prediction"):
        value = self.from_support_to_scalar(network_output.value, 
                                                      self.prediction_support_size if type_output == "prediction" else self.dynamics_support_size)
        reward = self.from_support_to_scalar(network_output.reward, 
                                                      self.prediction_support_size if type_output == "prediction" else self.dynamics_support_size)
        if softmax: policy_logits = torch.nn.functional.softmax(network_output.policy_logits, dim=1)
        else: policy_logits = network_output.policy_logits
        return NetworkOutput(value, reward, policy_logits, network_output.hidden_state)
    
    def from_support_to_scalar(self, weights: torch.Tensor, support_size: int) -> torch.Tensor:
        # Get value for each support
        support_vector = torch.arange(-(support_size-1)//2, (support_size-1)//2+1).expand(weights.shape).float().to(weights.device)
        w_softmax = torch.nn.functional.softmax(weights, dim=-1)
        result = torch.sum(support_vector*w_softmax, dim=1, keepdim=True) # Keep dims make it N x D -> N x 1
        # Result is trained with a scaling function h(x), apply it inversely
        return inverse_h(result)
    
    def save_network(self):
        networks = [(net, name) for net, name in zip([self.dynamics_network, self.prediction_network, self.representation_network], ["dynamics", "prediction", "representation"])]
        if not os.path.exists(self.network_path):
            os.makedirs(self.network_path)
        for net, name in networks:
            ut.remove_if_max(self.network_path, name, self.max_save)
            torch.save(net.state_dict(), os.path.join(self.network_path, name + "_" + str(ut.current_milli_time())))
    
    def load_network(self) -> Dict[str, bool]:
        networks: List[Tuple[torch.nn.Module, str]] = [(net, name) for net, name in zip([self.dynamics_network, self.prediction_network, self.representation_network], ["dynamics", "prediction", "representation"])]
        net_dict = {name:False for name[1] in networks}
        for net, name in networks:
            filenet = ut.find_latest(self.network_path, name)
            if filenet:
                net.load_state_dict(torch.load(filenet))
                net_dict[name] = True
        return net_dict

def h(x: torch.Tensor, eps = 1e-2) -> torch.Tensor:
    elem = torch.sqrt(torch.sign(x)+1) - 1
    return torch.sign(x) * elem + eps * x

def inverse_h(x: torch.Tensor, eps = 1e-2) -> torch.Tensor:
    elem = torch.abs(x) + 1 + eps
    elem = torch.sqrt(1 + 4 * eps * elem) - 1
    elem = ((elem / 2 * eps) ** 2) - 1
    return torch.sign(x) * elem

class MuZeroConnectN(MuZeroNetwork):

    def __init__(self, width = 6, length = 7, len_features = 7, configs = {}):
        super(MuZeroConnectN, self).__init__(len_features*2, width*length, length, configs)