from torch.nn import Module
from torch.nn.functional import softmax
from networks.common import GenericResidualNetwork, ContinousValuePredictor

class DynamicsNetwork(Module):

    def __init__(self, in_channels, board_total_slots, n_convs = 2, n_channels = 256, n_residual_layers=10, kernel_size = (3,3), support_size = 601):
        super(DynamicsNetwork, self).__init__()

        self.first_net = GenericResidualNetwork(in_channels=in_channels, n_channels=n_channels, n_layers=n_residual_layers, kernel_size=kernel_size)
        self.reward_predictor = ContinousValuePredictor(in_channels=n_channels, board_total_slots=board_total_slots, n_outputs=support_size, n_convs=n_convs)

    def forward(self, x):

        next_state = self.first_net(x)
        return next_state, softmax(self.reward_predictor(next_state), dim=1)
    
class PredictionNetwork(Module):

    def __init__(self, in_channels, board_total_slots, action_space_size, n_convs = 2, n_channels = 256, n_residual_layers=10, kernel_size = (3,3), support_size = 601):
        super(PredictionNetwork, self).__init__()

        self.first_net = GenericResidualNetwork(in_channels=in_channels, n_channels=n_channels, n_layers=n_residual_layers, kernel_size=kernel_size)
        self.value_predictor = ContinousValuePredictor(in_channels=n_channels, board_total_slots=board_total_slots, n_outputs=support_size, n_convs=n_convs)
        self.policy_predictor = ContinousValuePredictor(in_channels=n_channels, board_total_slots=board_total_slots, n_outputs=action_space_size, n_convs=n_convs)

    def forward(self, x):

        x = self.first_net(x)
        pp = self.policy_predictor(x)
        vp = self.value_predictor(x)
        return softmax(vp, dim=1), softmax(pp, dim=1)
    
class RepresentationNetwork(Module):

    def __init__(self, in_channels, n_channels = 256, n_residual_layers=10, kernel_size = (3,3)):
        super(RepresentationNetwork, self).__init__()

        self.net = GenericResidualNetwork(in_channels=in_channels, n_channels=n_channels, n_layers=n_residual_layers, kernel_size=kernel_size)

    def forward(self, x):

        return self.net(x)
    
