from torch.nn import Module, ModuleList
from torch.nn import Conv2d, Linear
from torch.nn import ReLU, BatchNorm2d

class ConvolutionalBlock(Module):

    def __init__(self, in_channels, n_channels = 256, kernel_size = (3,3), stride=(1,1), padding=(1,1)):
        super(ConvolutionalBlock, self).__init__()
        self.convo = Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.norm = BatchNorm2d(num_features=n_channels)
        self.act = ReLU()

    def forward(self, x):
        return self.act(self.norm(self.convo(x)))
    
class ResidualBlock(Module):

    def __init__(self, in_channels, n_channels = 256, kernel_size = (3,3), stride=(1,1)):
        super(ResidualBlock, self).__init__()
        self.convo1 = ConvolutionalBlock(in_channels=in_channels, n_channels=n_channels, kernel_size=kernel_size, stride=stride)
        self.convo2 = Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=(1,1), stride=stride)
        self.norm = BatchNorm2d(num_features=n_channels)
        self.act = ReLU()

    def forward(self, x):
        skip_conn = x
        x = self.convo1(x)
        x = self.norm(self.convo2(x))
        x = x + skip_conn
        x = self.act(x)
        return x
    
class GenericResidualNetwork(Module):

    def __init__(self, in_channels, n_channels = 256, n_layers=10, kernel_size = (3,3)):
        super(GenericResidualNetwork, self).__init__()

        # For now, simple, no downscale
        self.input_layer = ResidualBlock(in_channels=in_channels, n_channels=n_channels, kernel_size=kernel_size)

        self.residuals = ModuleList([ResidualBlock(in_channels=n_channels, n_channels=n_channels, kernel_size=kernel_size) for _ in range(n_layers-1)])

    def forward(self, x):

        x = self.input_layer(x)
        for res_layer in self.residuals:
            x = res_layer(x)
        return x
    
class ContinousValuePredictor(Module):

    def __init__(self, in_channels, board_total_slots, n_outputs, n_convs = 2):
        super(ContinousValuePredictor, self).__init__()

        self.convos = ModuleList()
        
        for _ in range(n_convs):
            out_channels = in_channels // 2
            convo = ConvolutionalBlock(in_channels=in_channels, n_channels=out_channels, kernel_size=(1,1), padding=(0,0))
            self.convos.append(convo)
            in_channels = out_channels

        self.output_size = board_total_slots * out_channels
        self.linear = Linear(self.output_size, n_outputs)

    def forward(self, x):

        for conv in self.convos:
            x = conv(x)

        x = x.view(-1, self.output_size)
        return self.linear(x)   