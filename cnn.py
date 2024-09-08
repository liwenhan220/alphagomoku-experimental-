import torch
from torch import nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    

class ResLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.batchnorm(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.batchnorm(x1)
        x1 += x
        x1 = self.relu(x1)
        return x1

class Policy(nn.Module):
    def __init__(self, in_channels, width, height):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, 1)
        self.batchnorm = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.ln = nn.Linear(2*width*height, width*height)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = nn.Flatten()(x)
        return self.ln(x)
    
class Value(nn.Module):
    def __init__(self, in_channels, width, height):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.ln1 = nn.Linear(width*height, 128)
        self.ln2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = nn.Flatten()(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        return nn.Tanh()(x)


class Network(nn.Module):
    def __init__(self, in_channels, num_filters, num_blocks, width, height):
        super().__init__()
        self.feature_extractor = nn.Sequential(ConvLayer(in_channels, num_filters))
        self.num_blocks = num_blocks
        for _ in range(num_blocks):
            self.feature_extractor.append(ResLayer(num_filters))
        self.policy_head = Policy(num_filters, width, height)
        self.value_head = Value(num_filters, width, height)

    def forward(self, x):
        x = self.feature_extractor(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
        

class DummyNN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.randn((len(x), 9)), torch.zeros(len(x))
    

def test_NN(): 
    x = torch.rand((1, 2, 15, 15)).cuda()

    # conv = ConvLayer(2, 256)
    # res = ResLayer(256)
    # policy = Policy(256)
    # value = Value(256)

    import time

    net = Network(2, 256, 20, 15, 15).cuda()
    net(x)

    old_t = time.time()
    net(x)
    print(time.time()-old_t)

# test_NN()