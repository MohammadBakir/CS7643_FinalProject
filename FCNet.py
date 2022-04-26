import torch
from torch import nn


class FCNet(nn.Module):
    def __init__(self, in_shape):
        super(FCNet, self).__init__()
        self.fc_layer1 = nn.Linear(in_shape, in_shape * 4)
        self.bn_layer1 = nn.BatchNorm1d(in_shape * 4)
        self.fc_layer2 = nn.Linear(in_shape * 4, in_shape * 8)
        self.bn_layer2 = nn.BatchNorm1d(in_shape * 8)
        self.fc_layer3 = nn.Linear(in_shape * 8, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.bn_layer1(torch.relu(self.fc_layer1(x)))
        x = self.bn_layer2(torch.relu(self.fc_layer2(x)))
        x = self.fc_layer3(x)
        return x.reshape(x.shape[0], -1)
