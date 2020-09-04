import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from conv import IConv
from torch_geometric.nn import GATConv, GCNConv, SGConv
from torch_geometric.nn.inits import glorot, zeros


class PREGGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(PREGGCN, self).__init__()
        self.m1 = GCN(num_features, num_classes, 64)
        self.conv = IConv(num_classes, num_classes, cached=True)

    def forward(self, x, edge_index):
        return self.m1(x, edge_index)

    def propagation(self, x, edge_index):
        return self.conv(self.m1(x, edge_index), edge_index)


class PREGGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(PREGGAT, self).__init__()
        self.m1 = GAT(num_features, num_classes, 16)
        self.conv = IConv(num_classes, num_classes, cached=True)

    def forward(self, x, edge_index):
        return self.m1(x, edge_index)

    def propagation(self, x, edge_index):
        return self.conv(self.m1(x, edge_index), edge_index)


class PREGMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(PREGMLP, self).__init__()
        self.m1 = MLP(num_features, num_classes, 64)
        self.conv = IConv(num_classes, num_classes, cached=True)

    def forward(self, x, edge_index):
        return self.m1(x, edge_index)

    def propagation(self, x, edge_index):
        return self.conv(self.m1(x, edge_index), edge_index)


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, num_classes, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(
            8 * hidden_channels, num_classes, heads=1, concat=True, dropout=0.6
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.fc1.weight)
        zeros(self.fc1.bias)
        glorot(self.fc2.weight)
        zeros(self.fc2.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x
