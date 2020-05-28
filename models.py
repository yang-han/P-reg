import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SGConv
from sage_conv import SAGEConv
from torch_geometric.nn.inits import glorot, zeros

from conv import IConv, NIConv

class ADGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, activate, hidden_channels):
        super(ADGCN, self).__init__()
        self.m1 = GCN(num_features, num_classes, hidden_channels)
        self.conv = GCNConv(num_classes, num_classes, cached=True)
        # self.mu = nn.Parameter(torch.ones(1)* 0.2, requires_grad=False)
        if activate == "iden":
            self.activate = torch.nn.Identity(num_classes)
        elif activate == "softmax":
            self.activate = torch.nn.Softmax(dim=1)
        elif activate == "relu":
            self.activate = torch.nn.ReLU()
        else:
            print("activate error")

    def forward(self, x, edge_index):
        x = self.activate(self.m1(x, edge_index))
        return self.conv(x, edge_index)

class ADGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, activate, hidden_channels):
        super(ADGAT, self).__init__()
        self.m1 = GAT(num_features, num_classes, hidden_channels)
        self.conv = GCNConv(num_classes, num_classes, cached=True)
        # self.mu = nn.Parameter(torch.ones(1)* 0.2, requires_grad=False)
        if activate == "iden":
            self.activate = torch.nn.Identity(num_classes)
        elif activate == "softmax":
            self.activate = torch.nn.Softmax(dim=1)
        elif activate == "relu":
            self.activate = torch.nn.ReLU()
        else:
            print("activate error")

    def forward(self, x, edge_index):
        x = self.activate(self.m1(x, edge_index))
        return self.conv(x, edge_index)

class ADSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, activate, hidden_channels):
        super(ADSAGE, self).__init__()
        self.m1 = SAGE(num_features, num_classes, hidden_channels)
        self.conv = GCNConv(num_classes, num_classes, cached=True)
        # self.mu = nn.Parameter(torch.ones(1)* 0.2, requires_grad=False)
        if activate == "iden":
            self.activate = torch.nn.Identity(num_classes)
        elif activate == "softmax":
            self.activate = torch.nn.Softmax(dim=1)
        elif activate == "relu":
            self.activate = torch.nn.ReLU()
        else:
            print("activate error")

    def forward(self, x, edge_index):
        x = self.activate(self.m1(x, edge_index))
        return self.conv(x, edge_index)

class NIADGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(NIADGCN, self).__init__()
        self.m1 = GCN(num_features, num_classes, hidden_channels)
        self.conv = NIConv(num_classes, num_classes, cached=True)
        # self.mu = nn.Parameter(torch.ones(1)* 0.2, requires_grad=False)

    def forward(self, x, edge_index):
        x = self.m1(x, edge_index)
        return self.conv(x, edge_index)

class IADGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(IADGCN, self).__init__()
        self.m1 = GCN(num_features, num_classes, hidden_channels)
        self.conv = IConv(num_classes, num_classes, cached=True)
        # self.mu = nn.Parameter(torch.ones(1)* 0.2, requires_grad=False)

    def forward(self, x, edge_index):
        x = self.m1(x, edge_index)
        return self.conv(x, edge_index)

class IADGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(IADGAT, self).__init__()
        self.m1 = GAT(num_features, num_classes, hidden_channels)
        self.conv = IConv(num_classes, num_classes, cached=True)
        # self.mu = nn.Parameter(torch.ones(1)* 0.2, requires_grad=False)

    def forward(self, x, edge_index):
        x = self.m1(x, edge_index)
        return self.conv(x, edge_index)

class IADSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(IADSAGE, self).__init__()
        self.m1 = SAGE(num_features, num_classes, hidden_channels)
        self.conv = IConv(num_classes, num_classes, cached=True)
        # self.mu = nn.Parameter(torch.ones(1)* 0.2, requires_grad=False)

    def forward(self, x, edge_index):
        x = self.m1(x, edge_index)
        return self.conv(x, edge_index)

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

class IPMADGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(IPMADGAT, self).__init__()
        self.m1 = PubMedGAT(num_features, num_classes, hidden_channels)
        self.conv = IConv(num_classes, num_classes, cached=True)
        # self.mu = nn.Parameter(torch.ones(1)* 0.2, requires_grad=False)

    def forward(self, x, edge_index):
        x = self.m1(x, edge_index)
        return self.conv(x, edge_index)

class PubMedGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(PubMedGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * hidden_channels, num_classes, heads=8, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return x


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * hidden_channels, num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return x

class SAGE(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=2, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class USGC(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(USGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=2, cached=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, num_classes)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        glorot(self.fc1.weight)
        zeros(self.fc1.bias)
        glorot(self.fc2.weight)
        zeros(self.fc2.bias)

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x
