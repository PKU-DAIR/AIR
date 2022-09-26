import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul
import torch.nn.functional as F


# adapted from https://github.com/chennnM/GBP
class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias='bn'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output


# MLP apply initial residual
class residualayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(residualayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(
            self.in_features, self.out_features))
        self.alpha = 0.5
        self.reset_parameters()
        self.bias = nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h0):
        #        support = (1-self.alpha)*input+self.alpha*h0
        support = input+h0
        output = torch.mm(support, self.weight)
        output = self.bias(output)
#        if self.in_features == self.out_features:
#            output = output+input
        return output

# MLP apply initial residual


class denseresidualayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(denseresidualayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(
            self.in_features, self.out_features))
        self.alpha = 0.5
        self.reset_parameters()
        self.bias = nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h0):
        #        support = (1-self.alpha)*input+self.alpha*h0
        support = input+h0
        output = torch.mm(support, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output+input
        return output

# adapted from dgl sign


class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(self.bns[layer_id](x)))
        return x


class FeedForwardNetII(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNetII, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.in_feats = in_feats
        self.hidden = hidden
        self.out_feats = out_feats
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats, bias=False))
        else:
            self.layers.append(Dense(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(denseresidualayer(hidden, hidden))
            self.layers.append(Dense(hidden, out_feats))
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        x = self.layers[0](x)
        h0 = x
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                continue
            elif layer_id == self.n_layers - 1:
                x = self.dropout(self.prelu(x))
                x = layer(x)
            else:
                x = self.dropout(self.prelu(x))
                x = layer(x, h0)
        return x


class FeedForwardNetIII(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNetII, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.in_feats = in_feats
        self.hidden = hidden
        self.out_feats = out_feats
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats, bias=False))
        else:
            self.layers.append(Dense(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(residualayer(hidden, hidden))
            self.layers.append(Dense(hidden, out_feats))
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        x = self.layers[0](x)
        h0 = x
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                continue
            elif layer_id == self.n_layers - 1:
                x = self.dropout(self.prelu(x))
                x = layer(x)
            else:
                x = self.dropout(self.prelu(x))
                x = layer(x, h0)
        return x


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = nn.Linear(num_classes, 1)
        self.lr_att = nn.Linear(num_classes+num_classes, 1)
        self.att_drop = torch.nn.Dropout(0.5)
        self.alpha = 0.95
        self.lr_att = nn.Linear(num_classes+num_classes, 1)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, x.size(0), dtype=x.dtype)
        x0 = x
        preds = []
        preds.append(x)
        for _ in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            x = self.alpha*x+(1-self.alpha)*x0
            preds.append(x)
        num_node = x.shape[0]
        attention_scores = []
        for i in range(len(preds)):
            attention_scores.append(torch.sigmoid(self.lr_att(
                torch.cat([preds[0], preds[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        W = F.softmax(attention_scores, 1)
        output = torch.mul(preds[0], self.att_drop(W[:, 0].view(num_node, 1)))
        for i in range(1, self.K):
            output = output + \
                torch.mul(preds[i], self.att_drop(W[:, i].view(num_node, 1)))
        return output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)

    def reset_parameters(self):
        self.proj.reset_parameters()


class GCNdenseConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = True,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias='bn', **kwargs):

        super(GCNdenseConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.lr_att = nn.Linear(out_channels+out_channels, 1)

        self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(in_channels, out_channels))
        self.norm = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)

    def forward(self, x, edge_index, h0, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, x.size(0), dtype=x.dtype)

        attention_score = self.lr_att(torch.cat([x, h0], dim=1))
        support = torch.matmul(x+h0, self.weight1)
#        initial = torch.mul(attention_score, h0)+torch.mul((1 -
#                                                            attention_score), torch.matmul(h0, self.weight2))
        out = self.propagate(edge_index, x=torch.mul(attention_score, support)+torch.mul((1 - attention_score), h0),
                             norm=norm)
        out = self.norm(out)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
