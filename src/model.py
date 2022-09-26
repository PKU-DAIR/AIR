import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import math


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(
            self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = torch.spmm(adj, input)
        input = torch.mm(input, self.weight)
        return input


class gcn_air(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_hops):
        super(gcn_air, self).__init__()
        self.num_layers = num_hops
        self.convs = nn.ModuleList()
        self.convs.append(GraphConvolution(nfeat, nhid))
        for i in range(self.num_layers-1):
            self.convs.append(GraphConvolution(nhid, nhid))
        self.dropout = dropout
        self.out_fc = nn.Linear(nhid, nclass)
        self.lr_att = nn.Linear(nhid+nhid, 1)
        self.prelu = nn.PReLU()
        self.bn = torch.nn.BatchNorm1d(nhid)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.input_fc.reset_parameters()
        self.out_fc.reset_parameters()
        self.lr_att.reset_reset_parameters()

    def forward(self, x, adj):
        num_node = x.shape[0]
        x = self.convs[0](x, adj)
        x_input = x
        for i in range(1, self.num_layers):
            alpha = torch.sigmoid(self.lr_att(
                torch.cat([x, x_input], dim=1))).view(num_node, 1)
            x = (1-alpha)*x+(alpha)*x_input
            x = self.prelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, adj)
        x = self.out_fc(self.prelu(
            F.dropout(x, p=self.dropout, training=self.training)))
#        x = self.out_fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class appnp_air(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_hops, alpha, adj):
        super(appnp_air, self).__init__()
        self.fc_1 = nn.Linear(nfeat, nhid)
        self.fc_2 = nn.Linear(nhid, nclass)
        self.lr_att = nn.Linear(nclass+nclass, 1)

        self.dropout = dropout
        self.adj = adj
        self.alpha = alpha
        self.num_hops = num_hops

    def forward(self, features):
        input = F.dropout(
            features, self.dropout, training=self.training)
        input = self.fc_1(input)
        input = F.relu(input)
        input = F.dropout(
            input, self.dropout, training=self.training)
        input = self.fc_2(input)

        h0 = input
        h_list = []
        h_list.append(input)
        for i in range(self.num_hops):
            input = self.alpha*torch.spmm(self.adj, input)+(1-self.alpha)*h0
            h_list.append(input)

        num_node = features.shape[0]
        attention_scores = []
        for i in range(len(h_list)):
            attention_scores.append(torch.sigmoid(self.lr_att(
                torch.cat([h_list[0], h_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        W = F.softmax(attention_scores, 1)
        output = torch.mul(h_list[0], W[:, 0].view(num_node, 1))
        for i in range(1, self.num_hops):
            output = output + torch.mul(h_list[i], W[:, i].view(num_node, 1))

        return F.log_softmax(output, dim=1)


class sgc_air(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_hops):
        super(sgc_air, self).__init__()
        self.num_hops = num_hops

        self.fc_1 = nn.Linear((num_hops+1)*nfeat, nhid)
        self.fc_2 = nn.Linear(nhid, nclass)
        self.lr_att = nn.Linear(nfeat+nfeat, 1)

        self.fc_out = nn.Linear(nfeat, nclass)
        self.dropout = dropout

    def forward(self, feature_list):
        drop_features = [F.dropout(
            feature, self.dropout, training=self.training) for feature in feature_list]

        num_node = feature_list[0].shape[0]
        attention_scores = []
        for i in range(len(drop_features)):
            attention_scores.append(torch.sigmoid(self.lr_att(
                torch.cat([drop_features[0], drop_features[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        W = F.softmax(attention_scores, 1)
        output = torch.mul(drop_features[0], W[:, 0].view(num_node, 1))
        for i in range(1, self.num_hops):
            output = output + \
                torch.mul(drop_features[i], W[:, i].view(num_node, 1))

        output = self.fc_out(output)
        return F.log_softmax(output, dim=1)
