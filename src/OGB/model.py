import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import FeedForwardNet, FeedForwardNetII, FeedForwardNetIII, Prop, GCNdenseConv


class sgc_air(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, n_layers, act, pre_process=False, residual=False):
        super(sgc_air, self).__init__()
        self.num_hops = num_hops
        self.residual = residual
        self.prelu = nn.PReLU()
        self.res_fc = nn.Linear(nfeat, hidden, bias=False)
        if pre_process:
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers, dropout)
            self.process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, dropout) for i in range(num_hops)])
        else:
            self.lr_att = nn.Linear(nfeat + nfeat, 1)
            self.lr_output = FeedForwardNetII(
                nfeat, hidden, nclass, n_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.pre_process = pre_process
        self.res_fc = nn.Linear(nfeat, hidden)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(self.num_hops):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        attention_scores = []
        for i in range(self.num_hops):
            attention_scores.append(self.act(self.lr_att(
                torch.cat([input_list[0], input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        right_1 = self.lr_output(right_1)
        return right_1


class appnp_air(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden, K, dropout):
        super(appnp_air, self).__init__()
        self.lin1 = nn.Linear(num_features, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.trans = FeedForwardNetIII(
            num_features, hidden, num_classes, 4, dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.bn2 = torch.nn.BatchNorm1d(num_classes)
        self.prop = Prop(num_classes, K)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index, norm = data.x, data.edge_index, data.norm
        x = self.trans(x)
        x = self.prop(x, edge_index, norm)
        return F.log_softmax(x, dim=1)


class gcn_air(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, alpha, norm):
        super(gcn_air, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers):
            self.convs.append(GCNdenseConv(
                hidden_channels, hidden_channels, bias=norm))
        self.convs.append(torch.nn.Linear(hidden_channels, out_channels))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(
            self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, data):
        x, edge_index, norm = data.x, data.edge_index, data.norm
        _hidden = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.convs[0](x))
        x0 = x
        for _, con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(con(x, edge_index,
                       x0, norm))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)
