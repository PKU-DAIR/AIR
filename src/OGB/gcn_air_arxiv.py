from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.nn import GCNConv

import random
from logger import Logger


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


dataset = PygNodePropPredDataset(
    name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
print(dataset)
data = dataset[0]
print(data)

split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-arxiv')

train_idx = split_idx['train']
test_idx = split_idx['test']


class GCN_air(nn.Module):
    def __init__(self, dataset, hidden=256, num_layers=6, dropout=0.5):
        super(GCN_air, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.input_fc = nn.Linear(dataset.num_node_features, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        for i in range(self.num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.out_fc = nn.Linear(hidden, dataset.num_classes)
        self.lr_att = nn.Linear(dataset.num_node_features+hidden, 1)
        self.lr_att_p = nn.Linear(dataset.num_node_features+hidden, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.input_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        num_node = x.shape[0]
        x = self.bn(self.input_fc(x))
        x_input = x
        for i in range(self.num_layers):
            alpha = torch.sigmoid(self.lr_att_p(
                torch.cat([x, x_input], dim=1))).view(num_node, 1)
            x = (1-alpha)*x+(alpha)*x_input
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.bns[i](self.convs[i](x, adj_t))
        alpha = torch.sigmoid(self.lr_att_p(
            torch.cat([x, x_input], dim=1))).view(num_node, 1)
        x = (1-alpha)*x+(alpha)*x_input
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_fc(x)
        x = F.log_softmax(x, dim=1)
        return x


model = GCN_air(dataset=dataset, hidden=128, num_layers=8, dropout=0.5)
print(model)

# 转换为cpu或cuda格式
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = data.to(device)
data.adj_t = data.adj_t.to_symmetric()
train_idx = train_idx.to(device)

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()

    out = model(data)
    loss = criterion(out[train_idx], data.y.squeeze(1)[train_idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test():
    model.eval()

    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


if __name__ == '__main__':
    runs = 10
    logger = Logger(runs)

    for run in range(runs):
        set_seed(run)
        print(sum(p.numel() for p in model.parameters()))
        model.reset_parameters()

        for epoch in range(1000):
            loss = train()
            if epoch % 10 == 0:
                result = test()
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

                logger.add_result(run, result)

        logger.print_statistics(run)
    logger.print_statistics()
