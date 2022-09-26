import argparse
import torch
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from layer import gcn_norm
from model import appnp_air
from utils import train, test, set_seed
from logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--num_hops', type=int, default=16)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--log_steps', type=int, default=10)
parser.add_argument('--root', type=str, default='./')
parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
args = parser.parse_args()
print(args)
set_seed(42)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.root)
data = dataset[0]
num_features = dataset.num_features
num_classes = dataset.num_classes

data.edge_index = to_undirected(data.edge_index, data.num_nodes)
data = data.to(device)

data.edge_index, data.norm = gcn_norm(
    data.edge_index, edge_weight=None, num_nodes=data.x.size(0), dtype=data.x.dtype)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

evaluator = Evaluator(name='ogbn-arxiv')
logger = Logger(args.runs, None)

model = appnp_air(num_features, num_classes, args.hidden,
                  args.num_hops, args.dropout).to(device)
print('#Parameters:', sum(p.numel() for p in model.parameters()))

for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, data, train_idx, optimizer)
        if epoch % args.log_steps == 0:
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}%')

    logger.print_statistics(run)
logger.print_statistics()
