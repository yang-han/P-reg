import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import PREGGNN

from tqdm import tqdm
import argparse
import time
import numpy as np
import os

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from utils import soft_cross_entropy
cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type, mu):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            node_pred = model.gnn_node.m1(batch)
            node_reg = model.gnn_node(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss_1 = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss_2 = soft_cross_entropy(
                    F.softmax(node_reg, dim=-1), F.softmax(node_pred, dim=-1)
                )
            else:
                loss_1 = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss_2 = soft_cross_entropy(
                    F.softmax(node_reg, dim=-1), F.softmax(node_pred, dim=-1)
                )
            loss = loss_1+mu*loss_2
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument('--mu', type=float, default=0.5,
                        help='hyperparameter')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='number of seeds (default: 10)')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    mu = args.mu
    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)
    num_seeds = args.num_seeds
    seeds = list(range(num_seeds))

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    result = np.zeros((4, num_seeds))
    for seed in seeds:
        torch.manual_seed(seed)
        if args.gnn == 'gin':
            model = PREGGNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        elif args.gnn == 'gin-virtual':
            model = PREGGNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        elif args.gnn == 'gcn':
            model = PREGGNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        elif args.gnn == 'gcn-virtual':
            model = PREGGNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        else:
            raise ValueError('Invalid GNN type')

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        valid_curve = []
        test_curve = []
        train_curve = []

        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print('Training...')
            train(model, device, train_loader, optimizer, dataset.task_type, mu)

            print('Evaluating...')
            train_perf = eval(model, device, train_loader, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)

            print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

        if 'classification' in dataset.task_type:
            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = max(train_curve)
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
            best_train = min(train_curve)

        print('Finished training!')
        print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
        print('Test score: {}'.format(test_curve[best_val_epoch]))

        result[0][seed] = best_val_epoch
        result[1][seed] = train_curve[best_val_epoch]
        result[2][seed] = valid_curve[best_val_epoch]
        result[3][seed] = test_curve[best_val_epoch]

        if not args.verbose:
            if not os.path.exists("result"):
                os.makedirs("result")

            torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train, 'mu': mu, 'valid_curve': valid_curve, 'test_curve': test_curve, "train_curve": train_curve, 'dataset': args.dataset, "model": "iad"+args.gnn, 'epochs': args.epochs}, 'result/'+args.dataset+"_preg"+args.gnn+"_"+str(mu)+"_"+str(args.epochs)+"_"+str(seed)+"_"+str(num_seeds)+".pth")


if __name__ == "__main__":
    main()
