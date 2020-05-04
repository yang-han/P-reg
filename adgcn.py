from utils import load_dataset

import os
import numpy as np
import pickle
from datetime import datetime
import argparse

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F

from models import GCN, GAT, ADGCN

# from utils import train, test


def create_parser():
    parser = argparse.ArgumentParser(description="train many times.")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--model", type=str, default="ADGCN")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="store_true")
    # parser.add_argument("--output", type=str, default="")
    return parser


def train(model, optimizer, data, edge_index, train_mask):
    model.train()
    optimizer.zero_grad()
    output_1 = F.softmax(model.m1(data.x, edge_index), dim=1)
    loss_1 = F.nll_loss(torch.log(output_1[train_mask]), data.y[train_mask])
    loss_2 = torch.norm(
        F.softmax(model(data.x, edge_index), dim=1)[train_mask] - F.softmax(output_1, dim=1)[train_mask], p=2
    )
    # loss_3 = F.kl_div(
    #     F.softmax(model(data.x, edge_index), dim=1), F.softmax(output_1, dim=1)
    # )
    # mu = 0
    mu = 0.001  # cora citeseer
    # mu = 0.045 # pubmed adgcn
    # mu = 0.05
    # print(loss_1.item(), loss_2.item(), end=' | ')
    # loss = loss_1
    loss = loss_1 + mu * loss_2
    # loss = loss_1 + mu * loss_3
    loss.backward()
    optimizer.step()


def test(model, data, edge_index, train_mask, val_mask, test_mask):
    model.eval()
    logits, accs = model(data.x, edge_index), []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def test_2(model, data, edge_index, train_mask, val_mask, test_mask):
    model.eval()
    logits, accs = model.m1(data.x, edge_index), []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


args = create_parser().parse_args()

epochs = args.epochs
num_seeds = args.num_seeds
model_cls = globals()[args.model.upper()]

seeds = list(range(num_seeds))

dataset = load_dataset(args.dataset, T.NormalizeFeatures())
data = dataset[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

edges = data.edge_index.to(device)

result = np.zeros((3, num_seeds, epochs))

t_results = list()
t_results_2 = list()


# def overall_test(model, data, edges, train_mask, val_mask, test_mask, end='\n'):
#     train_acc, val_acc, tmp_test_acc = test(
#         model, data, edges, data.train_mask, data.val_mask, data.test_mask
#     )
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         test_acc = tmp_test_acc
#     log = "Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
#     # result[0][seed][epoch] = train_acc
#     # result[1][seed][epoch] = best_val_acc
#     # result[2][seed][epoch] = test_acc
#     if args.verbose:
#         print(log.format(epoch + 1, train_acc, best_val_acc, test_acc))
#     elif epoch == epochs - 1:
#         print(f"{seed}", log.format(epoch + 1, train_acc, best_val_acc, test_acc), end=end)

for seed in seeds:
    torch.manual_seed(seed)
    model = model_cls(dataset.num_features, dataset.num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    best_val_acc = test_acc = 0
    best_val_acc_2 = test_acc_2 = 0
    for epoch in range(epochs):
        train(model, optimizer, data, edges, data.train_mask)
        train_acc, val_acc, tmp_test_acc = test(
            model, data, edges, data.train_mask, data.val_mask, data.test_mask
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = "Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
        # result[0][seed][epoch] = train_acc
        # result[1][seed][epoch] = best_val_acc
        # result[2][seed][epoch] = test_acc
        if args.verbose:
            print(log.format(epoch + 1, train_acc, best_val_acc, test_acc))
        elif epoch == epochs - 1:
            print(
                f"{seed}",
                log.format(epoch + 1, train_acc, best_val_acc, test_acc),
                end=" | ",
            )

        train_acc_2, val_acc_2, tmp_test_acc_2 = test_2(
            model, data, edges, data.train_mask, data.val_mask, data.test_mask
        )
        if val_acc_2 > best_val_acc_2:
            best_val_acc_2 = val_acc_2
            test_acc_2 = tmp_test_acc_2
        log = "Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
        # result[0][seed][epoch] = train_acc
        # result[1][seed][epoch] = best_val_acc
        # result[2][seed][epoch] = test_acc
        if args.verbose:
            print(log.format(epoch + 1, train_acc_2, best_val_acc_2, test_acc_2))
        elif epoch == epochs - 1:
            print(
                f"2 layer: {seed}",
                log.format(epoch + 1, train_acc_2, best_val_acc_2, test_acc_2),
            )

    t_results.append(test_acc)
    t_results_2.append(test_acc_2)

print(np.mean(t_results), np.var(t_results))
print(np.mean(t_results_2), np.var(t_results_2))
# print(result)
