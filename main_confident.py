from utils import load_dataset, load_split, load_json

import os
import numpy as np
import argparse

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn

from models import GCN, GAT, SGC

#path_json = "~/gnn/ADGCN/config"
path_json=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
EPS=1e-15


def create_parser():
    parser = argparse.ArgumentParser(description="train many times.")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--model", type=str, default="IADGCN")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_splits", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser

def loss_(model, data, mask, beta):
    logit = F.softmax(model(data.x, data.edge_index), dim=1)
    nll_loss = F.nll_loss(torch.log(logit[mask]), data.y[mask])
    entropy_loss = (-logit[mask] * torch.log(logit[mask]+EPS)).sum(dim=-1).mean()
    loss = nll_loss + beta*entropy_loss
    return loss

def train(model, optimizer, data, splits, beta):
    train_mask = splits[0].to(data.x.device)
    model.train()
    optimizer.zero_grad()
    loss = loss_(model, data, train_mask, beta)

    loss.backward()
    optimizer.step()

def val_loss_fn(model, data, splits, beta):
    model.eval()
    val_mask = splits[1].to(data.x.device)
    val_loss = loss_(model, data, val_mask, beta)
    return val_loss

def test(model, data, splits):
    train_mask = splits[0].to(data.x.device)
    val_mask = splits[1].to(data.x.device)
    test_mask = splits[2].to(data.x.device)
    model.eval()
    # hidden = 2 output
    logits, accs = model(data.x, data.edge_index), []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def run():
    # hyperhyparamter parse
    args = create_parser().parse_args()
    dataset = args.dataset
    device = args.gpu

    epochs = args.epochs
    num_seeds = args.num_seeds
    patience = args.patience
    lr = args.lr
    weight_decay = args.weight_decay
    num_splits = args.num_splits
    hidden_size = args.hidden_size
    model_cls = globals()[args.model.upper()]

    beta = args.beta

    seeds = list(range(num_seeds))

    dataset = load_dataset(dataset, T.NormalizeFeatures())
    data = dataset[0]

    result = np.zeros((4, num_seeds, num_splits))


    path_split = "/home/han/.datasets/splits"

    # For each split
    for split in range(num_splits):
        splits = load_split(os.path.join(path_split, args.dataset+'_'+str(split)+'.mask'))
        # In each split, run seeds times
        for seed in seeds:
            torch.manual_seed(seed)
            model = model_cls(dataset.num_features, dataset.num_classes, hidden_size).to(device)
            data = data.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_acc = test_acc = train_acc_best_val= 0.

            best_epoch = 0
            cnt_wait = 0
            best_val_loss=1e8

            for epoch in range(epochs):

                train(model, optimizer, data, splits, beta)
                train_acc, val_acc, tmp_test_acc = test(
                    model, data, splits
                )

                # cal val_loss
                val_loss = val_loss_fn(model, data, splits, beta)

                if val_acc > best_val_acc:
                #if val_loss < best_val_loss:
                    train_acc_best_val = train_acc
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                    best_val_epoch = epoch
                    cnt_wait = 0
                else:
                    cnt_wait += 1

                log = "Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
                if args.verbose:
                    print(log.format(epoch + 1, train_acc, best_val_acc, test_acc))

                if cnt_wait > patience:
                    break

            result[0][seed][split] = train_acc_best_val
            result[1][seed][split] = best_val_acc
            result[2][seed][split] = test_acc
            result[3][seed][split] = best_val_epoch


    path=os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if args.verbose:
        data_avr = np.mean(result, axis=(1,2))
        log = "Dataset: {}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}.(final layer)"
        print(log.format(dataset, int(data_avr[3]), data_avr[0], data_avr[1], data_avr[2]))
    else:
        data_avr = np.mean(result, axis=(1,2))
        log = "Dataset: {}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}.(final layer)"
        print(log.format(dataset, int(data_avr[3]), data_avr[0], data_avr[1], data_avr[2]))
        para = str(args.beta)+'_'+str(lr)+'_'+str(weight_decay)+'_'+str(patience)
        outfile = args.dataset+'_'+para+'.npy'
        with open(os.path.join(path, "result", args.model.lower()+'_confident', outfile), 'wb') as f:
            np.save(f, result)



if __name__ == "__main__":
    run()
