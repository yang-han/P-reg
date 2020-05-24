from utils import load_dataset, str2bool, load_split, soft_cross_entropy

import os
import numpy as np
import pickle
from datetime import datetime
import argparse

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F

from models import IADGCN, IADGAT, IADSAGE, IADMLP

def create_parser():
    parser = argparse.ArgumentParser(description="train many times.")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--model", type=str, default="IADGCN")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_splits", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--kl_div", type=int, default=1)
    return parser

def adloss(model, data, mask, mu, kl_div=1):
    output_1 = F.softmax(model.m1(data.x, data.edge_index), dim=1)
    loss_1 = F.nll_loss(torch.log(output_1[mask]), data.y[mask])

    if kl_div == 1:
        # kl_div(x, y)=y_n*(log y_n - x_n)
        loss_2 = F.kl_div(
            torch.log(output_1), F.softmax(model(data.x, data.edge_index), dim=1)
        )
    else:
        loss_2 = F.kl_div(
            F.log_softmax(model(data.x, data.edge_index), dim=1), output_1
        )
        #loss_2 = soft_cross_entropy(
        #    F.softmax(model(data.x, data.edge_index), dim=1), output_1
        #)
    #loss_2 = soft_cross_entropy(
    #    F.softmax(model(data.x, data.edge_index), dim=1), output_1
    #)
    loss = loss_1 + mu * loss_2

    return loss


def train(model, optimizer, data, splits, mu, kl_div):
    train_mask = splits[0].to(data.x.device)
    model.train()
    optimizer.zero_grad()
    loss = adloss(model, data, train_mask, mu, kl_div)
    loss.backward()
    optimizer.step()

def val_loss_fn(model, data, splits, mu):
    model.eval()
    val_mask = splits[1].to(data.x.device)
    val_loss = adloss(model, data, val_mask, mu, kl_div)
    return val_loss

def test(model, data, splits):
    train_mask = splits[0].to(data.x.device)
    val_mask = splits[1].to(data.x.device)
    test_mask = splits[2].to(data.x.device)
    model.eval()
    # final output
    logits, accs = model(data.x, data.edge_index), []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def test_2(model, data, splits):
    train_mask = splits[0].to(data.x.device)
    val_mask = splits[1].to(data.x.device)
    test_mask = splits[2].to(data.x.device)
    model.eval()
    # penu output
    logits, accs = model.m1(data.x, data.edge_index), []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def run():
    # hyperhyparamter parse
    args = create_parser().parse_args()
    epochs = args.epochs
    num_seeds = args.num_seeds
    mu = args.mu
    patience = args.patience
    lr = args.lr
    weight_decay = args.weight_decay
    device = args.gpu
    num_splits = args.num_splits
    hidden_size = args.hidden_size
    kl_div = args.kl_div
    print(args)

    model_cls = globals()[args.model.upper()]

    seeds = list(range(num_seeds))

    dataset = load_dataset(args.dataset, T.NormalizeFeatures())
    data = dataset[0]

    result = np.zeros((2, 4, num_seeds, num_splits))

    path_split = "/home/han/.datasets/splits"

    # For each split
    for split in range(num_splits):
        splits = load_split(os.path.join(path_split, args.dataset+'_'+str(split)+'.mask'))
        # In each split, run seeds times
        for seed in seeds:
            torch.manual_seed(seed)
            if args.model.upper() in ["IADGCN", "IADGAT", "IADSAGE", "IADMLP"]:
                model = model_cls(dataset.num_features, dataset.num_classes, hidden_size).to(device)
            else:
                raise NotImplementedError("model selection error")

            data = data.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_acc = test_acc = train_acc_best_val= 0.
            best_val_acc_2 = test_acc_2 = train_acc_best_val_2 = 0.

            best_epoch = 0
            best_epoch_2 = 0
            cnt_wait = 0

            for epoch in range(epochs):

                train(model, optimizer, data, splits, mu,kl_div)
                train_acc, val_acc, tmp_test_acc = test(
                    model, data, splits
                )

                train_acc_2, val_acc_2, tmp_test_acc_2 = test_2(
                    model, data, splits
                )

                if val_acc > best_val_acc:
                #if val_loss < best_val_loss:
                    train_acc_best_val = train_acc
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                    best_val_epoch = epoch
                    cnt_wait = 0
                else:
                    cnt_wait += 1

                if val_acc_2 >= best_val_acc_2:
                    train_acc_best_val_2 = train_acc_2
                    best_val_acc_2 = val_acc_2
                    test_acc_2 = tmp_test_acc_2
                    best_val_epoch_2 = epoch

                if cnt_wait > patience:
                    break

            result[0][0][seed][split] = train_acc_best_val
            result[0][1][seed][split] = best_val_acc
            result[0][2][seed][split] = test_acc
            result[0][3][seed][split] = best_val_epoch
            result[1][0][seed][split] = train_acc_best_val_2
            result[1][1][seed][split] = best_val_acc_2
            result[1][2][seed][split] = test_acc_2
            result[1][3][seed][split] = best_val_epoch_2

    path=os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if args.verbose:
        data_avr = np.mean(result, axis=(2,3))
        log1 = "Dataset: {}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}.(final layer)"
        log2 = "Dataset: {}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}.(intermediate layer)"
        print(log1.format(dataset, int(data_avr[0][3]), data_avr[0][0], data_avr[0][1], data_avr[0][2]))
        print(log2.format(dataset, int(data_avr[1][3]), data_avr[1][0], data_avr[1][1], data_avr[1][2]))
    else:
        data_avr = np.mean(result, axis=(2,3))
        log1 = "Dataset: {}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}.(final layer)"
        log2 = "Dataset: {}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}.(intermediate layer)"
        print(log1.format(dataset, int(data_avr[0][3]), data_avr[0][0], data_avr[0][1], data_avr[0][2]))
        print(log2.format(dataset, int(data_avr[1][3]), data_avr[1][0], data_avr[1][1], data_avr[1][2]))
        para = str(mu)+'_'+str(lr)+'_'+str(weight_decay)+'_'+str(patience)+"_"+str(hidden_size)+"_"+str(kl_div)
        outfile = args.dataset+'_'+para+'.npy'
        with open(os.path.join(path, "result", args.model.lower()+"_kl", outfile), 'wb') as f:
            np.save(f, result)


if __name__ == "__main__":
    run()
