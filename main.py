import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from models import PREGGAT, PREGGCN, PREGMLP
from phi import kl_div, soft_cross_entropy, squared_error
from utils import EPS, load_dataset, load_split


def create_parser():
    parser = argparse.ArgumentParser(description="Propogation Regularization.")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--model", type=str, default="PREGGCN")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--num_seeds", type=int, default=10)
    return parser


def adloss(model, data, mask, mu):
    output_1 = F.softmax(model(data.x, data.edge_index), dim=1)
    loss_1 = F.nll_loss(torch.log(output_1[mask]+EPS), data.y[mask])

    loss_2 = soft_cross_entropy(
        F.softmax(model.propagation(data.x, data.edge_index), dim=1), output_1
    )
    loss = loss_1 + mu * loss_2

    return loss


def train(model, optimizer, data, splits, mu):
    train_mask = splits[0].to(data.x.device)
    model.train()
    optimizer.zero_grad()
    loss = adloss(model, data, train_mask, mu)
    loss.backward()
    optimizer.step()


def val_loss_fn(model, data, splits, mu):
    model.eval()
    val_mask = splits[1].to(data.x.device)
    val_loss = adloss(model, data, val_mask, mu)
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


def run():
    # hyperhyparamter parse
    args = create_parser().parse_args()
    epochs = args.epochs
    num_seeds = args.num_seeds
    mu = args.mu
    patience = args.patience
    lr = args.lr
    weight_decay = args.weight_decay
    num_splits = args.num_splits

    print(args)
    if num_splits == 1:
        print("Running using the standard split...")
    else:
        print("Running using {} random splits...".format(num_splits))

    if args.model.upper() in ['PREGGCN', 'PREGGAT', 'PREGMLP', 'PREGGAT_PUBMED']:
        model_cls = globals()[args.model.upper()]
    else:
        raise NotImplementedError("model selection error")

    seeds = list(range(num_seeds))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(args.dataset, T.NormalizeFeatures())
    data = dataset[0]

    result = np.zeros((4, num_seeds, num_splits))

    # If you want to change the data split, change the path_split to your own split
    # Attention: If you set num_splits=1, the codes will use the Plantoid standrad split.
    path_split = "splits"

    # For each split
    for split in range(num_splits):
        if num_splits == 1:
            # Using the standard split
            splits = data.train_mask, data.val_mask, data.test_mask
        else:
            splits = load_split(os.path.join(path_split, args.dataset.lower()+'_'+str(split)+'.mask'))
        # For each split, run num_seeds times
        for seed in seeds:
            torch.manual_seed(seed)
            model = model_cls(dataset.num_features, dataset.num_classes).to(device)

            data = data.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_acc = test_acc = best_val_train_acc = best_val_test_acc = 0.
            best_val_epoch = 0
            cnt_wait = 0

            for epoch in range(epochs):
                train(model, optimizer, data, splits, mu)

                train_acc, val_acc, test_acc = test(
                    model, data, splits
                )

                if val_acc > best_val_acc:
                    cnt_wait = 0

                if val_acc >= best_val_acc:
                    best_val_train_acc = train_acc
                    best_val_acc = val_acc
                    best_val_test_acc = test_acc
                    best_val_epoch = epoch
                else:
                    cnt_wait += 1

                if cnt_wait > patience:
                    break

            result[0][seed][split] = best_val_train_acc
            result[1][seed][split] = best_val_acc
            result[2][seed][split] = best_val_test_acc
            result[3][seed][split] = best_val_epoch
            print('seed:', seed, 'Epoch:', best_val_epoch, 'Train Acc:', best_val_train_acc, 'Val Acc:', best_val_acc, 'Test Acc:', best_val_test_acc)

    # summarize and store the result.
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)))
    symbol = "\u00b1"
    data_avr = np.mean(result, axis=(1,2))
    data_std = np.std(result, axis=(1,2))
    log2 = "Result: Dataset: {}, Model: {}, Avg. Epochs: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.5f}{}{:.5f}"
    print(log2.format(dataset, model_cls, int(data_avr[3]), data_avr[0], data_avr[1], data_avr[2], symbol, data_std[2]))
    para = '{:.02f}'.format(mu)+'_'+str(lr)+'_'+str(weight_decay)+'_'+str(patience)
    outfile = args.dataset.lower()+'_'+para+'.npy'
    if num_splits == 1:
        if not os.path.exists(os.path.join(path, "results", args.model.lower()+"_stand")):
            os.makedirs(os.path.join(path, "results", args.model.lower()+"_stand"))
        with open(os.path.join(path, "results", args.model.lower()+"_stand", outfile), 'wb') as f:
            np.save(f, result)
    else:
        if not os.path.exists(os.path.join(path, "results", args.model.lower())):
            os.makedirs(os.path.join(path, "results", args.model.lower()))
        with open(os.path.join(path, "results", args.model.lower(), outfile), 'wb') as f:
            np.save(f, result)


if __name__ == "__main__":
    run()
