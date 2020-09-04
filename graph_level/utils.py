import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy import stats
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from sklearn.manifold import TSNE

EPS=1e-15

ALL_DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "cs",
    "physics",
    "computers",
    "photo"
]


class Mask(object):
    def __init__(self, train_mask, val_mask, test_mask):
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask


def load_dataset(dataset, transform=None):
    if dataset.lower() in ["cora", "citeseer", "pubmed"]:
        path = os.path.join(".datasets", "Plantoid")
        dataset = Planetoid(path, dataset.lower(), transform=transform)
    elif dataset.lower() in ["cs", "physics"]:
        path = os.path.join(".datasets", "Coauthor", dataset.lower())
        dataset = Coauthor(path, dataset, transform=transform)
    elif dataset.lower() in ["computers", "photo"]:
        path = os.path.join(".datasets", "Amazon", dataset.lower())
        dataset = Amazon(path, dataset, transform=transform)
    else:
        print("Dataset not supported!")
        assert False
    return dataset


def generate_percent_split(dataset, seed, train_percent=70, val_percent=20):
    torch.manual_seed(seed)
    dataset = load_dataset(dataset)
    data = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero().flatten()
        num_c = all_c_idx.size(0)
        train_num_per_c = num_c * train_percent // 100
        val_num_per_c = num_c * val_percent // 100
        perm = torch.randperm(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask
    return train_mask, val_mask, test_mask


def generate_split(dataset, seed=0, train_num_per_c=20, val_num_per_c=30):
    torch.manual_seed(seed)
    dataset = load_dataset(dataset)
    data = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = torch.randperm(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask
    return train_mask, val_mask, test_mask


def load_split(path):
    mask = torch.load(path)
    return mask.train_mask, mask.val_mask, mask.test_mask


def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def soft_cross_entropy(predict, soft_target):
    return -(soft_target * torch.log(predict+EPS)).sum(dim=1).mean()

