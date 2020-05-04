import os
import pickle

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, CoraFull

ALL_DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "cs",
    "physics",
    "computers",
    "photo",
    # "corafull",
]

EPOCHS_CONFIG = {
    "cora": 400,
    "citeseer": 400,
    "pubmed": 400,
    "cs": 2000,
    "physics": 400,
    "computers": 1000,
    "photo": 2000,
}


class Mask(object):
    def __init__(self, train_mask, val_mask, test_mask):
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask


def load_dataset(dataset, transform=None):
    if dataset.lower() in ["cora", "citeseer", "pubmed"]:
        path = os.path.join("~", ".datasets", "Plantoid")
        dataset = Planetoid(path, dataset.lower(), transform=transform)
    elif dataset.lower() in ["cs", "physics"]:
        path = os.path.join("~", ".datasets", "Coauthor", dataset.lower())
        dataset = Coauthor(path, dataset, transform=transform)
    elif dataset.lower() in ["computers", "photo"]:
        path = os.path.join("~", ".datasets", "Amazon", dataset.lower())
        dataset = Amazon(path, dataset, transform=transform)
    elif dataset.lower() == "corafull":
        path = os.path.join("~", ".datasets", "CoraFull")
        dataset = CoraFull(path, transform=transform)
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


def train(model, optimizer, data, edge_index, train_mask):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(
        F.log_softmax(model(data.x, edge_index)[train_mask], dim=1), data.y[train_mask]
    ).backward()
    optimizer.step()


def test(model, data, edge_index, train_mask, val_mask, test_mask):
    model.eval()
    logits, accs = model(data.x, edge_index), []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def prune(edge_index, y, threshold):
    # print("pruning..")
    row, col = edge_index
    y_r, y_c = y[row], y[col]
    dot_sim = torch.bmm(
        y_r.view(y_r.size(0), 1, y_r.size(1)), y_c.view(y_c.size(0), y_c.size(1), 1)
    ).view(edge_index.size(1))
    # print("done")
    return edge_index[:, dot_sim >= threshold]


class ProactiveEdges:
    def __init__(self, y, edges, device, normalization=False):
        self.y = y.detach()
        self.num_nodes = self.y.size(0)
        self.edges = edges
        self.normalization = normalization
        self.device = device
        self._cal_sim_m()  # get sim_m adj_bool_m adj_2_bool_m

    def _cal_sim_m(self):
        y = self.y
        adj_m = get_adj_m(self.edges, self.y.size(0), dtype=torch.uint8)
        self.adj_bool_m = adj_m > 0
        # adj_m = adj_m.to(self.device, torch.float16)
        # self.adj_2_bool_m = (torch.matmul(adj_m, adj_m) > 0).to("cpu")
        del adj_m
        norm = y.norm(p=2, dim=1)
        y = (y.t() / norm).t()
        if self.normalization:
            self.y = y
        self.sim_m = torch.matmul(y, y.t())
        flatten_sim = self.sim_m.flatten().topk(self.edges.size(1) * 4)
        self.f_sim_indices = flatten_sim.indices
        self.f_sim_values = flatten_sim.values
        # print("calulate sim_m done!")

    def modify(self, threshold):
        raise NotImplementedError


class ProactiveEdges2:
    def __init__(self, y, edges, device, normalization=False):
        self.y = y.detach()
        self.num_nodes = self.y.size(0)
        self.edges = edges
        self.normalization = normalization
        self.device = device
        self._cal_sim_m()  # get sim_m adj_bool_m adj_2_bool_m

    def _cal_sim_m(self):
        y = self.y
        adj_m = get_adj_m(self.edges, self.y.size(0), dtype=torch.uint8)
        self.adj_bool_m = adj_m > 0
        # adj_m = adj_m.to(self.device, torch.float16)
        # self.adj_2_bool_m = (torch.matmul(adj_m, adj_m) > 0).to("cpu")
        del adj_m
        norm = y.norm(p=2, dim=1)
        y = (y.t() / norm).t()
        if self.normalization:
            self.y = y
        self.sim_m = torch.matmul(y, y.t())

        # print("pruning..")
        row, col = self.edges
        y_r, y_c = y[row], y[col]
        self.dot_sim = torch.bmm(
            y_r.view(y_r.size(0), 1, y_r.size(1)), y_c.view(y_c.size(0), y_c.size(1), 1)
        ).view(self.edges.size(1))
        # print("done")
        # flatten_sim = self.sim_m.to("cpu").flatten().sort()
        # self.f_sim_indices = flatten_sim.indices
        # self.f_sim_values = flatten_sim.values
        # print("calulate sim_m done!")

    def modify(self, threshold):
        raise NotImplementedError


class EdgesDropper2(ProactiveEdges2):
    def modify(self, threshold):
        return self.edges[:, self.dot_sim >= threshold]


class EdgesDropper(ProactiveEdges):
    def modify(self, threshold, adj_m=None, last_deleted=0, last_added=0):
        num_nodes = self.num_nodes
        pre_selected = int(num_nodes * num_nodes * threshold)
        edges_f = self.f_sim_indices[:pre_selected]
        num_deleted = 0
        num_added = 0
        if adj_m is None:
            adj_m = self.adj_bool_m.clone()
            edges_f = self.f_sim_indices[:pre_selected]
        else:
            num_deleted = last_deleted
            num_added = last_added
            edges_f = self.f_sim_indices[
                int(num_nodes * num_nodes * (threshold - 0.01)) : pre_selected
            ]
        for edge_f in edges_f:
            row, col = edge_f // num_nodes, edge_f % num_nodes
            if adj_m[row, col].item() or adj_m[col, row].item():
                adj_m[row, col] = False
                adj_m[col, row] = False
                num_deleted += 2
        print("pre_selected", pre_selected, "added", 0, "deleted", num_deleted, end=" ")
        return num_deleted, num_added, adj_m


class EdgesAdder(ProactiveEdges):
    def modify(self, threshold, adj_m=None):
        num_nodes = self.num_nodes
        num_cand = self.f_sim_indices.size(0)
        pre_selected = int(num_cand * threshold)
        # edges_f = self.f_sim_indices[:pre_selected]
        # adj_m = self.adj_bool_m.clone()

        if adj_m is None:
            adj_m = self.adj_bool_m.clone()
            edges_f = self.f_sim_indices[:pre_selected]
        else:
            # num_deleted = last_deleted
            # num_added = last_added
            edges_f = self.f_sim_indices[
                int(num_cand * (threshold - 0.01)) : pre_selected
            ]
        # num_deleted = 0
        # num_added = 0
        for edge_f in edges_f:
            row, col = edge_f // num_nodes, edge_f % num_nodes
            if not adj_m[row, col].item() or not adj_m[col, row].item():
                adj_m[row, col] = True
                adj_m[col, row] = True
                # num_added += 2
        # print("added", 0, "deleted", num_deleted, end=" ")
        return adj_m


# class EdgesModifier(ProactiveEdges):
#     def modify(self, threshold):
#         num_nodes = self.y.size(0)
#         num_edges = self.edges.size(1)
#         need_edges = num_edges

#         pruned_edges = prune(self.edges, self.y, threshold)
#         pruned_adj_bool_m = get_adj_m(pruned_edges, num_nodes, dtype=torch.bool)

#         edges_flattened = self.sim_m.flatten().topk(need_edges + num_nodes).indices
#         print(
#             "modify", pruned_edges.size(1), need_edges - pruned_edges.size(1), end=" "
#         )
#         return _combine_edges_2(
#             edges_flattened,
#             pruned_adj_bool_m,
#             self.adj_2_bool_m,
#             num_nodes,
#             need_edges - pruned_edges.size(1),
#         )


class OldEdgesModifier:
    def __init__(self, y, edges, device, normalization=False):
        self.y = y.detach()
        self.edges = edges
        self.normalization = normalization
        self.device = device
        self._cal_sim_m()  # get sim_m adj_bool_m adj_2_bool_m

    def _cal_sim_m(self):
        y = self.y
        adj_m = get_adj_m(self.edges, self.y.size(0), dtype=torch.uint8)
        self.adj_bool_m = adj_m > 0
        adj_m = adj_m.to(self.device, torch.float16)
        self.adj_2_bool_m = (torch.matmul(adj_m, adj_m) > 0).to("cpu")
        del adj_m
        norm = y.norm(p=2, dim=1)
        y = (y.t() / norm).t()
        if self.normalization:
            self.y = y
        self.sim_m = torch.matmul(y, y.t())
        # print("calulate sim_m done!")

    def modify(self, threshold):
        num_nodes = self.y.size(0)
        num_edges = self.edges.size(1)
        need_edges = num_edges

        pruned_edges = prune(self.edges, self.y, threshold)
        pruned_adj_bool_m = get_adj_m(pruned_edges, num_nodes, dtype=torch.bool)

        edges_flattened = self.sim_m.flatten().topk(need_edges + num_nodes).indices
        print(
            "modify", pruned_edges.size(1), need_edges - pruned_edges.size(1), end=" "
        )
        return _combine_edges_2(
            edges_flattened,
            pruned_adj_bool_m,
            self.adj_2_bool_m,
            num_nodes,
            need_edges - pruned_edges.size(1),
        )


def get_adj_m(edge_index, num_nodes, dtype=torch.bool):
    adj = torch.zeros((num_nodes, num_nodes), dtype=dtype)
    row, col = edge_index
    adj[row, col] = 1
    adj[col, row] = 1
    return adj


# Mainly add the logic that it won't add the self-loop when adding.
def _combine_edges_2(
    edges_flattened, adj_bool, except_adj_bool_num_m, num_nodes, num_compensate_edges
):
    # edges = torch.zeros((2, edges_flattened.size(0)), dtype=torch.long)
    # rows = [edge // num_nodes for edge in edges_flattened]
    # cols = [edge % num_nodes for edge in edges_flattened]
    p_c = 0
    num_compensate_edges += num_compensate_edges % 2
    adj_bool = adj_bool.clone()
    for edge_f in edges_flattened:
        if p_c == num_compensate_edges:
            break
        row, col = edge_f // num_nodes, edge_f % num_nodes
        if row == col:
            continue
        if not (except_adj_bool_num_m[row, col].item() or adj_bool[row, col].item()):
            adj_bool[row, col] = True
            adj_bool[col, row] = True
            p_c += 2

    if p_c != num_compensate_edges:
        print("!!!!! p_c != num_compensate_edges", p_c, num_compensate_edges, end=" ")
    return adj_bool.nonzero().t()

