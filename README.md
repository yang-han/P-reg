# [AAAI2021] Rethinking Regularization for Graph Neural Networks

This is the source code to reproduce the experimental results for *[Rethinking Graph Regularization for Graph Neural Networks](https://arxiv.org/abs/2009.02027)*.

The code for graph-level experiments is in the `./graph_level/` sub-folder.

## Dependencies

```shell
python==3.7.6
pytorch==1.5.0
pytorch_geometric==1.4.3
numpy==1.18.1
```

## Code Description

### `main.py`

The entry file. Load the datasets and models, train and evaluate the model.

### `conv.py`

The `IConv` is modified from the `torch.geometric.nn.GCNConv`, to implement the propagation of output, i.e., $\hat{A}Z$ in the paper.

Comparing to the original `GCNConv`, `IConv` removed the `bias` matrix, and replaced the `weight` matrix by an untrainable Identity matrix.

### `models.py`

`GCN`, `GAT` and `MLP` are implemented in a standard way and provided in this file.

`PREGGCN`, `PREGGAT` and `PREGMLP` have an additional method `propagation()`, which is to further propagate the output of the vanilla `GCN`,`GAT` and `MLP` models.

A typical Propagation-regularization can be computed as:

```python3
soft_cross_entropy(
    F.softmax(
        model.propagation(data.x, data.edge_index),
        dim=1
    ),
    F.softmax(
        model(data.x, data.edge_index),
        dim=1
    )
)
```

### `phi.py`

`soft_cross_entropy()`, `kl_div()`, `squared_error()` are provided in `phi.py` as different $\phi$ functions.

### `loss.py`

`LabelSmoothingLoss`, `confidence_penalty` and `laplacian_reg` are provided in `loss.py` as baselines.

### `utils.py`

Some useful functions are implemented in this file.

`generate_split()` is used to generate the random splits for each dataset. And the generated splits we used in our experiments are in the `./splits/` folder.

`Mask` is the structure that random `split` are stored.

`load_dataset(), load_split()` are provided to load the datasets and random splits.

## Reproducing Experimental Results

### Random splits (in Table 1)

Passing `--num_splits 5` to `main.py` means using the first 5 randomly generated splits provided in the `./splits/` folder. Set `--mu 0` to use the vanilla models without P-reg.

```bash
models: ['PREGGCN', 'PREGGAT', 'PREGMLP']
datasets: ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'computers', 'photo']
```

The command to train and evaluate a model is:

```bash
python main.py --dataset $dataset --model $model --mu $mu --num_seeds $num_seeds --num_splits $num_splits
```

For example, experiments with GCN+P-reg (mu=0.5) on CORA dataset for 5 splits and 5 seeds for each split:

```bash
python main.py --dataset cora --model preggcn --mu 0.5 --num_seeds 5 --num_splits 5
```

For complete commands to run all experiments, please refer to `random_run.sh`.

### Plantoid standard split (in Table 2)

Passing `--num_splits 1` to `main.py` means using the standard split of the Plaintoid datasets. Set `--mu 0` to use the vanilla models without P-reg.

```bash
models: ['PREGGCN', 'PREGGAT']
datasets: ['cora', 'citeseer', 'pubmed']
```

Commands to reproduce experimental results on CORA, CiteSeer and PubMed datasets:

```bash
# CORA GAT+P-reg mu=0.45 standard split 10 seeds
python main.py --num_splits 1 --num_seeds 10 --dataset cora --model preggat --mu 0.45
# CiteSeer GCN+P-reg mu=0.35 standard split 10 seeds
python main.py --num_splits 1 --num_seeds 10 --dataset citeseer --model preggcn --mu 0.35
# PubMed GCN+P-reg mu=0.15 standard split 10 seeds
python main.py --num_splits 1 --num_seeds 10 --dataset pubmed --model preggcn --mu 0.15
```

## Tips

1. `--model PREGGCN --mu 0` means to use the vanilla `GCN` model. (Similarly, to use vanilla `GAT` and `MLP`, please set `--mu 0`.)
2. `--num_splits 1` means to use the standard split that is provided in the Plantoid dataset (CORA, CiteSeer and PubMed), while `--num_splits 5` to `main.py` means using the first 5 randomly generated splits provided in the `./splits/` folder (for all 7 datasets).
3. In `main.py`, replace the `soft_cross_entropy` with `kl_div` or `squared_error` (provided in `phi.py`) to experiment with different $phi$ functions.
4. In `main.py`, replace the `nll_loss` to `LabelSmoothingLoss` (provided in `loss.py`) to experiment with Label Smoothing. Add `confidence_penalty` or `laplacian_reg` (provided in `loss.py`) to the original loss item to experiment with Confidence Penalty or Laplacian Regularizer.
5. In our experiments, for GCN and MLP, we use `hidden_size=64`, while for GAT, we use `hidden_size=16`.
6. In our experiments, for CORA, CiteSeer and PubMed, we use `weight_decay=5e-4`, while for CS, Physics, Computers and Photo, we use `weight_decay=0`. This is determined by the vanilla model performance.
7. By default, the training is stopped with validation accuracy no longer increases for 200 epochs (patience=200).
8. The code of other state-of-the-art methods is either from their corresponding official repository or pytoch-geometric benchmarking code. Details are attached below.
9. The code for graph-level experiments is in the `./graph_level/` folder.

## Citation

```BibTex
@inproceedings{yang2021rethinking,
  author    = {Han Yang and Kaili Ma and James Cheng},
  title     = {Rethinking Graph Regularization for Graph Neural Networks},
  booktitle = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2021, Virtual Event, February 2-9, 2021},
  pages     = {4573--4581},
  year      = {2021}
}
```
