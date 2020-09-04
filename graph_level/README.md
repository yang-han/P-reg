# Graph-Level Experiments of P-reg

The code is modified from the codebase in [OpenGraphBenchmark](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).
We directly implemented our P-reg into the original `GNN` in the `gnn.py` file.

## Dependencies

```shell
python==3.7.6
pytorch==1.5.0
pytorch_geometric==1.4.3
numpy==1.18.1
tqdm==4.46.0
ogb==1.1.1
```

## Reproducing Experimental Results

We adopt the training and evaluation code provided in OpenGraphBenchmark.

```bash
dataset: ['ogbg-molbbbp', 'ogbg-moltox21', 'ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-molhiv']
model: ['gcn', 'gcn-virtual', 'gin', 'gin-virtual']
```

To evaluate the performance of `GCN+P-reg` with $\mu=0.9$ on `obgb-molbbbp` dataset, please run the command below:

```bash
dataset=ogbg-molbbbp
model=gcn
python main.py --dataset $dataset --gnn $model --mu 0.9 --num_workers 4 --num_seeds 10
```

For commands to reproduce all the experimental results in Table 5, please refer to `./run.sh`.
