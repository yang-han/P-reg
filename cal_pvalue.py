
import os
import re
import numpy as np
import argparse
from utils import str2bool, t_test
parser = argparse.ArgumentParser(description="train many times.")
parser.add_argument("--model", type=str, default="adgcn")
args = parser.parse_args()

datasets=["cora","citeseer","pubmed","cs","physics","computers","photo"]
#datasets=["cora","citeseer","pubmed", "cs", "physics"]
model=args.model
assert model in ["gcn"]
dist1_path = 'result/'+model
dist2_path = 'result/ad'+model

symbol = "\u00b1"
log1 = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}.(final layer)"
log2 = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}.(intermediate layer)"
log = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}."
from totable import stat, stat2, get_nobs

dist1=stat(dist1_path)
dist2=stat2(dist2_path)
pvalue = np.zeros(dist1.shape[1])
nobs = get_nobs(dist1_path)
for i in range(dist1.shape[1]):
    pvalue[i] = t_test(dist1[0][i], dist1[1][i], nobs, dist2[0][i], dist2[1][i], nobs)
    print("{}, pvlue: {:.4f}".format(datasets[i], pvalue[i]))
