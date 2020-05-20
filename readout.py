import os
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="train many times.")
parser.add_argument("--model", type=str, default="adgcn")
args = parser.parse_args()

datasets=["cora","citeseer","pubmed","cs","physics","computers","photo"]
#datasets=["cora","citeseer","pubmed","cs","photo", "computers"]
path = 'result/backup1/'+args.model

symbol = "\u00b1"
log1 = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}.(final layer)"
log2 = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}.(intermediate layer)"
log = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}."
files = os.listdir(path)

def stat_ad(result):
    data_avr = np.mean(result, axis=(2,3))
    data_var = np.var(result, axis=(2,3))
    print(re.split(r'_', f))
    print(log1.format(int(data_avr[0][3]), symbol, int(data_var[0][3]),\
                      data_avr[0][0], symbol, data_var[0][0], data_avr[0][1], symbol, data_var[0][1],\
                      data_avr[0][2], symbol, data_var[0][2]))
    print(log2.format(int(data_avr[1][3]), symbol, int(data_var[1][3]),\
                      data_avr[1][0], symbol, data_var[1][0], data_avr[1][1], symbol, data_var[1][1],\
                      data_avr[1][2], symbol, data_var[1][2]))
    print("==========================================")


def stat_baseline(result):
    data_avr = np.mean(result, axis=(1,2))
    data_var = np.var(result, axis=(1,2))
    print(re.split(r'_', f))
    print(log.format(int(data_avr[3]), symbol, int(data_var[3]),\
                      data_avr[0], symbol, data_var[0], data_avr[1], symbol, data_var[1],\
                      data_avr[2], symbol, data_var[2]))
    print("==========================================")

model = args.model
if model in ["adgcn", "adgat"]:
    stat = stat_ad
elif model in ["gcn", "gat"]:
    stat = stat_baseline
else:
    raise NotImplementedError

for dataset in datasets:
    #for folder in folders:
    #    files = os.listdir(os.path.join(path, folder))
    for f in files:
        if re.split(r'[_\.]', f)[0] == dataset:
            result = np.load(os.path.join(path, f))
            #result = np.load(os.path.join(path, folder, f))
            stat(result)



