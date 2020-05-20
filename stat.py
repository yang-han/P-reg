import numpy as np
import os
import argparse
from utils import str2bool

def create_parser():
    parser = argparse.ArgumentParser(description="train many times.")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--model", type=str, default="ADGCN")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--kl_div", type=str2bool, default=False)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_splits", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--activate", type=str, default="iden")
    return parser


args = create_parser().parse_args()
mu=args.mu
kl_div=args.kl_div
lr=args.lr
weight_decay=args.weight_decay
patience=args.patience
model = args.model
dataset = args.dataset
activate = args.activate


#print("parameter: mu:{}, kl_div:{}, lr:{}, weight_decay:{}, patience:{}, model: {}".\
#      format(mu, kl_div, lr, weight_decay, patience, model))
#print("=================================================")
log1 = "Dataset: {}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}.(final layer)"
log2 = "Dataset: {}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}.(intermediate layer)"

para = str(mu)+'_'+str(kl_div)+'_'+str(lr)+'_'+str(weight_decay)+'_'+str(patience)+"_"+activate
name = dataset+'_'+ para +'.npy'
path_=os.path.join(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(path_, 'result', model.lower(), name)
result = np.load(path)

data_avr = np.mean(result, axis=(2,3))
print(log1.format(dataset, int(data_avr[0][3]), data_avr[0][0], data_avr[0][1], data_avr[0][2]))
print(log2.format(dataset, int(data_avr[1][3]), data_avr[1][0], data_avr[1][1], data_avr[1][2]))
print("==========================================")

with open(os.path.join(path_, 'result', model.lower(), para+'.txt'), 'a') as f:
    print(log1.format(dataset, int(data_avr[0][3]), data_avr[0][0], data_avr[0][1], data_avr[0][2]), file=f)
    print(log2.format(dataset, int(data_avr[1][3]), data_avr[1][0], data_avr[1][1], data_avr[1][2]), file=f)
    print("==========================================", file=f)

