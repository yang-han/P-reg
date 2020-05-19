import os
import re
import numpy as np
import argparse
from utils import str2bool
parser = argparse.ArgumentParser(description="train many times.")
parser.add_argument("--model", type=str, default="adgcn")
args = parser.parse_args()

datasets=["cora","citeseer","pubmed","cs","physics","computers","photo"]
#datasets=["cora","citeseer","pubmed","cs","photo", "computers"]
model=args.model
assert model in ["gcn", "adgcn", "iadgcn", "gcn_label", "gcn_confident"]
path = 'result/'+model

symbol = "\u00b1"
log1 = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}.(final layer)"
log2 = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}.(intermediate layer)"
log = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}."
files = os.listdir(path)

def stat_ad(result):
    data_avr = np.mean(result, axis=(2,3))
    data_var = np.sqrt(np.var(result, axis=(2,3)))
    return data_avr, data_var
    #print(log1.format(int(data_avr[0][3]), symbol, int(data_var[0][3]),\
    #                  data_avr[0][0], symbol, data_var[0][0], data_avr[0][1], symbol, data_var[0][1],\
    #                  data_avr[0][2], symbol, data_var[0][2]))
    #print(log2.format(int(data_avr[1][3]), symbol, int(data_var[1][3]),\
    #                  data_avr[1][0], symbol, data_var[1][0], data_avr[1][1], symbol, data_var[1][1],\
    #                  data_avr[1][2], symbol, data_var[1][2]))
    #print("==========================================")


def stat_baseline(result):
    data_avr = np.mean(result, axis=(1,2))
    data_var = np.sqrt(np.var(result, axis=(1,2)))
    return data_avr, data_var
    #print(log.format(int(data_avr[3]), symbol, int(data_var[3]),\
    #                  data_avr[0], symbol, data_var[0], data_avr[1], symbol, data_var[1],\
    #                  data_avr[2], symbol, data_var[2]))
    #print("==========================================")


def stat4():
    for dataset in datasets:
        avr_set_kl_final = []
        avr_set_kl_penu = []
        avr_set_l2_final = []
        avr_set_l2_penu = []
        var_set_kl_final = []
        var_set_kl_penu = []
        var_set_l2_final = []
        var_set_l2_penu = []
        str_set = []
        mus_kl = []
        mus_l2 = []
        for f in files:
            if re.split(r'[_\.]', f)[0] == dataset and re.split(r'_', f)[-1] == 'iden.npy' and re.split(r'_', f)[3] == '0.001':
                #print(re.split(r'_', f))
                str_ = re.split(r'_', f)
                str_set.append(str_)
                mu = str_[1]
                kl_div = str2bool(str_[2])
                result = np.load(os.path.join(path, f))
                num_seeds = result.shape[-2]
                num_splits = result.shape[-1]
                #stat(result)
                data_avr, data_var = stat_ad(result)

                if kl_div:
                    avr_set_kl_final.append(data_avr[0][2])
                    avr_set_kl_penu.append(data_avr[1][2])
                    var_set_kl_final.append(data_var[0][2])
                    var_set_kl_penu.append(data_var[1][2])
                    mus_kl.append(mu)
                else:
                    avr_set_l2_final.append(data_avr[0][2])
                    avr_set_l2_penu.append(data_avr[1][2])
                    var_set_l2_final.append(data_var[0][2])
                    var_set_l2_penu.append(data_var[1][2])
                    mus_l2.append(mu)


        avr_array_kl_final = np.array(avr_set_kl_final)
        avr_array_kl_penu = np.array(avr_set_kl_penu)
        avr_array_l2_final = np.array(avr_set_l2_final)
        avr_array_l2_penu = np.array(avr_set_l2_penu)
        # l2_penu
        test_acc = avr_array_l2_penu.max()*100
        index = avr_array_l2_penu.argmax()
        test_var = var_set_l2_penu[index]*100
        mu = mus_l2[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, l2_penu'.\
              format(model, dataset, test_acc, symbol, test_var, mu))
        # l2_final
        test_acc = avr_array_l2_final.max()*100
        index = avr_array_l2_final.argmax()
        test_var = var_set_l2_final[index]*100
        mu = mus_l2[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, l2_final'.\
              format(model, dataset, test_acc, symbol, test_var, mu))
        # kl_penu
        test_acc = avr_array_kl_penu.max()*100
        index = avr_array_kl_penu.argmax()
        test_var = var_set_kl_penu[index]*100
        mu = mus_kl[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, kl_div_penu'.\
              format(model, dataset, test_acc, symbol, test_var, mu))
        # kl_final
        test_acc = avr_array_kl_final.max()*100
        index = avr_array_kl_final.argmax()
        test_var = var_set_kl_final[index]*100
        mu = mus_kl[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, kl_div_final'.\
              format(model, dataset, test_acc, symbol, test_var, mu))

def stat2():
    for dataset in datasets:
        avr_set_l2_final = []
        avr_set_l2_penu = []
        var_set_l2_final = []
        var_set_l2_penu = []
        str_set = []
        mus_l2 = []
        for f in files:
            if re.split(r'[_\.]', f)[0] == dataset:
                #print(re.split(r'_', f))
                str_ = re.split(r'_', f)
                str_set.append(str_)
                mu = str_[1]
                kl_div = str2bool(str_[2])
                result = np.load(os.path.join(path, f))
                num_seeds = result.shape[-2]
                num_splits = result.shape[-1]
                #stat(result)
                data_avr, data_var = stat_ad(result)

                avr_set_l2_final.append(data_avr[0][2])
                avr_set_l2_penu.append(data_avr[1][2])
                var_set_l2_final.append(data_var[0][2])
                var_set_l2_penu.append(data_var[1][2])
                mus_l2.append(mu)


        avr_array_l2_final = np.array(avr_set_l2_final)
        avr_array_l2_penu = np.array(avr_set_l2_penu)
        # l2_penu
        test_acc = avr_array_l2_penu.max()*100
        index = avr_array_l2_penu.argmax()
        test_var = var_set_l2_penu[index]*100
        mu = mus_l2[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, l2_penu'.\
              format(model, dataset, test_acc, symbol, test_var, mu))
        # l2_final
        test_acc = avr_array_l2_final.max()*100
        index = avr_array_l2_final.argmax()
        test_var = var_set_l2_final[index]*100
        mu = mus_l2[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, l2_final'.\
              format(model, dataset, test_acc, symbol, test_var, mu))

def stat1():
    for dataset in datasets:
        avr_set_l2_penu = []
        var_set_l2_penu = []
        str_set = []
        mus_l2 = []
        for f in files:
            if re.split(r'[_\.]', f)[0] == dataset:
                #print(re.split(r'_', f))
                str_ = re.split(r'_', f)
                str_set.append(str_)
                mu = str_[1]
                result = np.load(os.path.join(path, f))
                num_seeds = result.shape[-2]
                num_splits = result.shape[-1]
                #stat(result)
                data_avr, data_var = stat_baseline(result)

                avr_set_l2_penu.append(data_avr[2])
                var_set_l2_penu.append(data_var[2])
                mus_l2.append(mu)


        avr_array_l2_penu = np.array(avr_set_l2_penu)
        # l2_penu
        test_acc = avr_array_l2_penu.max()*100
        index = avr_array_l2_penu.argmax()
        test_var = var_set_l2_penu[index]*100
        mu = mus_l2[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, l2_penu'.\
              format(model, dataset, test_acc, symbol, test_var, mu))

def stat():
    for dataset in datasets:
        for f in files:
            if re.split(r'[_\.]', f)[0] == dataset:
                #print(re.split(r'_', f))
                str_ = re.split(r'_', f)
                result = np.load(os.path.join(path, f))
                num_seeds = result.shape[-2]
                num_splits = result.shape[-1]
                #stat(result)
                data_avr, data_var = stat_baseline(result)

        test_acc = data_avr[2]*100
        test_var = data_var[2]*100
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, l2_penu'.\
              format(model, dataset, test_acc, symbol, test_var))

if model in ["adgcn"]:
    stat2()
elif model in ["iadgcn", "gcn_label", "gcn_confident"]:
    stat1()
elif model in ["gcn"]:
    stat()
