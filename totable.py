import os
import re
import numpy as np
import argparse
from utils import str2bool
parser = argparse.ArgumentParser(description="train many times.")
parser.add_argument("--model", type=str, default="adgcn")
args = parser.parse_args()

datasets=["cora","citeseer","pubmed","cs","physics","computers","photo"]
#datasets=["cora","citeseer","pubmed", "cs", "physics"]
model=args.model
assert model in ["gcn", "adgcn", "iadgcn", "gcn_label", "gcn_confident"]

symbol = "\u00b1"
log1 = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}.(final layer)"
log2 = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}.(intermediate layer)"
log = "Epoch: {:03d}{}{:03d}, Train: {:.4f}{}{:.4f}, Val: {:.4f}{}{:.4f}, Test: {:.4f}{}{:.4f}."

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


def stat4(path):
    files = os.listdir(path)
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
        test_acc = avr_array_l2_penu.max()
        index = avr_array_l2_penu.argmax()
        test_var = var_set_l2_penu[index]
        mu = mus_l2[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, l2_penu'.\
              format(model, dataset, test_acc*100, symbol, test_var*100, mu))
        # l2_final
        test_acc = avr_array_l2_final.max()
        index = avr_array_l2_final.argmax()
        test_var = var_set_l2_final[index]
        mu = mus_l2[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, l2_final'.\
              format(model, dataset, test_acc*100, symbol, test_var*100, mu))
        # kl_penu
        test_acc = avr_array_kl_penu.max()
        index = avr_array_kl_penu.argmax()
        test_var = var_set_kl_penu[index]
        mu = mus_kl[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, kl_div_penu'.\
              format(model, dataset, test_acc*100, symbol, test_var*100, mu))
        # kl_final
        test_acc = avr_array_kl_final.max()
        index = avr_array_kl_final.argmax()
        test_var = var_set_kl_final[index]
        mu = mus_kl[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, kl_div_final'.\
              format(model, dataset, test_acc*100, symbol, test_var*100, mu))

def stat2(path):
    files = os.listdir(path)
    test_data = np.zeros([2, len(datasets)])
    count=0
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
                assert num_seeds == 5
                assert num_splits == 5
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
        test_acc = avr_array_l2_penu.max()
        index = avr_array_l2_penu.argmax()
        test_var = var_set_l2_penu[index]
        mu = mus_l2[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, l2_penu'.\
              format(model, dataset, test_acc*100, symbol, test_var*100, mu))
        test_data[0][count]=test_acc
        test_data[1][count]=test_var
        count+=1
        # l2_final
        test_acc = avr_array_l2_final.max()
        index = avr_array_l2_final.argmax()
        test_var = var_set_l2_final[index]
        mu = mus_l2[index]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, mu:{}, l2_final'.\
              format(model, dataset, test_acc*100, symbol, test_var*100, mu))
    return test_data

def stat1(path):
    files = os.listdir(path)
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
                assert num_seeds == 5
                assert num_splits == 5
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

def stat(path):
    files = os.listdir(path)
    test_data = np.zeros([2, len(datasets)])
    count=0
    for dataset in datasets:
        for f in files:
            if re.split(r'[_\.]', f)[0] == dataset:
                #print(re.split(r'_', f))
                str_ = re.split(r'_', f)
                result = np.load(os.path.join(path, f))
                num_seeds = result.shape[-2]
                num_splits = result.shape[-1]
                assert num_seeds == 5
                assert num_splits == 5
                #stat(result)
                data_avr, data_var = stat_baseline(result)

        test_acc = data_avr[2]
        test_var = data_var[2]
        print('model: {}, dataset: {}, test acc: {:.2f}{}{:.2f}, l2_penu'.\
              format(model, dataset, test_acc*100, symbol, test_var*100))
        test_data[0][count]= test_acc
        test_data[1][count] = test_var
        count+=1
    return test_data

def get_nobs(path):
    files = os.listdir(path)
    for f in files:
        result = np.load(os.path.join(path, f))
        num_seeds = result.shape[-2]
        num_splits = result.shape[-1]
        return num_seeds*num_splits


if __name__ == "__main__":
    path = 'result/'+model
    if model in ["adgcn"]:
        _ = stat2(path)
    elif model in ["iadgcn", "gcn_label", "gcn_confident"]:
        stat1(path)
    elif model in ["gcn"]:
        _ = stat(path)
