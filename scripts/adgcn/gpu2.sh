dataset="physics"
echo "running $dataset"
model="adgcn"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" && pwd  )"
RESULT_DIR=$DIR$"/../../result/"$model

path=$DIR$"/../../config/"
config_path=$path$model$".json"
lr=$(jq .$dataset.lr $config_path)
weight_decay=$(jq .$dataset.weight_decay $config_path)
patience=$(jq .$dataset.patience $config_path)
hidden_size=$(jq .$dataset.hidden_size $config_path)
epochs=$(jq .$dataset.epochs $config_path)
num_seeds=$(jq .$dataset.num_seeds $config_path)
num_splits=$(jq .$dataset.num_splits $config_path)

kl_div=False
gpu=2
activate_set="iden"
mus="0.0001 0.001 0.01 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99"

for activate in ${activate_set}; do
  for mu in ${mus}; do
    name=${mu}_${kl_div}_${lr}_${weight_decay}_${patience}_${activate}
    echo $name
    python ../../main.py --num_seeds $num_seeds --num_splits $num_splits --model $model --dataset $dataset --lr $lr --weight_decay $weight_decay --mu $mu --kl_div $kl_div --patience $patience --activate $activate --epochs $epochs --gpu $gpu

done
done
