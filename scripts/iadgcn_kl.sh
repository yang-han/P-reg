dataset=$1
echo "running $dataset"
model="iadgcn"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" && pwd  )"
RESULT_DIR=$DIR$"/../result/"$model$"_kl"
if [ ! -d "$RESULT_DIR"  ]; then
    mkdir $RESULT_DIR    # Control will enter here if $DIR doesn't exist.
fi

path=$DIR$"/../config/"
config_path=$path$model$".json"
lr=$(jq .$dataset.lr $config_path)
weight_decay=$(jq .$dataset.weight_decay $config_path)
patience=$(jq .$dataset.patience $config_path)
hidden_size=$(jq .$dataset.hidden_size $config_path)
epochs=$(jq .$dataset.epochs $config_path)
num_seeds=$(jq .$dataset.num_seeds $config_path)
num_splits=$(jq .$dataset.num_splits $config_path)

gpu=$2
mus="0.0001 0.001 0.01 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99"
kls="1 2"
for mu in ${mus}; do
  for kl in ${kls}; do
  python ../main_laplacian.py --dataset $dataset --model $model --gpu $gpu \
      --epochs $epochs --num_seeds $num_seeds --num_splits $num_splits --patience $patience \
      --hidden_size $hidden_size --lr $lr --weight_decay $weight_decay --mu $mu --kl_div $kl
done
done
