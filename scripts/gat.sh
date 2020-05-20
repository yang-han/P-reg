datasets="cora citeseer pubmed cs physics computers photo"
model="gat"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" && pwd  )"
RESULT_DIR=$DIR$"/../result/"$model
echo $RESULT_DIR

if [ ! -d "$RESULT_DIR"  ]; then
    mkdir $RESULT_DIR    # Control will enter here if $DIR doesn't exist.
fi

path=$DIR$"/../config/"
config_path=$path$model$".json"
gpu=0

for dataset in ${datasets}; do
    lr=$(jq .$dataset.lr $config_path)
    weight_decay=$(jq .$dataset.weight_decay $config_path)
    patience=$(jq .$dataset.patience $config_path)
    num_seeds=$(jq .$dataset.num_seeds $config_path)
    num_splits=$(jq .$dataset.num_splits $config_path)
    epochs=$(jq .$dataset.epochs $config_path)
    hidden_size=$(jq .$dataset.hidden_size $config_path)
    echo "running $dataset"
    python ../main_vanilla.py --dataset $dataset --model $model --gpu $gpu \
      --epochs $epochs --num_seeds $num_seeds --num_splits $num_splits --patience $patience \
      --hidden_size $hidden_size --lr $lr --weight_decay $weight_decay
done
