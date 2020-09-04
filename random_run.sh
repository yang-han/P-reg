set -eux

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" && pwd  )"
path=$DIR$"/config/"

num_seeds=5
num_splits=5


echo "**************************preggcn******************************"
model=preggcn
config_path=$path$model$".json"
echo "=======================running cora============================"
dataset="cora"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running citeseer============================"
dataset="citeseer"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running pubmed============================"
dataset="pubmed"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running cs============================"
dataset="cs"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running physics============================"
dataset="physics"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running computers============================"
dataset="computers"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running photo============================"
dataset="photo"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu


echo "**************************preggat******************************"
model=preggat
config_path=$path$model$".json"
echo "=======================running cora============================"
dataset="cora"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running citeseer============================"
dataset="citeseer"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running pubmed============================"
dataset="pubmed"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running cs============================"
dataset="cs"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running physics============================"
dataset="physics"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running computers============================"
dataset="computers"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running photo============================"
dataset="photo"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "**************************pregmlp******************************"
model=pregmlp
config_path=$path$model$".json"
echo "=======================running cora============================"
dataset="cora"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running citeseer============================"
dataset="citeseer"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running pubmed============================"
dataset="pubmed"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running cs============================"
dataset="cs"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running physics============================"
dataset="physics"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running computers============================"
dataset="computers"
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu

echo "=======================running photo============================"
dataset=photo
weight_decay=$(jq .$dataset.weight_decay $config_path)
mu=$(jq .$dataset.mu $config_path)
python main.py --dataset $dataset --model $model --weight_decay $weight_decay --num_seeds $num_seeds --num_splits $num_splits --mu $mu
echo "=========================done======================================"
