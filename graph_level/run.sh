set -eux

if [ $# == 1 ]; then
  gpu=$1
else
  gpu=0
fi

echo "===============================running ogbg-molbbbp=========================="
dataset=ogbg-molbbbp

echo "*****************PREGGNN*******************************"
model=gcn
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.9

echo "*****************PREGGNN-Virtual*******************************"
model=gcn-virtual
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.5

echo "*****************PREGGIN*******************************"
model=gin
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.9

echo "*****************PREGGIN-Virtual*******************************"
model=gin-virtual
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.1

echo "===============================running ogbg-moltox21=========================="
dataset=ogbg-moltox21

echo "*****************PREGGNN*******************************"
model=gcn
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.3

echo "*****************PREGGNN-Virtual*******************************"
model=gcn-virtual
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.1

echo "*****************PREGGIN*******************************"
model=gin
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.4

echo "*****************PREGGIN-Virtual*******************************"
model=gin-virtual
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.3

echo "===============================running ogbg-molesol=========================="
dataset=ogbg-molesol

echo "*****************PREGGNN*******************************"
model=gcn
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.8

echo "*****************PREGGNN-Virtual*******************************"
model=gcn-virtual
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.7

echo "*****************PREGGIN*******************************"
model=gin
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.6

echo "*****************PREGGIN-Virtual*******************************"
model=gin-virtual
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.3

echo "===============================running ogbg-molfreesolv=========================="
dataset=ogbg-molfreesolv

echo "*****************PREGGNN*******************************"
model=gcn
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.9

echo "*****************PREGGNN-Virtual*******************************"
model=gcn-virtual
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.7

echo "*****************PREGGIN*******************************"
model=gin
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.1

echo "*****************PREGGIN-Virtual*******************************"
model=gin-virtual
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 1.0

echo "===============================running ogbg-molhiv=========================="
dataset=ogbg-molhiv

echo "*****************PREGGNN*******************************"
model=gcn
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.1

echo "*****************PREGGNN-Virtual*******************************"
model=gcn-virtual
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.2

echo "*****************PREGGIN*******************************"
model=gin
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.2

echo "*****************PREGGIN-Virtual*******************************"
model=gin-virtual
python main.py --dataset $dataset --gnn $model --device $gpu --num_workers 4 --num_seeds 10 --mu 0.3
