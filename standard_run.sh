set -eux

echo "=====================================running cora========================================"
python main.py --num_splits 1 --num_seeds 10 --dataset cora --model preggat --mu 0.45
echo "=====================================running citeseer========================================"
python main.py --num_splits 1 --num_seeds 10 --dataset citeseer --model preggcn --mu 0.35
echo "=====================================running pubmed========================================"
python main.py --num_splits 1 --num_seeds 10 --dataset pubmed --model preggcn --mu 0.15

