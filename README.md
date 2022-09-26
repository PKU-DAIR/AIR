## Model Degradation Hinders Deep Graph Neural Networks



### Requirements

Environments: Xeon Platinum 8255C (CPU), 384GB (RAM), Tesla V100 32GB (GPU), Ubuntu 16.04 (OS).

The PyTorch version we use is torch 1.7.1+cu110. Please refer to the official website -- https://pytorch.org/get-started/locally/ -- for the detailed installation instructions.

We also use PyG to preprocess the data and construct models. Please refer to the official website -- https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html -- for the detailed installation instructions.

To install other requirements:

```setup
pip install -r requirements.txt
```

 ### SGC+AIR Training:

To reproduce the experimental results on the three small citation networks, please run the following commands:

```bash
cd ./src

python sgc_air.py --dataset cora --lr 0.1 --weight_decay 5e-4 --hidden 200 --dropout 0.4 --hops 10

python sgc_air.py --dataset citeseer --lr 0.1 --weight_decay 1e-3 --hidden 200 --dropout 0.2 --hops 15

python sgc_air.py --dataset pubmed --lr 0.05 --weight_decay 5e-4 --hidden 200 --dropout 0.5 --hops 30
```

For the three OGB dataset, please run the following commands:

```bash
cd ./src/OGB
python sgc_air_ogb.py --dataset ogbn-arxiv --num-hops 5 --hidden 1024 --n-layers 6 --input-drop 0 --att-drop 0.5 --dropout 0.5 --pre-process --epochs 1000 --patience 300 --num-runs 10 --eval-every 1 --eval-batch 500000 --act leaky_relu --batch 50000 --seed 0 --gpu 0 --root ./

python sgc_air_ogb.py --dataset ogbn-products --num-hops 5 --hidden 1024 --n-layers 2 --input-drop 0.5 --att-drop 0.4 --dropout 0.2 --pre-process --epochs 1000 --patience 300 --num-runs 10 --eval-every 1 --eval-batch 500000 --act leaky_relu --batch 50000 --seed 0 --gpu 0 --root ./

python preprocess_papers100m.py --num-hops 16
python sgc_air_ogb.py --dataset ogbn-papers100M --num-hops 15 --hidden 1024 --n-layers 6 --input-drop 0 --att-drop 0.5 --dropout 0.5 --pre-process --epochs 500 --patience 300 --num-runs 3 --eval-every 1 --eval-batch 500000 --act leaky_relu --batch 50000 --seed 0 --gpu 0 --root ./
```

 ### APPNP+AIR Training:

To reproduce the experimental results on the three small citation networks, please run the following commands:

```bash
cd ./src

python appnp_air.py --dataset cora --lr 0.1 --weight_decay 3e-3 --hidden 200 --dropout 0.2 --alpha 0.95 --hops 10

python appnp_air.py --dataset citeseer --lr 0.01 --weight_decay 3e-3 --hidden 200 --dropout 0.2 --alpha 0.95 --hops 10

python apppnp_air.py --dataset pubmed --lr 0.05 --weight_decay 5e-4 --hidden 200 --dropout 0.5 --alpha 0.95 --hops 10
```

For the ogbn-arxiv dataset, please run the following commands:

```bash
cd ./src/OGB
python appnp_air_arxiv.py --root ./
```

 ### GCN+AIR Training:

To reproduce the experimental results on the three small citation networks, please run the following commands:

```bash
cd ./src

python gcn_air.py --dataset cora --lr 0.01 --weight_decay 5e-4 --hidden 50 --dropout 0.8 --hops 4

python gcn_air.py --dataset citeseer --lr 0.01 --weight_decay 1e-2 --hidden 40 --dropout 0.6 --hops 2

python gcn_air.py --dataset pubmed --lr 0.1 --weight_decay 1e-3 --hidden 40 --dropout 0.7 --hops 2
```

For the ogbn-arxiv dataset, please run the following commands:

```bash
cd ./src/OGB
python gcn_air_arxiv.py --root ./
```



### Node Classification Results:

<img src=".\node_classifi_perf.png" style="zoom:80%;" />
