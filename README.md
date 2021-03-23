# Federated Semi-Supervised Learning with Inter-Client Consistency & Disjoint Learning

This repository is an official Tensorflow 2 implementation of [Federated Semi-Supervised Learning with Inter-Client Consistency & Disjoint Learning ](https://openreview.net/forum?id=ce6CFXBh30h) (**ICLR 2021** and ICML-FL'20 Workshop (Best Student Paper Award))

> Currently working on PyTorch version 

## Abstract

![FedMatch](https://github.com/wyjeong/fedmatch-test/blob/main/imgs/fedmatch.jpg)

While existing federated learning approaches mostly require that clients have fully-labeled data to train on, in realistic settings, data obtained at the client-side often comes without any accompanying labels. Such deficiency of labels may result from either high labeling cost, or difficulty of annotation due to the requirement of expert knowledge. Thus the private data at each client may be either partly labeled, or completely unlabeled with labeled data being available only at the server, which leads us to a new practical federated learning problem, namely Federated Semi-Supervised Learning (FSSL). In this work, we study two essential scenarios of FSSL based on the location of the labeled data. The first scenario considers a conventional case where clients have both labeled and unlabeled data (labels-at-client), and the second scenario considers a more challenging case, where the labeled data is only available at the server (labels-at-server). We then propose a novel method to tackle the problems, which we refer to as Federated Matching (FedMatch). FedMatch improves upon naive combinations of federated learning and semi-supervised learning approaches with a new inter-client consistency loss and decomposition of the parameters for disjoint learning on labeled and unlabeled data. Through extensive experimental validation of our method in the two different scenarios, we show that our method outperforms both local semi-supervised learning and baselines which naively combine federated learning with semi-supervised learning. 

The main contributions of this work are as follows:

* Introduce a practical problem of federated learning with deficiency of supervision, namely **Federated Semi-Supervised Learning (FSSL)**, and study two different scenarios, where the local data is partly labeled (**Labels-at-Client**) or completely unlabeled (**Labels-at-Server**).
* Propose a novel method, Federated Matching (FedMatch), which learns **inter-client consistency** between multiple clients, and **decomposes model parameters** to reduce both interference between supervised and unsupervised tasks, and communication cost.  
* Show that our method, FedMatch, significantly **outperforms** both local SSL and the naive combination of FL with SSL algorithms under the conventional labels-at-client and the novel labels-at-server scenario, across multiple clients with both **non-i.i.d.** and **i.i.d. data**.
	

## Environmental Setup

Please install packages from `requirements.txt` after creating your own environment with `python 3.8.x`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Data Generation
Please see `config.py` to set your custom path for both `datasets` and `output files`.
```python
args.dataset_path = '/path/to/data/'  # for datasets
args.output_path = '/path/to/outputs/' # for logs, weights, etc.
```
Run below script to generate datasets
```bash
$ cd scripts
$ sh gen-data.sh
```
The following tasks will be generated from `CIFAR-10`.

* **`lc-biid-c10`**: `bath-iid` task in `labels-at-client` scenario
* **`lc-bimb-c10`**: `bath-non-iid` task in `labels-at-client` scenario
* **`ls-biid-c10`**: `bath-iid` task in `labels-at-server` scenario
* **`ls-bimb-c10`**: `bath-non-iid` task in `labels-at-server` scenario

## Run Experiments
To reproduce experiments, execute `train-xxx.sh` files in `scripts` folder, or you may also run the following comamnd line directly:

```bash
python3 ../main.py  --gpu 0,1,2,3,4 \
            --work-type train \
            --model fedmatch \
            --task lc-biid-c10 \
            --frac-client 0.05 \
```
Please replace an argument for **`--task`** with one of `lc-biid-c10`, `lc-bimb-c10`, `ls-biid-c10`, and `ls-bimb-c10`. For the other options (i.e. hyper-parameters, batch-size, number of rounds, etc.), please refer to `config.py` file at the project root folder.

> Note: while training, 100 clients are logically swiched across the physical gpus given by `--gpu` options (5 gpus in the above example). 

## Results
All clients and server create their own log files in `\path\to\output\logs\`, which include evaluation results, such as local & global performance and communication costs (S2C and C2S), and the experimental setups, such as learning rate, batch-size, number of rounds, etc. The log files will be updated for every comunication rounds. 

## Citations
```
@inproceedings{
    jeong2021federated,
    title={Federated Semi-Supervised Learning with Inter-Client Consistency {\&} Disjoint Learning},
    author={Wonyong Jeong and Jaehong Yoon and Eunho Yang and Sung Ju Hwang},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=ce6CFXBh30h}
}
```
