# TSP Solver with Deep RL
This is PyTorch implementation of NEURAL COMBINATORIAL OPTIMIZATION WITH REINFORCEMENT LEARNING, Bello et al. 2016
[https://arxiv.org/abs/1611.09940]

---

## Fork modifications (DDORWA)

**Original (upstream) repository:** `ci-ke/TSP-DRL_Bello` — <https://github.com/ci-ke/TSP-DRL_Bello>
(itself a typed/refactored fork of Rintaro Kobayashi's original implementation
`Rintarooo/TSP_DRL_PtrNet` — <https://github.com/Rintarooo/TSP_DRL_PtrNet>).

This fork (`amomen9/TSP-DRL_Bello`) diverges from upstream at commit
`147ad84` ("pass mypy check"). Every change made since that fork commit is
listed below.

### 1. Checkpoint save location (commit `b8fe271`)
- **`config.py`** — `model_dir` now defaults to
  `<project_root>/Checkpoints/TSP-DRL_Bello/` (resolved from the repo location
  via the new `DEFAULT_MODEL_DIR`) instead of `./Pt/`, so trained actors are
  written into the parent project's central checkpoint tree.
- **`train.py`** — comment updated to reflect the new default model directory.
- **`Csv/1113_12_12_train20_param.csv`** — regenerated parameter log reflecting
  the new path.

### 2. Learning from a duration matrix (current change set)
Goal: let the same Pointer Network train and infer on a fixed, possibly
**asymmetric** duration matrix (a DDORWA TSP instance) instead of only random
2-D Euclidean coordinates — while leaving the original coordinate behaviour
intact and keeping coordinates available for plotting.

All changes are **additive and backward-compatible**: with the default
`input_dim = 2` every original code path (coordinate input, Euclidean tour
length, the pretrained `Pt/…_act.pt` checkpoint, `test.py`, `search.py`) behaves
exactly as before.

- **`config.py`** — new hyperparameter `input_dim` (default `2`) with a
  `-id/--input_dim` CLI flag and a backward-compat guard (`Config.__init__`
  injects `input_dim = 2` for pickles dumped before this change). `2` selects
  the original `(x, y)` coordinate encoder; `2*city_t` selects the
  duration-matrix encoder.
- **`actor.py` (`PtrNet1`)** and **`critic.py` (`PtrNet2`)** — the input
  embedding is now `nn.Linear(cfg.input_dim, cfg.embed)` instead of the
  hard-coded `nn.Linear(2, …)`. No other layer changes; with `input_dim = 2`
  the architecture is byte-for-byte the original.
- **`env.py` (`Env_tsp`)** — new methods (the original coordinate methods are
  left untouched):
  - `set_duration_matrix(D)` — attach a fixed `(city_t, city_t)` matrix
    (diagonal forced to 0; `city_t` synced to its size).
  - `matrix_node_features(D=None, normalize=True)` — per-city feature =
    concat(outgoing **row**, incoming **column**) → `(city_t, 2*city_t)`. For a
    symmetric matrix the two halves are identical.
  - `stack_matrix_nodes(n_samples, …)` — repeat one fixed instance's features
    into a `(n_samples, city_t, 2*city_t)` training batch.
  - `stack_l_matrix(tours, D=None)` — **directed** tour cost
    `Σ D[t_i, t_{i+1}] + D[t_last, t_0]`. For a symmetric matrix this equals the
    undirected Euclidean tour length, so the represented problem reduces exactly
    to the original symmetric TSP.

**Symmetric-equivalence guarantee:** for a symmetric input matrix the directed
objective and the (mirrored) node features coincide with the
undirected/coordinate formulation, so the modified Bello optimises the same
problem the original did. Verified numerically: `stack_l_matrix` equals the
original `stack_l_fast` Euclidean cost on a coordinate-derived symmetric matrix.

> The experiment-runner integration that drives this matrix mode
> (single-instance retraining as a benchmark baseline, disk save/load with the
> project's matching rules, and the post-training symmetrisation → MDS
> coordinate reconstruction used for plotting) lives in the parent DDORWA
> project and is documented in its README.

---
  
Pointer Networks is the model architecture proposed by Vinyals et al. 2015
[https://arxiv.org/abs/1506.03134]
  
This model uses attention mechanism to output a permutation of the input index.

![Screen Shot 2021-02-25 at 12 45 34 AM](https://user-images.githubusercontent.com/51239551/109026426-13756500-7703-11eb-9880-6b8be0b47b4e.png)

<br><br>
In this work, we will tackle Traveling Salesman Problem(TSP), which is one of the combinatorial optimization problems known as NP-hard. TSP seeks for the shortest tour for a salesman to visit each city exactly once.

## Training without supervised solution

In the training phase, this TSP solver will optimize 2 different types of Pointer Networks, Actor and Critic model. 

Given a graph of cities where the cities are the nodes, critic model predicts expected tour length, which is generally called state-value. Parameters of critic model are optimized as the estimated tour length catches up with the actual length calculated from the tour(city permutation) predicted by actor model. Actor model updates its policy parameters with the value called advantage which subtracts state-value from the actual tour length.

### Actor-Critic
``` 
Actor:  Defines the agent's behavior, its policy
Critic: Estimates the state-value 
```

<br><br>

## Inference
### Active Search and Sampling
In this paper, two approaches to find the best tour at inference time are proposed, which we refer to as Sampling and Active Search. 

Search strategy called Active Search takes actor model and use policy gradient for updating its parameters to find the shortest tour. Sampling simply just select the shortest tour out of 1 batch.

![Figure_1](https://user-images.githubusercontent.com/51239551/99033373-17e79900-25be-11eb-83c3-c7f4ce50c2be.png)

## Usage

### Training

First generate the pickle file contaning hyperparameter values by running the following command  

(in this example, train mode, batch size 512, 20 city nodes, 13000 steps).

```bash
python config.py -m train -b 512 -t 20 -s 13000
```
`-m train` could be replaced with `-m train_emv`. emv is the abbreviation of 'Exponential Moving Average', which doesn't need critic model. Then, go on training.
```bash
python train.py -p Pkl/train20.pkl
```  
<br><br>

### Inference
If training is done, set the configuration for inference.  
Now, you can see how the training process went from the csv files in the `Csv` dir.  
You may use my pre-trained weight `Pt/train20_1113_12_12_step14999_act.pt` which I've trained for 20 nodes'.
```bash
python config.py -m test -t 20 -s 10 -ap Pt/train20_1113_12_12_step14999_act.pt --islogger --seed 123
```
```bash
python test.py -p Pkl/test20.pkl
```
<br><br>

## Environment
I leave my own environment below. I tested it out on a single GPU.
* OS:
	* Linux(Ubuntu 18.04.5 LTS) 
* GPU:
	* NVIDIA® GeForce® RTX 2080 Ti VENTUS 11GB OC
* CPU:
	* Intel® Xeon® CPU E5640 @ 2.67GHz
* NVIDIA® Driver = 455.45.01
* Docker = 20.10.3
* [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker)(for GPU)

### Dependencies
* Python = 3.6.10
* PyTorch = 1.6.0
* numpy
* tqdm (if you need)
* matplotlib (only for plotting)

### Docker(option)
Make sure you've already installed `Docker`
```bash
docker version
```
latest `NVIDIA® Driver`
```bash
nvidia-smi
```
and `nvidia-docker2`(for GPU)
<br>
#### Usage

1. build or pull docker image

build image(this might take some time)
```bash
./docker.sh build
```
pull image from [dockerhub](https://hub.docker.com/repository/docker/docker4rintarooo/tspdrl/tags?page=1&ordering=last_updated)
```bash
docker pull docker4rintarooo/tspdrl:latest
```

2. run container using docker image(-v option is to mount directory)
```bash
./docker.sh run
```
If you don't have a GPU, you can run
```bash
./docker.sh run_cpu
```
<br><br>

## Reference
* https://github.com/higgsfield/np-hard-deep-reinforcement-learning
* https://github.com/zhengsr3/Reinforcement_Learning_Pointer_Networks_TSP_Pytorch
* https://github.com/pemami4911/neural-combinatorial-rl-pytorch
* https://github.com/MichelDeudon/neural-combinatorial-optimization-rl-tensorflow
* https://github.com/jingw2/neural-combinatorial-optimization-rl
* https://github.com/dave-yxw/rl_tsp
* https://github.com/shirgur/PointerNet
* https://github.com/MichelDeudon/encode-attend-navigate
* https://github.com/qiang-ma/HRL-for-combinatorial-optimization
* https://www.youtube.com/watch?v=mxCVgVrUw50&ab_channel=%D0%9A%D0%BE%D0%BC%D0%BF%D1%8C%D1%8E%D1%82%D0%B5%D1%80%D0%BD%D1%8B%D0%B5%D0%BD%D0%B0%D1%83%D0%BA%D0%B8
