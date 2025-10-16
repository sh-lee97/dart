# Source code for "Differentiable Acoustic Radiance Transfer"

---
### Repository Structure
Our repository comprises two main parts: one for the DART experiments in `experiments` and the other one for geometric acoustics methods written in `PyTorch` in `torch_geometric_acoustics`, planned for a release as a library, including DART.
```bash
.
├── README.md
├── experiments
│   ├── dataset
│   ├── loss.py
│   ├── plot.py
│   ├── solver.py
│   └── utils.py
├── requirements.txt
└── src
    └── torch_geometric_acoustics
```

---
### Requirements
See `requirements.txt`.

---
### Data Preparation
The `HAA` and `CR` dataset must be downloaded and preprocessed to run the optimization. First, download the `HAA` dataset from [here](https://zenodo.org/records/11195833). Place the data anywhere you want, and modify `experiments/dataset/haa.py` to match the path to the downloaded data. Same applies to the `CR` dataset; download the dataset from [here](https://zenodo.org/records/7848561). Then, run the both preprocessing and postprocessing code in `experiments/dataset/coupled_room.py`. These are required to fix several issues, e.g., wrong source position in the original data. Also, the paths in the code should be replaced to yours.

---
### Experiments
See `experiments/solver.py` for more details on training arguments.

For example, to optimize the nonparametric DART on `classroomBase` scene from `HAA` dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/solver.py \
    --dataset HAA --room complexBase \
    --no_auxiliary \
    --N_azi 12 --N_ele 12 \
    --reflection_only \
    --project dart --name nonparametric
```
For the parametric variant on the same scene:
```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/solver.py \
    --dataset HAA --room complexBase \
    --no_auxiliary \
    --N_azi 12 --N_ele 12 \
    --parametric --reflection_only \
    --project dart --name parametric
```
For the `office_to_kitchen` scene from `CR` dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/solver.py \
    --dataset CR --room office_to_kitchen \
    --no_auxiliary \
    --N_azi 12 --N_ele 12 \
    --project dart --name nonparametric
```
For the parametric variant on the same scene:
```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/solver.py \
    --dataset HAA --room complexBase \
    --no_auxiliary \
    --N_azi 12 --N_ele 12 \
    --parametric \
    --project dart --name parametric
```