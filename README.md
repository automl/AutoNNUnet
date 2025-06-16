<p align="center">
    <a href="./docs/images/autonnunet_logo_lm.png#gh-light-mode-only">
        <img src="./docs/images/autonnunet_logo_lm.png#gh-light-mode-only" alt="Auto-nnU-Net Logo" width="80%"/>
    </a>
    <a href="./docs/images/autonnunet_logo_dm.png#gh-dark-mode-only">
        <img src="./docs/images/autonnunet_logo_dm.png#gh-dark-mode-only" alt="Auto-nnU-Net Logo" width="80%"/>
    </a>
</p>

<div align="center">
    
![Python](https://img.shields.io/badge/Python-3.10-3776AB)
![License](https://img.shields.io/badge/License-BSD3-orange)
    
</div>

<div align="center">
    Integration of Automated Machine Learning (AutoML) methods into <a href="https://github.com/MIC-DKFZ/nnUNet">nnU-Net</a>.  
📜 Free software under BSD license.
</div>

<div align="center">
    <h3>
      <a href="#installation">🚀 Installation</a> |
      <a href="#reproduction-of-experiments">🔬 Reproduction of Experiments</a> |
      <a href="#known-issues">❗ Known Issues</a> |
      <a href="#license-&-citation">📄 License & Citation</a>
    </h3>
</div>

---

## 🗂 Repo Structure

The repository is structured in the following directories:

- `autonnunet`: Main Python package
  - `analysis`: Plotting, DeepCAVE utilities
  - `datasets`: MSD dataset handling
  - `evaluation`: Prediction tools for MSD test set
  - `experiment_planning`: Extensions to nnU-Net
  - `hnas`: Hierarchical NAS integration
  - `inference`: Inference logic
  - `utils`: Utility functions
- `data`: MSD-related data
- `output`: Locally generated results
- `results_zipped`: Pre-compressed results
- `runscripts`: Scripts for experiments
- `submodules`: External dependencies (nnU-Net, hypersweeper, etc.)
- `tests`: Unit tests
- `paper`: Paper plots and tables

---

## 🚀 Installation

🧪 *Tested on Rocky Linux 9.5 and CUDA 12.4 (not Windows)*  
⚠️ *CUDA drivers are highly recommended for a successful PyTorch installation.*

### 🐳 Containerized Setup (Recommended)

Auto-nnU-Net provides a pre-built Docker container on [Docker Hub](https://hub.docker.com/r/becktepe/autonnunet).

1. **Pull the Docker image**
```
docker pull becktepe/autonnunet:latest
```

2. **Run the container**
```
docker run -it --rm becktepe/autonnunet:latest
```

3. **(Optional) With CUDA**
```
docker run -it --rm --gpus all becktepe/autonnunet:latest
```

---

### 🧱 Manual Installation

> ⚠️ This method is more brittle due to Python/package version constraints.

1. **Clone the repository**
```
git clone https://github.com/automl/AutoNNUnet.git autonnunet
cd autonnunet
```

2. **Create and activate a conda environment**
```
conda create -n autonnunet python=3.10
conda activate autonnunet
```

3. **Install via `make`**
```
make install
```

If that fails, use manual install:
```
cd submodules/batchgenerators && git checkout master && git pull && pip install . && cd ../../
cd submodules/hypersweeper && git checkout dev && git pull && pip install . && cd ../../
cd submodules/MedSAM && git checkout MedSAM2 && git pull && pip install . && cd ../../
cd submodules/neps && git checkout master && git pull && pip install . && cd ../../
cd submodules/nnUNet && git checkout dev && git pull && pip install . && cd ../../

pip install -e ".[dev]"
```

#### 📊 Optional: Install DeepCAVE
```
pip install deepcave
```

---

## 🔬 Reproduction of Experiments

### 🧪 Cluster Setup

Use `runscripts/configs/cluster` to configure SLURM or local execution.  
Use `cluster=gpu` for SLURM and `cluster=local` for local execution.

---

### 📥 Download Datasets

Single dataset:
```
python autonnunet/datasets/msd_dataset.py --dataset_name=Dataset001_BrainTumour
```

All datasets:
```
./runscripts/download_msd.sh
```

---

### ⚙️ Preprocess for nnU-Net

```
python runscripts/convert_and_preprocess_nnunet.py -m "dataset=glob(*)" "cluster=gpu"
```

---

### 🧠 Baseline Training

**nnU-Net Conv**
```
python runscripts/train.py -m "dataset=glob(*)" "fold=range(5)"
```

**nnU-Net ResM**
```
python runscripts/train.py -m "dataset=glob(*)" "fold=range(5)" "hp_config.encoder_type=ResidualEncoderM"
```

**nnU-Net ResL**
```
python runscripts/train.py -m "dataset=glob(*)" "fold=range(5)" "hp_config.encoder_type=ResidualEncoderL"
```

---

### 🧬 MedSAM2 Fine-tuning

> Requires baseline nnU-Net training for split creation.

**Preprocess (must run locally)**
```
python runscripts/convert_and_preprocess_medsam2.py -m "dataset=glob(*)" "cluster=local"
```

**Download checkpoint**
```
cd submodules/MedSAM && mkdir checkpoints && cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
cd ../../../
```

**Fine-tune**
```
python runscripts/finetune_medsam2.py -m "dataset=glob(*)" "fold=range(5)" "cluster=gpu"
```

---

### 📈 Compute Hyperband Budgets

```
python runscripts/determine_hyperband_budgets.py --b_min=10 --b_max=1000 --eta=3
```

---

### 🔧 HPO and NAS Training

**HPO only**
```
python runscripts/train.py --config-name=tune_hpo -m "dataset=Dataset001_BrainTumour"
```

**HPO + NAS**
```
python runscripts/train.py --config-name=tune_hpo_nas -m "dataset=Dataset001_BrainTumour"
```

**HPO + HNAS**
```
python runscripts/train.py --config-name=tune_hpo_hnas -m "dataset=Dataset001_BrainTumour"
```

---

### 🏆 Extract and Train Incumbents

Extract:
```
python runscripts/extract_incumbents.py --approach=hpo
```

Train:
```
python runscripts/train.py -m "dataset=Dataset001_BrainTumour" "+incumbent=Dataset001_BrainTumour_hpo" "fold=range(5)"
```

Cross-evaluation:
```
./runscripts/train_cross_eval.sh Dataset001_BrainTumour
```

---

### 🔍 Inference and MSD Submission

Run inference:
```
python runscripts/run_inference.py --approach=hpo_nas
```

Or via SLURM:
```
sbatch runscripts/run_inference.sh hpo_nas
```

MSD submission will be saved to `output/msd_submissions`.

---

### 📊 Plots and Tables

Generate paper plots:
```
python runscripts/plot.py
```

Results will be saved in `output/paper`.

---

## ❗ Known Issues

### 💥 TorchInductor JSON Error

If PyTorch crashes with a JSON error, clear the cache:

```
rm -rf ~/.cache/torch
rm -rf ~/.cache/triton/
rm -rf ~/.nv/ComputeCache
```

---

## 📄 License & Citation

**License**: BSD-3-Clause

If you use Auto-nnU-Net, please cite:

```
@inproceedings{
    becktepe2025autonnunet,
    title={Auto-nnU-Net: Towards Automated Medical Image Segmentation},
    author={Jannis Becktepe and Leona Hennig and Steffen Oeltze-Jafra and Marius Lindauer},
    booktitle={AutoML 2025 ABCD Track},
    year={2025},
    url={https://openreview.net/forum?id=XSTIEVoEa2}
}
```

---

## 🙏 Acknowledgements

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) using the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) template.
