# AutoNNU-Net

Integration of Automated Machine Learning (AutoML) methods into nnU-Net.

- Free software: BSD license
- Documentation: https://autonnunet.readthedocs.io.

## Installation

1. Init submodules
2. Switch branches (dev for hypersweeper and MedSAM2 for MedSAM)
3. make install

## Features

- TODO

## Download Datasets
```bash
python autonnunet/datasets/msd_dataset.py --dataset_name=<dataset>
```

## Convert and Pre-process Datasets for nnU-Net

Important: This has to be executed on the same cluster/compute environment as the target for the training to get the right nnU-Net configuration.

```bash
python runscripts/convert_and_preprocess_nnunet.py -m "dataset=glob(*)"
```

## Convert and Pre-process Datasets for MedSAM2

Important: The pre-processing for MedSAM2 must be executed locally, i.e. cannot be submitted on a SLURM cluster due to compatibility issues between pickle and multiprocessing.

```bash
python runscripts/convert_and_preprocess_medsam2.py -m "dataset=glob(*)"
```

## Baseline Training

### nnU-Net Conv
```bash
python runscripts/train.py -m "dataset=glob(*)" "fold=range(5)"
```

### nnU-Net ResM
```bash
python runscripts/train.py -m "dataset=glob(*)" "fold=range(5)" "hp_config.encoder_type=ResidualEncoderM"
```

### nnU-Net ResL
```bash
python runscripts/train.py -m "dataset=glob(*)" "fold=range(5)" "hp_config.encoder_type=ResidualEncoderL"
```

### MedSAM2
```bash
python runscripts/finetune_medsam2.py -m "dataset=glob(*)" "fold=range(5)"
```

## Compute Hyperband budgets

```bash
python runscripts/determine_hyperband_budgets.py --b_min=10 --b_max=1000 --eta=3
```

## HPO

```bash
python runscripts/train.py --config-name=tune_hpo -m "dataset=Dataset001_BrainTumour"
```

## HPO + NAS

```bash
python runscripts/train.py --config-name=tune_hpo_nas -m "dataset=Dataset001_BrainTumour"
```

## Extract & Train Incumbent

Incumbent configurations are stored in `runscripts/configs/incumbent`. You can find our incumbent configurations already in this directory.
If you want to re-create them after running the experiments, you need to run:
```bash
python runscripts/extract_incumbents.py --approach=hpo
```

Using these configs, you can than run the training of the incumbent configurations using the command:
```bash
python runscripts/train.py -m "dataset=<dataset_name>" "+incumbent=Dataset001_BrainTumour_<approach>" "fold=range(5)" "pipeline.remove_validation_files=False"
```
Please note that you could also use the model saved during the optimization. 
In our experiments, we did not store model checkpoints in the respective run directories to reduce the memory consumption.

To run nnU-Net with the incumbent configuration for the HPO approach on D01, run
```bash
python runscripts/train.py -m "dataset=Dataset001_BrainTumour" "+incumbent=Dataset001_BrainTumour_hpo" "fold=range(5)"
```

## Inference and MSD Submission

```bash
python runscripts/run_inference.py --approach=<approach>
```

Or directly submit it to SLURM:
```bash
sbatch runscripts/run_inference.sh <approach>
```

Creates the MSD submission in `output/msd_submissions`

## Plots and Tables

To generate all plots and tables in the thesis and store them in `output/thesis`, run
```bash
python runscripts/plot.py
```

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

## Common Issues

### TorchInductor fails when loading JSON, found extra data
Sometimes during optimization, jobs fail while loading cached torch inductor files.
To fix this, run
```bash
rm -rf ~/.cache/torch
rm -rf ~/.cache/triton/
rm -rf ~/.nv/ComputeCache
```