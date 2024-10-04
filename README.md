# AutoNNU-Net


Integration of Automated Machine Learning (AutoML) methods into nnU-Net.

- Free software: BSD license
- Documentation: https://autonnunet.readthedocs.io.


## Features

- TODO

## Download Datasets
```bash
python autonnunet/datasets/msd_dataset.py --dataset_name=Dataset001_BrainTumour
python autonnunet/datasets/msd_dataset.py --dataset_name=Dataset002_Heart
python autonnunet/datasets/msd_dataset.py --dataset_name=Dataset003_Liver
python autonnunet/datasets/msd_dataset.py --dataset_name=Dataset004_Hippocampus
python autonnunet/datasets/msd_dataset.py --dataset_name=Dataset005_Prostate
python autonnunet/datasets/msd_dataset.py --dataset_name=Dataset006_Lung
python autonnunet/datasets/msd_dataset.py --dataset_name=Dataset007_Pancreas
python autonnunet/datasets/msd_dataset.py --dataset_name=Dataset008_HepaticVessel
python autonnunet/datasets/msd_dataset.py --dataset_name=Dataset009_Spleen
python autonnunet/datasets/msd_dataset.py --dataset_name=Dataset010_Colon
```

## Convert and Preprocess Datasets

Important: This has to be executed on the same cluster/compute environment as the target for the training to get the right nnU-Net configuration.

```bash
python runscripts/convert_and_preprocess.py -m "dataset=glob(*)"
```

## Baseline Training

```bash
python runscripts/train.py -m "dataset=glob(*)" "fold=range(5)"
```

## Compute Hyperband budgets

```bash
python autonnunet/utils/hyperband.py --b_min=15 --b_max=1000 --eta=4
```

## HPO

```bash
python runscripts/train.py --config-name=tune_smac_mf -m "dataset=glob(*)" "fold=range(5)"
```

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
