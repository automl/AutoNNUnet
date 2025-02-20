#!/bin/bash

python runscripts/train.py -m "+cross_eval=${1}_hpo_nas" "fold=range(5)" "pipeline.remove_validation_files=False" "dataset=glob(*, exclude=${1})"