#!/bin/bash

python runscripts/train.py -m "dataset=Dataset001_BrainTumour" "+incumbent=Dataset001_BrainTumour_$1" "fold=range(5)" "pipeline.remove_validation_files=False"
python runscripts/train.py -m "dataset=Dataset002_Heart" "+incumbent=Dataset002_Heart_$1" "fold=range(5)" "pipeline.remove_validation_files=False"
python runscripts/train.py -m "dataset=Dataset003_Liver" "+incumbent=Dataset003_Liver_$1" "fold=range(5)" "pipeline.remove_validation_files=False"
python runscripts/train.py -m "dataset=Dataset004_Hippocampus" "+incumbent=Dataset004_Hippocampus_$1" "fold=range(5)" "pipeline.remove_validation_files=False"
python runscripts/train.py -m "dataset=Dataset005_Prostate" "+incumbent=Dataset005_Prostate_$1" "fold=range(5)" "pipeline.remove_validation_files=False"
python runscripts/train.py -m "dataset=Dataset006_Lung" "+incumbent=Dataset006_Lung_$1" "fold=range(5)" "pipeline.remove_validation_files=False"
python runscripts/train.py -m "dataset=Dataset007_Pancreas" "+incumbent=Dataset007_Pancreas_$1" "fold=range(5)" "pipeline.remove_validation_files=False"
python runscripts/train.py -m "dataset=Dataset008_HepaticVessel" "+incumbent=Dataset008_HepaticVessel_$1" "fold=range(5)" "pipeline.remove_validation_files=False"
python runscripts/train.py -m "dataset=Dataset009_Spleen " "+incumbent=Dataset009_Spleen_$1" "fold=range(5)" "pipeline.remove_validation_files=False"
python runscripts/train.py -m "dataset=Dataset010_Colon" "+incumbent=Dataset010_Colon_$1" "fold=range(5)" "pipeline.remove_validation_files=False"
