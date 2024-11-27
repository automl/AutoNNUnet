#!/bin/bash
cd ..
python runscripts/train.py -m "dataset=Dataset001_BrainTumour" "+incumbent=Dataset001_BrainTumour_hpo" "fold=range(5)"
python runscripts/train.py -m "dataset=Dataset002_Heart" "+incumbent=Dataset002_Heart_hpo" "fold=range(5)"
python runscripts/train.py -m "dataset=Dataset003_Liver" "+incumbent=Dataset003_Liver_hpo" "fold=range(5)"
python runscripts/train.py -m "dataset=Dataset004_Hippocampus" "+incumbent=Dataset004_Hippocampus_hpo" "fold=range(5)"
python runscripts/train.py -m "dataset=Dataset005_Prostate" "+incumbent=Dataset005_Prostate_hpo" "fold=range(5)"
python runscripts/train.py -m "dataset=Dataset006_Lung" "+incumbent=Dataset006_Lung_hpo" "fold=range(5)"
python runscripts/train.py -m "dataset=Dataset007_Pancreas" "+incumbent=Dataset007_Pancreas_hpo" "fold=range(5)"
# python runscripts/train.py -m "dataset=Dataset008_HepaticVessel" "+incumbent=Dataset008_HepaticVessel_hpo" "fold=range(5)"
python runscripts/train.py -m "dataset=Dataset009_Spleen" "+incumbent=Dataset009_Spleen_hpo" "fold=range(5)"
python runscripts/train.py -m "dataset=Dataset010_Colon" "+incumbent=Dataset010_Colon_hpo" "fold=range(5)"

