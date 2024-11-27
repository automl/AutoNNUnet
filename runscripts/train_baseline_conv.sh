#!/bin/bash
cd ..
python runscripts/train.py -m "dataset=Dataset001_BrainTumour" "fold=range(5)" "hp_config.encoder_type=ConvolutionalEncoder"
python runscripts/train.py -m "dataset=Dataset002_Heart" "fold=range(5)" "hp_config.encoder_type=ConvolutionalEncoder"
python runscripts/train.py -m "dataset=Dataset003_Liver" "fold=range(5)" "hp_config.encoder_type=ConvolutionalEncoder"
python runscripts/train.py -m "dataset=Dataset004_Hippocampus" "fold=range(5)" "hp_config.encoder_type=ConvolutionalEncoder"
python runscripts/train.py -m "dataset=Dataset005_Prostate" "fold=range(5)" "hp_config.encoder_type=ConvolutionalEncoder"
python runscripts/train.py -m "dataset=Dataset006_Lung" "fold=range(5)" "hp_config.encoder_type=ConvolutionalEncoder"
python runscripts/train.py -m "dataset=Dataset007_Pancreas" "fold=range(5)" "hp_config.encoder_type=ConvolutionalEncoder"
python runscripts/train.py -m "dataset=Dataset008_HepaticVessel" "fold=range(5)" "hp_config.encoder_type=ConvolutionalEncoder"
python runscripts/train.py -m "dataset=Dataset009_Spleen" "fold=range(5)" "hp_config.encoder_type=ConvolutionalEncoder"
python runscripts/train.py -m "dataset=Dataset010_Colon" "fold=range(5)" "hp_config.encoder_type=ConvolutionalEncoder"

