#!/usr/bin/env bash
eval "$(conda shell.bash hook)"

# create env
conda create -n omnitab python=3.7

# activate env
conda activate omnitab

# install
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install transformers==4.24.0
pip install datasets==2.4.0
pip install nltk==3.7
