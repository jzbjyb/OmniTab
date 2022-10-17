#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

# create env
conda env create --file env.yml

# activate env
conda activate omnitab

# download resources
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
