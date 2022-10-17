# OmniTab: Omnivorous Pretraining for Table-based QA

This repository contains the code, pre-trained models, and data for our paper [OmniTab: Pretraining with Natural and Synthetic Data for Few-shot Table-based Question Answering](https://arxiv.org/pdf/2207.03637.pdf).

## Overview

We propose an **omnivorous** pretraining approach that consumes **natural** data to endow models with the ability to understand and align NL with tables, and **synthetic** questions to train models to perform reasoning.

<p align="center">
  <img align="middle" src="res/omnitab.png" height="350" alt="OmniTab"/>
</p>

## Installation

### Conda
Create a conda env with the name `omnitab` using `./setup.sh`.

### Docker
Dependencies are specified in `Dockerfile`.
You can either build your own image using `docker build .`, or use [pre-built image](https://hub.docker.com/repository/docker/jzbjyb/my-repo) by running `docker pull jzbjyb/my-repo`.

## Pretrained models and data
Download the best OmniTab model (OmniTab-large pretrained on natural and synthetic data and fine-tuned in full setting on WikiTableQuestions) and the WikiTableQuestions dataset from [Google Drive](https://drive.google.com/drive/u/1/folders/14IAqJb9ObVDE5oOJouhkqgd_mn11PkYY). You can download it programmatically with [gdrive](https://anaconda.org/conda-forge/gdrive) using `gdrive download -r 14IAqJb9ObVDE5oOJouhkqgd_mn11PkYY`.

## Experiment

### Pretraining
Our best-performing model is based on `bart-large`, initialized with `tapex-large`, and trained with both natural data, synthetic data, and SQL data using the following command format:
```bash
./run_model.sh $ngpu $model_size $nat_data:$syn_data:$sql_data $model_dir $init_model $bs $acc $nepoch
```
where `$ngpu` is the number of GPUs used in training, `$model_size` is `large`, `$nat_data:$syn_data:$sql_data` specifies the directory of each type of data, the pretrained model is saved at `$model_dir`, `$init_model` is `tapex-large`, `$bs` is the batch size per GPU, `$acc` is the number of gradient accumulation steps, and `$nepoch` is the number of epochs.
The effective batch size (i.e., number of examples used in each parameter update) is `$ngpu $bs $acc`.

### Finetuning
To finetune the pretrained model on WikiTableQuestions dataset, run the following command:
```bash
./finetune_predict.sh default $model_size full 50 $model_dir
```
which finetunes the pretrained model in `$model_dir` for 50 epochs using all examples (full setting).

### Inference
Run inference using the bset OmniTab model on WikiTableQuestions test split:
```bash
./run_vanilla.sh 1 omnitab_download/wtq_preprocessed omnitab_download/omnitab seq2seq 6 1 omnitab_download/omnitab/pytorch_model.bin --base-model-name facebook/bart-large --only_test --mode generate-test --output_file output.tsv
```
which saves predictions in `omnitab_download/omnitab/output.tsv`.

### Evaluation
Evaluate accuracy on WikiTableQuestions test split:
```bash
python -m utils.eval \
  --prediction omnitab_download/omnitab/output.tsv \
  --gold omnitab_download/wtq_preprocessed/gold.jsonl \
  --tagged omnitab_download/wtq_preprocessed/tagged.tsv \
  --multi_ans_sep ", "
```
We provided model predictions on WikiTableQuestions test split in `omnitab_download/omnitab/prediction.tsv`.

## Reference

```bibtex
@inproceedings{jiang-etal-2022-omnitab,
  title = "{O}mni{T}ab: Pretraining with Natural and Synthetic Data for Few-shot Table-based Question Answering",
  author = "Jiang, Zhengbao and Mao, Yi and He, Pengcheng and Neubig, Graham and Chen, Weizhu",
  booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
  month = jul,
  year = "2022",
}
```

Our codebase is based on [TaBERT](https://github.com/facebookresearch/TaBERT) and [TAPEX](https://github.com/microsoft/Table-Pretraining), so take a look their repositories if you want to explore more details.
