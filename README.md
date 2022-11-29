# OmniTab: Omnivorous Pretraining for Table-based QA

This repository contains the code, pre-trained models, and data for our paper [OmniTab: Pretraining with Natural and Synthetic Data for Few-shot Table-based Question Answering](https://arxiv.org/pdf/2207.03637.pdf).

## Overview

We propose an **omnivorous** pretraining approach that consumes **natural** data to endow models with the ability to understand and align NL with tables, and **synthetic** questions to train models to perform reasoning.

<p align="center">
  <img align="middle" src="res/omnitab.png" height="350" alt="OmniTab"/>
</p>

## Installation

### Conda (recommended)
Create a conda env with the name `omnitab` using `./setup.sh`.

### Docker
Dependencies are specified in `Dockerfile`.
You can either build your own image using `docker build .`, or use [pre-built image](https://hub.docker.com/repository/docker/jzbjyb/my-repo) by running `docker pull jzbjyb/my-repo`.

## Finetuning or run inference using Huggingface Transformers 🤗
You can directly load the OmniTab model (`neulab/omnitab-large-finetuned-wtq`) from HuggingFace's model hub.
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large-finetuned-wtq')
model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large-finetuned-wtq')

data = {
    'year': [1896, 1900, 1904, 2004, 2008, 2012],
    'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london']
}
table = pd.DataFrame.from_dict(data)

query = 'In which year did beijing host the Olympic Games?'
encoding = tokenizer(table=table, query=query, return_tensors='pt')

outputs = model.generate(**encoding)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# [' 2008']
```

### Model list

- Pretrained models
  - [neulab/omnitab-large](https://huggingface.co/neulab/omnitab-large): Pretrained on natural and synthetic data generated with a SQL2NL model trained in the full setting.
  - [neulab/omnitab-large-16shot](https://huggingface.co/neulab/omnitab-large-16shot): Pretrained on natural and synthetic data generated with a SQL2NL model trained in the 16-shot setting.
  - [neulab/omnitab-large-16shot](https://huggingface.co/neulab/omnitab-large-128shot): Pretrained on natural and synthetic data generated with a SQL2NL model trained in the 128-shot setting.
  - [neulab/omnitab-large-16shot](https://huggingface.co/neulab/omnitab-large-1024shot): Pretrained on natural and synthetic data generated with a SQL2NL model trained in the 1024-shot setting.
- Finetuned models
  - [neulab/omnitab-large-finetuned-wtq](https://huggingface.co/neulab/omnitab-large-finetuned-wtq): `neulab/omnitab-large` finetuned on WTQ in the full setting.
  - [neulab/omnitab-large-16shot-finetuned-wtq-16shot](https://huggingface.co/neulab/omnitab-large-16shot-finetuned-wtq-16shot): `neulab/omnitab-large-16shot` finetuned on WTQ in the 16-shot setting.
  - [neulab/omnitab-large-128shot-finetuned-wtq-128shot](https://huggingface.co/neulab/omnitab-large-128shot-finetuned-wtq-128shot): `neulab/omnitab-large-128shot` finetuned on WTQ in the 128-shot setting.
  - [neulab/omnitab-large-1024shot-finetuned-wtq-1024shot](https://huggingface.co/neulab/omnitab-large-1024shot-finetuned-wtq-1024shot): `neulab/omnitab-large-1024shot` finetuned on WTQ in the 1024-shot setting.

### Peformance
The table below contains the peformance of OmniTab models of various settings on validation/test split of WTQ before (`omnitab-large-{f}shot`) and after finetuning (`...-finetuned-wtq-{f}shot`).

| **Split**  |      **Validation**     |        **Validation**       |         **Test**        |           **Test**          |
|------------|:-----------------------:|:---------------------------:|:-----------------------:|:---------------------------:|
| **Model**  | `omnitab-large-{f}shot` | `...-finetuned-wtq-{f}shot` | `omnitab-large-{f}shot` | `...-finetuned-wtq-{f}shot` |
| **f=16**   |                   0.249 |                       0.220 |                   0.235 |                       0.233 |
| **f=128**  |                   0.299 |                       0.415 |                   0.294 |                       0.412 |
| **f=1024** |                   0.349 |                       0.534 |                   0.346 |                       0.526 |
| **full**   |                   0.411 |                       0.625 |                   0.417 |                       0.633 |

## Pretraining data and WikiTableQuestions dataset
Download the pretraining data and the WikiTableQuestions dataset from [Google Drive](https://drive.google.com/drive/u/1/folders/14IAqJb9ObVDE5oOJouhkqgd_mn11PkYY). You can download it programmatically with [gdrive](https://anaconda.org/conda-forge/gdrive) using `gdrive download -r 14IAqJb9ObVDE5oOJouhkqgd_mn11PkYY`.
It includes:
```shell
|-- pretrain_data
    |-- natural.jsonl # natural pretraining data (generated from a subset of the TAPAS pretraining data (https://github.com/google-research/tapas/blob/master/PRETRAIN_DATA.md))
    |-- synthetic.jsonl # synthetic pretraining data (generated from sql.jsonl)
    |-- sql.jsonl # SQL pretraining data (a subset of the TAPEX pretraining data (https://github.com/microsoft/Table-Pretraining#pre-training-corpus))
|-- wtq # the WikiTableQuestions dataset
    |-- fewshot_ids # ids of training examples used in few-shot finetuning
    |-- predictions_validation # predictions of various OmniTab models on the WTQ validation split
    |-- predictions_test # predictions of various OmniTab models on the WTQ test split
    |-- tagged # annotation files used in computing metrics
    |-- validation_ids.txt # ids of validation examples used in computing metrics
```

## Experiment

### Pretraining
Our best-performing model is based on `bart-large`, initialized with `tapex-large`, and trained with both natural data, synthetic data, and SQL data using the following command format:
```shell
./run_model.sh $ngpu $model_size $nat_data:$syn_data:$sql_data $model_dir $init_model $bs $acc $nepoch
```
where `$ngpu` is the number of GPUs used in training, `$model_size` is `large`, `$nat_data:$syn_data:$sql_data` specifies the directory of each type of data, the pretrained model is saved at `$model_dir`, `$init_model` is `tapex-large`, `$bs` is the batch size per GPU, `$acc` is the number of gradient accumulation steps, and `$nepoch` is the number of epochs.
The effective batch size (i.e., number of examples used in each parameter update) is `$ngpu $bs $acc`.

### Finetuning
To finetune the pretrained model on WikiTableQuestions dataset, run the following command:
```shell
./finetune_predict.sh default $model_size full 50 $model_dir
```
which finetunes the pretrained model in `$model_dir` for 50 epochs using all examples (full setting).

### Inference
Run inference using the bset OmniTab model on WikiTableQuestions dev/test split:
```shell
./run_vanilla.sh 1 omnitab_download/wtq omnitab_download/omnitab/model seq2seq 6 1 omnitab_download/omnitab/model/pytorch_model.bin --base-model-name facebook/bart-large --only_test --mode generate-[dev|test] --output_file output.tsv
```
which saves predictions in `omnitab_download/omnitab/model/output.tsv`.

### Evaluation
We provided predictions from the best OmniTab model on WikiTableQuestions dev/test split in `omnitab_download/omnitab/wtq/[dev|test].tsv`.
To compute accuracy:
```shell
python -m utils.eval \
  --prediction omnitab_download/omnitab/wtq/[dev|test].tsv \
  --tagged omnitab_download/wtq/tagged_[dev|test].tsv \
  --multi_ans_sep ", "
```

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
