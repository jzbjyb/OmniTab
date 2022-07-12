# OmniTab: Omnivorous Pretraining for Table-based QA

This repository contains the code, pre-trained models, and data for our paper [OmniTab: Pretraining with Natural and Synthetic Data for Few-shot Table-based Question Answering](https://arxiv.org/pdf/2207.03637.pdf).

## Overview

We propose an **omnivorous** pretraining approach that consumes **natural** data to endow models with the ability to understand and align NL with tables, and **synthetic** questions to train models to perform reasoning.

<p align="center">
  <img align="middle" src="res/omnitab.png" height="350" alt="OmniTab"/>
</p>

## Experiment

### Environment

We use docker to create the environement, with dependencies specified in `Dockerfile`.
You can build your own docker image using `docker build .`, or you can use [our built image](https://hub.docker.com/repository/docker/jzbjyb/my-repo) by running `docker pull jzbjyb/my-repo`.

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

## Pretraining data dnd pretrained models

WIP, stay tuned!

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
