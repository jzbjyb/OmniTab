#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import List
import logging
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from functools import partial
import wandb

import torch
import random

import torch.nn as nn
import torch.distributed
from fairseq.data import GroupedIterator
from fairseq.optim.adam import FairseqAdam
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import PolynomialDecaySchedule
from fairseq.options import eval_str_list
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import json
import numpy as np

from omnitab.vanilla_table_bert import VanillaTableBert
from omnitab.vertical.config import VerticalAttentionTableBertConfig
from omnitab.vertical.dataset import VerticalAttentionTableBertDataset
from omnitab.vertical.vertical_attention_table_bert import VerticalAttentionTableBert
from utils.comm import init_distributed_mode, init_signal_handler
from omnitab.config import TableBertConfig
from omnitab.dataset import TableDataset
from utils.evaluator import Evaluator
from utils.trainer import Trainer
from utils.util import get_logger


task_dict = {
    'vanilla': {
        'dataset': TableDataset,
        'config': TableBertConfig,
        'model': VanillaTableBert
    },
    'vertical_attention': {
        'dataset': VerticalAttentionTableBertDataset,
        'config': VerticalAttentionTableBertConfig,
        'model': VerticalAttentionTableBert
    }
}


def parse_train_arg():
    parser = ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        default='vanilla',
                        choices=['vanilla', 'vertical_attention'])
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--cpu",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--save-init', action='store_true', help='save the initial checkpoint')

    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)

    parser.add_argument('--base-model-name', type=str, required=False, default=None,
                        help='Used to overide the default model specified in config file from the data directory')
    parser.add_argument("--table-bert-extra-config", type=json.loads, default='{}')
    parser.add_argument('--no-init', action='store_true', default=False)
    # parser.add_argument('--config-file', type=Path, help='table_bert config file if do not use pre-trained BERT table_bert.')
    parser.add_argument('--objective-function', type=str, default='mlm')

    # wandb
    parser.add_argument('--entity', type=str, default='jzbjyb')
    parser.add_argument('--project', type=str, default='structured-pretrain')
    parser.add_argument('--name', type=str, default='test')

    # distributed training
    parser.add_argument("--ddp-backend", type=str, default='pytorch', choices=['pytorch', 'apex'])
    parser.add_argument("--local_rank", "--local-rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--master-port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--debug-slurm", action='store_true',
                        help="Debug multi-GPU / multi-node within a SLURM job")

    # training details
    parser.add_argument("--train-batch-size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--max-epoch", default=-1, type=int)
    # parser.add_argument("--total-num-update", type=int, default=1000000, help="Number of steps to train for")
    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr-scheduler", type=str, default='polynomial_decay', help='Learning rate scheduler')
    parser.add_argument("--optimizer", type=str, default='adam', help='Optimizer to use')
    parser.add_argument('--lr', '--learning-rate', default='0.00005', type=eval_str_list,
                        metavar='LR_1,LR_2,...,LR_N',
                        help='learning rate for the first N epochs; all epochs >N using LR_N'
                             ' (note: this may be interpreted differently depending on --lr-scheduler)')
    parser.add_argument('--clip-norm', default=0., type=float, help='clip gradient')
    parser.add_argument('--empty-cache-freq', default=0, type=int,
                        help='how often to clear the PyTorch CUDA cache (0 to disable)')
    parser.add_argument('--save-checkpoint-every-niter', default=10000, type=int)
    parser.add_argument('--log-every-niter', default=100, type=int)

    # test details
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--mode', type=str, choices=[
        'generate-test', 'generate-test_all', 'evaluate-test', 'generate-dev', 'evaluate-dev', 'represent-test',
        'represent-dev', 'represent-train', 'generate-train', 'computeloss-train', None], default=None)
    parser.add_argument('--index_repr', type=str, choices=['whole', 'whole_avg_cell', 'span_context', 'span_noncontext'], default='whole', help='how to build representations for index')
    parser.add_argument('--num_beams', type=int, default=5, help='beam search size for the generate mode')
    parser.add_argument('--top_k', type=int, default=None, help='top k sampling')
    parser.add_argument('--top_p', type=float, default=None, help='top p sampling')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='number of generate sequences')
    parser.add_argument('--max_generate_length', type=int, default=None, help='max number of tokens generated for the generate mode')
    parser.add_argument('--min_generate_length', type=int, default=None, help='min number of tokens generated for the generate mode')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--return_log_prob', action='store_true')

    FairseqAdam.add_args(parser)
    PolynomialDecaySchedule.add_args(parser)

    # FP16 training
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--memory-efficient-fp16',
                        action='store_true',
                        help='Use memory efficient fp16')
    parser.add_argument('--threshold-loss-scale', type=float, default=None)
    parser.add_argument('--fp16-init-scale', type=float, default=128)
    # parser.add_argument('--fp16-scale-window', type=int, default=0)
    parser.add_argument('--fp16-scale-tolerance', type=float, default=0.0)
    parser.add_argument('--min-loss-scale', default=1e-4, type=float, metavar='D',
                        help='minimum FP16 loss scale, after which training is stopped')

    parser.add_argument('--debug-dataset', default=False, action='store_true')

    args = parser.parse_args()

    model_cls = task_dict[args.task]['model']
    if hasattr(model_cls, 'add_args'):
        model_cls.add_args(parser)
        args = parser.parse_args()

    return args


def main():
    args = parse_train_arg()
    task = task_dict[args.task]

    init_distributed_mode(args)
    logger = get_logger(args)

    if args.base_model_name:
        logger.info(f'use {args.base_model_name} to override the default model')
        args.table_bert_extra_config['base_model_name'] = args.base_model_name

    init_signal_handler()

    args.data_dir: List[Path] = [Path(dd) for dd in args.data_dir.split(':')]

    train_data_dirs = [dd / 'train' for dd in args.data_dir]
    dev_data_dirs = [dd / 'dev' for dd in args.data_dir]
    test_data_dirs = [dd / 'test' for dd in args.data_dir]

    train_data_exist = [dd for dd in train_data_dirs if dd.exists()]
    dev_data_exist = [dd for dd in dev_data_dirs if dd.exists()]
    test_data_exist = [dd for dd in test_data_dirs if dd.exists()]
    none_or_all_have = {0, len(train_data_dirs)}
    assert len(train_data_exist) in none_or_all_have and \
           len(dev_data_exist) in none_or_all_have and \
           len(test_data_exist) in none_or_all_have, \
        'train/dev/test sub-dir must exist or not exist for all data dirs'

    has_dev = len(dev_data_exist) > 0
    has_test = len(test_data_exist) > 0

    if args.multi_gpu and args.local_rank != 0:  # load tokenizer needs barrier
        torch.distributed.barrier()
    table_bert_config = task['config'].from_file(
        args.data_dir[0] / 'config.json', **args.table_bert_extra_config)
    if args.multi_gpu and args.local_rank == 0:
        torch.distributed.barrier()

    if args.is_master:
        args.output_dir.mkdir(exist_ok=True, parents=True)
        if not args.only_test:
            with (args.output_dir / 'train_config.json').open('w') as f:
                json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

            logger.info(f'Table Bert Config: {table_bert_config.to_log_string()}')

            table_bert_config.save(args.output_dir / 'tb_config.json')

    wandb_run = None
    if args.is_master and not args.only_test:
        wandb_run = wandb.init(entity=args.entity, project=args.project, name=args.name)

    assert args.data_dir[0].is_dir(), \
        "--data_dir should point to the folder of files made by pregenerate_training_data.py!"

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{torch.cuda.current_device()}')

    logger.info("device: {} gpu_id: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.local_rank, bool(args.multi_gpu), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.cpu:
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logger.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare model
    if args.multi_gpu and args.local_rank != 0:
        torch.distributed.barrier()

    if args.no_init:
        raise NotImplementedError
    else:
        model = task['model'](table_bert_config)

    if args.multi_gpu and args.local_rank == 0:
        torch.distributed.barrier()

    if args.fp16:
        model = model.half()

    model = model.to(device)
    if args.multi_gpu:
        if args.ddp_backend == 'pytorch':
            model = nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=True,
                device_ids=[args.local_rank], output_device=args.local_rank,
                broadcast_buffers=False
            )
        else:
            import apex
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

        model_ptr = model.module
    else:
        model_ptr = model

    if args.save_init:
        logger.info('save init checkpoint')
        torch.save(model_ptr.state_dict(), str(args.output_dir / 'model.bin'))
        exit()

    dataset_cls = task['dataset']

    if not args.only_test:
        train_set_info = dataset_cls.get_dataset_info_multi(train_data_dirs, args.max_epoch)
        # adjust batch size for really small datasets (or few-shot learning)
        args.train_batch_size = min(args.train_batch_size, train_set_info['one_epoch_size'] // args.world_size // args.gradient_accumulation_steps)
        assert args.train_batch_size >= 1, 'batch size is not positive'
        # set up update parameters for LR scheduler
        total_num_updates = train_set_info['total_size'] // args.train_batch_size // args.world_size // args.gradient_accumulation_steps
        args.max_epoch = train_set_info['max_epoch']
        logger.info(f'Train data size: {train_set_info["total_size"]} for {args.max_epoch} epochs, total num. updates: {total_num_updates}')
        args.total_num_update = total_num_updates
        args.warmup_updates = int(total_num_updates * 0.1)
    else:
        args.total_num_update = args.warmup_updates = 0

    # init trainer
    trainer = Trainer(model, args)

    # we also partitation the dev set for every local process
    logger.info('Loading dev/test (optional) set...')
    sys.stdout.flush()
    dev_set = dataset_cls(epoch=0, training_paths=dev_data_dirs, tokenizer=model_ptr.tokenizer, config=table_bert_config,
                          multi_gpu=args.multi_gpu, debug=args.debug_dataset) if has_dev else None
    test_set = dataset_cls(epoch=0, training_paths=test_data_dirs, tokenizer=model_ptr.tokenizer, config=table_bert_config,
                           multi_gpu=args.multi_gpu, debug=args.debug_dataset) if has_test else None

    if args.only_test:
        assert len(args.data_dir) == 1
        assert args.mode is not None, 'need to set mode for only_test'
        mode, which_part = args.mode.split('-')
        if which_part == 'train':  # load train dataset without shuffle
            logger.info(f'run test with multi gpu {args.multi_gpu}')
            train_set = dataset_cls(
                epoch=0, training_paths=args.data_dir[0] / 'train_noshuf', tokenizer=model_ptr.tokenizer,
                config=table_bert_config, multi_gpu=args.multi_gpu, debug=args.debug_dataset, not_even=True)
        elif which_part == 'test_all':  # test on multiple test set
            raw_output_file = trainer.args.output_file
            assert raw_output_file, 'output_file should be set'
            all_test_data_dirs = [d for d in args.data_dir[0].iterdir() if d.is_dir() and d.name.startswith('test_')]
            for _test_data_dir in all_test_data_dirs:
                trainer.args.output_file = raw_output_file + f'.{_test_data_dir.name}'
                _test_set = dataset_cls(epoch=0, training_paths=_test_data_dir, tokenizer=model_ptr.tokenizer,
                                        config=table_bert_config, multi_gpu=False, debug=args.debug_dataset)
                trainer.test(_test_set, mode=mode)
            exit()
        trainer.test(eval(f'{which_part}_set'), mode=mode)
        exit()

    # load checkpoint
    checkpoint_file = args.output_dir / 'model.ckpt.bin'
    is_resumed = False
    # trainer.save_checkpoint(checkpoint_file)
    if checkpoint_file.exists():
        logger.info(f'Logging checkpoint file {checkpoint_file}')
        is_resumed = True
        trainer.load_checkpoint(checkpoint_file)
    model.train()

    logger.info("***** Running training *****")
    logger.info(f"  Current config: {args}")

    if trainer.num_updates > 0:
        logger.info(f'Resume training at epoch {trainer.epoch}, '
                    f'epoch step {trainer.in_epoch_step}, '
                    f'global step {trainer.num_updates}')

    start_epoch = trainer.epoch
    for epoch in range(start_epoch, args.max_epoch):  # inclusive
        model.train()

        with torch.random.fork_rng(devices=None if args.cpu else [device.index]):
            torch.random.manual_seed(131 + epoch)

            epoch_dataset = dataset_cls(epoch=trainer.epoch, training_paths=train_data_dirs, config=table_bert_config,
                                        tokenizer=model_ptr.tokenizer, multi_gpu=args.multi_gpu, debug=args.debug_dataset)
            train_sampler = RandomSampler(epoch_dataset)
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0,
                                          collate_fn=partial(
                                              epoch_dataset.collate,
                                              pad_id=table_bert_config.pad_id,
                                              sep_id=table_bert_config.sep_id,
                                              max_allow_len=TableBertConfig.MAX_SOURCE_LEN))

        samples_iter = GroupedIterator(iter(train_dataloader), args.gradient_accumulation_steps)
        trainer.resume_batch_loader(samples_iter)

        with tqdm(total=len(samples_iter), initial=trainer.in_epoch_step,
                  desc=f"Epoch {epoch}", file=sys.stdout, disable=not args.is_master, miniters=100) as pbar:

            for samples in samples_iter:
                logging_output = trainer.train_step(samples)

                pbar.update(1)
                pbar.set_postfix_str(', '.join(f"{k}: {v:.4f}" for k, v in logging_output.items()))
                if (
                    wandb_run is not None and args.is_master and
                    0 < trainer.num_updates and
                    trainer.num_updates % args.log_every_niter == 0
                ):
                    logging_output['num_updates'] = trainer.num_updates
                    wandb_run.log(logging_output)

                if (
                    0 < trainer.num_updates and
                    trainer.num_updates % args.save_checkpoint_every_niter == 0 and
                    args.is_master
                ):
                    # Save model checkpoint
                    logger.info("** ** * Saving checkpoint file ** ** * ")
                    trainer.save_checkpoint(checkpoint_file)

            logger.info(f'Epoch {epoch} finished.')

            if args.is_master:
                # Save a trained table_bert
                logger.info("** ** * Saving fine-tuned table_bert ** ** * ")
                model_to_save = model_ptr  # Only save the table_bert it-self
                output_model_file = args.output_dir / f"pytorch_model_epoch{epoch:02d}.bin"
                torch.save(model_to_save.state_dict(), str(output_model_file))

            # perform validation
            logger.info("** ** * Perform validation ** ** * ")
            dev_results = {}
            if has_dev:
                dev_results = trainer.validate(dev_set)
                dev_results['epoch'] = epoch
                dev_results = {f'dev-{k}': v for k, v in dev_results.items()}
            test_results = {}
            if has_test:
                test_results = trainer.validate(test_set)
                test_results['epoch'] = epoch
                test_results = {f'test-{k}': v for k, v in test_results.items()}

            if args.is_master:
                logger.info('** ** * Validation Results ** ** * ')
                logger.info(f'Epoch {epoch} Validation Results: {dev_results}, Test Results: {test_results}')
                if wandb_run is not None:
                    wandb_run.log(dev_results)
                    if has_test:
                        wandb_run.log(test_results)

            # flush logging information to disk
            sys.stderr.flush()

        trainer.next_epoch()

    # run evaluation at the end of the training
    if args.mode is not None and args.is_master:  # evaluate after training using the whole dataset
        dev_set = dataset_cls(epoch=0, training_paths=dev_data_dirs, tokenizer=model_ptr.tokenizer, config=table_bert_config, multi_gpu=False) if has_dev else None
        test_set = dataset_cls(epoch=0, training_paths=test_data_dirs, tokenizer=model_ptr.tokenizer, config=table_bert_config, multi_gpu=False) if has_test else None
        mode, which_part = args.mode.split('-')
        trainer.args.output_file = f'ep{args.max_epoch - 1}.tsv'
        trainer.test(eval(f'{which_part}_set'), mode=mode)

    # syn to avoid bugs when we run multiple jobs in sequence
    if args.multi_gpu:
        torch.distributed.barrier()

    logger.info(f'End of {args.local_rank}')


if __name__ == '__main__':
    main()
