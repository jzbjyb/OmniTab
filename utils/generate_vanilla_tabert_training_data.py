#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os, sys
import subprocess
import time
import traceback
from argparse import ArgumentParser, Namespace
import logging
from multiprocessing import connection
from typing import List, Set, Iterator, Callable

import h5py
import numpy as np
import json
import ujson
import msgpack
import signal

import gc
import torch
import zmq

from pathlib import Path

from omnitab.config import TableBertConfig
from omnitab.input_formatter import VanillaTableBertInputFormatter
from tqdm import tqdm, trange

from random import shuffle, choice, sample, random

from omnitab.input_formatter import VanillaTableBertInputFormatter, TableBertBertInputFormatter, trim_count, tablededup2count
from omnitab.config import TableBertConfig
from omnitab.dataset import Example, TableDatabase
#from utils.prepare_training_data import sample_context
from utils.generate_vertical_tabert_training_data import sample_context


def generate_for_epoch(table_db: TableDatabase,
                       indices: List[int],
                       epoch_file: Path,
                       input_formatter: TableBertBertInputFormatter,
                       args: Namespace):
    debug_file = epoch_file.with_suffix('.sample.json') if args.is_master else None
    if debug_file:
        f_dbg = open(debug_file, 'w')

    sequences = []
    column_token_to_column_id = []
    context_token_to_mention_id = []
    mentions_cells = []
    segment_a_lengths = []
    sequence_offsets = []
    mentions_cells_offsets = []
    masked_lm_positions = []
    masked_lm_label_ids = []
    masked_lm_offsets = []
    is_positives = []
    target_sequences = []
    target_sequence_offsets = []

    def _save_shard():
        data = {
            'sequences': np.uint16(sequences),
            'segment_a_lengths': np.uint16(segment_a_lengths),
            'sequence_offsets': np.uint64(sequence_offsets),
            'masked_lm_positions': np.uint16(masked_lm_positions),
            'masked_lm_label_ids': np.uint16(masked_lm_label_ids),
            'masked_lm_offsets': np.uint64(masked_lm_offsets),
            'is_positives': np.uint16(is_positives),
        }
        if len(target_sequences) > 0:  # has target
            data['target_sequences'] = np.uint16(target_sequences)
            data['target_sequence_offsets'] = np.uint64(target_sequence_offsets)
        if len(column_token_to_column_id) > 0:
            data['column_token_to_column_id'] = np.int16(column_token_to_column_id)
        if len(context_token_to_mention_id) > 0:
            data['context_token_to_mention_id'] = np.int16(context_token_to_mention_id)
        if len(mentions_cells) > 0:
            data['mentions_cells'] = np.int16(mentions_cells)
            data['mentions_cells_offsets'] = np.uint64(mentions_cells_offsets)

        with h5py.File(str(epoch_file), 'w') as f:
            for key, val in data.items():
                f.create_dataset(key, data=val)

        del sequences[:]
        del column_token_to_column_id[:]
        del context_token_to_mention_id[:]
        del mentions_cells[:]
        del segment_a_lengths[:]
        del sequence_offsets[:]
        del mentions_cells_offsets[:]
        del masked_lm_positions[:]
        del masked_lm_label_ids[:]
        del masked_lm_offsets[:]
        del is_positives[:]
        del target_sequences[:]
        del target_sequence_offsets[:]

    for example_idx in tqdm(indices, desc=f"Generating dataset {epoch_file}", file=sys.stdout):
        example = table_db[example_idx]
        try:
            instances = input_formatter.get_pretraining_instances_from_example(example, sample_context)

            for instance in instances:
                if debug_file and random() <= 0.05:
                    f_dbg.write(json.dumps(instance) + os.linesep)

                input_formatter.remove_unecessary_instance_entries(instance)

                cur_pos = len(sequences)
                sequence_len = len(instance['token_ids'])
                sequences.extend(instance['token_ids'])
                segment_a_lengths.append(instance['segment_a_length'])
                sequence_offsets.append([cur_pos, cur_pos + sequence_len])

                if 'target_token_ids' in instance:
                    cur_pos_tgt = len(target_sequences)
                    target_sequence_offsets.append([cur_pos_tgt, cur_pos_tgt + len(instance['target_token_ids'])])
                    target_sequences.extend(instance['target_token_ids'])
                if 'column_token_to_column_id' in instance:
                    column_token_to_column_id.extend(instance['column_token_to_column_id'])
                if 'context_token_to_mention_id' in instance:
                    context_token_to_mention_id.extend(instance['context_token_to_mention_id'])
                if 'mentions_cells' in instance:
                    cur_pos_mc = len(mentions_cells)
                    mentions_cells.extend(instance['mentions_cells'])
                    mentions_cells_offsets.append([cur_pos_mc, cur_pos_mc + len(instance['mentions_cells'])])

                cur_pos_mlm = len(masked_lm_positions)
                masked_lm_positions.extend(instance['masked_lm_positions'])
                masked_lm_label_ids.extend(instance['masked_lm_label_ids'])
                masked_lm_offsets.append([cur_pos_mlm, cur_pos_mlm + len(instance['masked_lm_positions'])])
                is_positives.append(int(example.is_positive))
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            # raise
            typ, value, tb = sys.exc_info()
            print('*' * 50 + 'Exception' + '*' * 50, file=sys.stderr)
            print(example.serialize(), file=sys.stderr)
            print('*' * 50 + 'Stack Trace' + '*' * 50, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # print('*' * 50 + 'Exception' + '*' * 50, file=sys.stderr)

            sys.stderr.flush()

    print('\ntotal count {}\n'.format(len(sequence_offsets)))
    _save_shard()


def convert_already_preprocessed_data(datapath: Path,
                                      output_dir: Path,
                                      data_iterator: Callable,
                                      tokenizer, indices: List[int],
                                      args: Namespace,
                                      config: TableBertConfig):
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / 'train').mkdir(exist_ok=True)

    indices: Set[int] = set(indices)

    for epoch in trange(args.epochs_to_generate, desc='Epoch'):
        gc.collect()

        epoch_filename = output_dir / 'train' / f'epoch_{epoch}.shard{args.global_rank}.h5'

        sequences = []
        segment_a_lengths = []
        sequence_offsets = []

        target_sequences = []
        target_sequence_offsets = []

        masked_lm_positions = []
        masked_lm_label_ids = []
        masked_lm_offsets = []

        is_positives = []

        for context, table, target in data_iterator(datapath, indices):
            # get source tokens
            context_tokens: List[str] = tokenizer.tokenize(context)
            table_tokens: List[str] = tokenizer.tokenize(table)

            context_tokens = context_tokens[:config.MAX_SOURCE_LEN - 2]  # cls and sep
            seg_a_len = len(context_tokens) + 1  # cls
            source_tokens = [config.cls_token] + \
                            (context_tokens + table_tokens)[:config.MAX_SOURCE_LEN - 2] + \
                            [config.sep_token]
            source_token_ids: List[int] = tokenizer.convert_tokens_to_ids(source_tokens)

            # get target tokens
            target_tokens: List[str] = [config.cls_token] + \
                                       tokenizer.tokenize(target)[:config.MAX_TARGET_LEN - 2] + \
                                       [config.sep_token]
            target_token_ids: List[int] = tokenizer.convert_tokens_to_ids(target_tokens)

            # save
            cur_pos = len(sequences)
            sequence_offsets.append([cur_pos, cur_pos + len(source_token_ids)])
            sequences.extend(source_token_ids)
            segment_a_lengths.append(seg_a_len)

            cur_pos_tgt = len(target_sequences)
            target_sequence_offsets.append([cur_pos_tgt, cur_pos_tgt + len(target_token_ids)])
            target_sequences.extend(target_token_ids)

            masked_lm_offsets.append([0, 0])
            is_positives.append(1)

        data = {
            'sequences': np.uint16(sequences),
            'segment_a_lengths': np.uint16(segment_a_lengths),
            'sequence_offsets': np.uint64(sequence_offsets),
            'masked_lm_positions': np.uint16(masked_lm_positions),
            'masked_lm_label_ids': np.uint16(masked_lm_label_ids),
            'masked_lm_offsets': np.uint64(masked_lm_offsets),
            'is_positives': np.uint16(is_positives),
            'target_sequences': np.uint16(target_sequences),
            'target_sequence_offsets': np.uint64(target_sequence_offsets)
        }

        with h5py.File(str(epoch_filename), 'w') as f:
            for key, val in data.items():
                f.create_dataset(key, data=val)

        del sequences[:]
        del segment_a_lengths[:]
        del sequence_offsets[:]
        del masked_lm_positions[:]
        del masked_lm_label_ids[:]
        del masked_lm_offsets[:]
        del is_positives[:]
        del target_sequences[:]
        del target_sequence_offsets[:]


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of preprocess to pregenerate")
    parser.add_argument('--no_wiki_tables_from_common_crawl', action='store_true', default=False)
    parser.add_argument('--global_rank', type=int, default=os.environ.get('SLURM_PROCID', 0))
    parser.add_argument('--world_size', type=int, default=os.environ.get('SLURM_NTASKS', 1))
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--dev_num', type=int, default=None, help='number of examples used for validation')
    parser.add_argument('--already_preprocessed', type=str, default=None, choices=['tapex', 'totto', None],
                        help='type of the already preprocessed data')

    TableBertConfig.add_args(parser)

    args = parser.parse_args()
    args.is_master = args.global_rank == 0

    logger = logging.getLogger('DataGenerator')
    handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info(f'Rank {args.global_rank} out of {args.world_size}')
    sys.stderr.flush()

    table_bert_config = TableBertConfig.from_dict(vars(args))
    tokenizer = table_bert_config.tokenizer_cls.from_pretrained(table_bert_config.base_model_name)
    tokenizer_fast = table_bert_config.tokenizer_fast_cls.from_pretrained(table_bert_config.base_model_name)
    input_formatter = VanillaTableBertInputFormatter(table_bert_config, tokenizer)

    total_tables_num = int(subprocess.check_output(f"wc -l {args.train_corpus}", shell=True).split()[0])
    dev_table_num = min(int(total_tables_num * 0.1), 100000) if args.dev_num is None else args.dev_num
    train_table_num = total_tables_num - dev_table_num

    # seed the RNG to make sure each process follows the same spliting
    rng = np.random.RandomState(seed=5783287)

    corpus_table_indices = list(range(total_tables_num))
    if not args.no_shuffle:
        rng.shuffle(corpus_table_indices)
    dev_table_indices = corpus_table_indices[:dev_table_num]
    train_table_indices = corpus_table_indices[dev_table_num:]

    if args.no_shuffle:
        dev_single_size = int(np.ceil(len(dev_table_indices) / args.world_size))
        train_single_size = int(np.ceil(len(train_table_indices) / args.world_size))
        local_dev_table_indices = dev_table_indices[args.global_rank * dev_single_size:(args.global_rank + 1) * dev_single_size]
        local_train_table_indices = train_table_indices[args.global_rank * train_single_size:(args.global_rank + 1) * train_single_size]
    else:
        local_dev_table_indices = dev_table_indices[args.global_rank::args.world_size]
        local_train_table_indices = train_table_indices[args.global_rank::args.world_size]
    local_indices = local_dev_table_indices + local_train_table_indices

    logger.info(f'total tables: {total_tables_num}')
    logger.debug(f'local dev table indices: {local_dev_table_indices[:1000]}')
    logger.debug(f'local train table indices: {local_train_table_indices[:1000]}')

    def tapex_data_iterator(datapath: Path, indices: List[int]):
        with open(str(datapath), 'r') as sfin, open(str(datapath).replace('.src', '.tgt'), 'r') as tfin:
            for line_id, source in tqdm(enumerate(sfin)):
                source = source.strip()
                target = tfin.readline().strip()
                if line_id not in indices:
                    continue
                cp = source.find('col :')
                context = source[:cp].strip()
                table = source[cp:].strip()
                yield context, table, target

    def totto_data_iterator(datapath: Path, indices: List[int], field: str = 'subtable'):
        assert field in {'full_table', 'subtable'}
        with open(str(datapath), 'r') as fin:
            for line_id, l in tqdm(enumerate(fin)):
                l = json.loads(l)
                if line_id not in indices:
                    continue
                source = l[f'{field}_metadata_str']
                target = l['sentence_annotations'][0]['final_sentence'].strip()
                cp = source.find('<table>')
                context = source[:cp].strip()
                table = source[cp:].strip()
                yield context, table, target

    if args.already_preprocessed:
        di = eval(f'{args.already_preprocessed}_data_iterator')
        convert_already_preprocessed_data(
            args.train_corpus, args.output_dir, data_iterator=di,
            tokenizer=tokenizer, indices=local_indices, args=args, config=table_bert_config)
        exit()

    # use noshuf as the name for train to distinguish it from randomly sampled train
    train_subdir = 'train_noshuf' if args.no_shuffle else 'train'
    with TableDatabase.from_jsonl(args.train_corpus, backend='memory', tokenizer=tokenizer, tokenizer_fast=tokenizer_fast,
                                  indices=local_indices,
                                  skip_column_name_longer_than=table_bert_config.skip_column_name_longer_than,
                                  not_skip_empty_column_name=table_bert_config.not_skip_empty_column_name,
                                  only_keep_highlighted_rows=table_bert_config.only_keep_highlighted_rows,
                                  highlight_table=table_bert_config.highlight_table) as table_db:
        local_indices = {idx for idx in local_indices if idx in table_db}
        local_dev_table_indices = [idx for idx in local_dev_table_indices if idx in local_indices]
        local_train_table_indices = [idx for idx in local_train_table_indices if idx in local_indices]

        args.output_dir.mkdir(exist_ok=True, parents=True)
        print(f'Num tables to be processed by local worker: {len(table_db)}', file=sys.stdout)

        if args.is_master:
            with (args.output_dir / 'config.json').open('w') as f:
                json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

        (args.output_dir / train_subdir).mkdir(exist_ok=True)
        (args.output_dir / 'dev').mkdir(exist_ok=True)

        # generate dev data first
        dev_file = args.output_dir / 'dev' / f'epoch_0.shard{args.global_rank}.h5'
        generate_for_epoch(table_db, local_dev_table_indices, dev_file, input_formatter, args)

        for epoch in trange(args.epochs_to_generate, desc='Epoch'):
            gc.collect()
            epoch_filename = args.output_dir / train_subdir / f"epoch_{epoch}.shard{args.global_rank}.h5"
            generate_for_epoch(table_db, local_train_table_indices, epoch_filename, input_formatter, args)

    print('trimed table statistics', trim_count)
    print(f'table dup cells {sorted(tablededup2count.items())}')


if __name__ == '__main__':
    main()
