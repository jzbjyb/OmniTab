#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import re

import ujson
import msgpack
import logging
import math
import multiprocessing
import sys
import copy
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Iterator, Set, Union, List, Tuple
import redis
import random

import numpy as np
import torch
import zmq
from omnitab.utils import BertTokenizer, BertTokenizerFast
from omnitab.config import TableBertConfig
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
from tqdm import tqdm
from omnitab.table import Column, Table
import h5py


class DistributedSampler(Sampler):
    """Sampler that restricts preprocess loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class TableDataset(Dataset):
    DEFAULT_CONFIG_CLS = TableBertConfig

    def __init__(self,
                 training_paths: Union[Path, List[Path]],
                 epoch=0,
                 config=None,
                 tokenizer=None,
                 reduce_memory=False,
                 multi_gpu=False,
                 indices=None,
                 debug=False,
                 not_even=False):
        # self.vocab = tokenizer.vocab
        # self.tokenizer = tokenizer
        self.config = config or self.DEFAULT_CONFIG_CLS()
        if type(training_paths) is not list:
            training_paths = [training_paths]
        self.epoch = epoch

        self.examples = []
        for training_path in training_paths:
            num_data_epochs = self.get_data_num_epochs(training_path)

            data_epoch = epoch
            if data_epoch >= num_data_epochs:
                data_epoch = data_epoch % num_data_epochs  # reuse dataset for more epochs
            data_file_prefix = training_path / f'epoch_{data_epoch}'

            epoch_info = self.get_epoch_shards_info(data_file_prefix)
            epoch_dataset_size = epoch_info['total_size']

            assert reduce_memory is False, 'reduce_memory is not implemented'

            if not indices:
                if multi_gpu:
                    num_shards = torch.distributed.get_world_size()
                    local_shard_id = torch.distributed.get_rank()

                    shard_size = None
                    if not not_even:
                        shard_size = epoch_dataset_size // num_shards

                    logging.info(f'dataset_size={epoch_dataset_size}, shard_size={shard_size}')

                    g = torch.Generator()
                    g.manual_seed(self.epoch)
                    _indices = torch.randperm(epoch_dataset_size, generator=g).tolist()

                    if not not_even:  # make it evenly divisible
                        _indices = _indices[:shard_size * num_shards]
                        assert len(_indices) == shard_size * num_shards

                    # subsample
                    _indices = _indices[local_shard_id:len(_indices):num_shards]
                    if not not_even:
                        assert len(_indices) == shard_size

                    _indices = set(_indices)
                else:
                    _indices = set(range(epoch_dataset_size))
            else:
                _indices = set(indices)

            if debug:
                _indices = set(list(_indices)[:1000])

            logging.info(f"Loading examples from {training_path} for epoch {epoch}")
            if _indices:
                logging.info(f'Load a sub-sample of the whole dataset')

            self.examples.extend(self.load_epoch(data_file_prefix, epoch_info['shard_num'], _indices))

    @classmethod
    def get_dataset_info_multi(cls, data_paths: List[Path], max_epoch=-1):
        assert len(data_paths) > 0, 'no path'
        infos: List[Dict] = []
        for data_path in data_paths:
            infos.append(cls.get_dataset_info(data_path, max_epoch=max_epoch))
        info = infos[0]
        for _info in infos[1:]:
            info['total_size'] += _info['total_size']
        info['one_epoch_size'] = info['total_size'] // info['max_epoch']
        return info

    @classmethod
    def get_dataset_info(cls, data_path: Path, max_epoch=-1):
        if max_epoch == -1:
            epoch_files = list(data_path.glob('epoch_*'))
            epoch_ids = [
                int(re.search(r'epoch_(\d+)', str(f)).group(1))
                for f in epoch_files
            ]
            max_epoch = max(epoch_ids) + 1

        data_size = 0
        num_data_epochs = cls.get_data_num_epochs(data_path)
        for epoch_id in range(max_epoch):
            eid = epoch_id % num_data_epochs
            epoch_file = data_path / f'epoch_{eid}'
            epoch_info = cls.get_epoch_shards_info(epoch_file)
            data_size += epoch_info['total_size']

        return {
            'total_size': data_size,
            'max_epoch': max_epoch,
            'one_epoch_size': data_size // max_epoch,
        }

    @classmethod
    def get_shard_size(cls, shard_file: Path):
        with h5py.File(str(shard_file), 'r', rdcc_nbytes=1024 * 1024 * 2048) as data:
            shard_size = data['masked_lm_offsets'].shape[0]

        return shard_size

    @classmethod
    def get_data_num_epochs(cls, root_dir: Path):
        prefix = root_dir / 'epoch_'
        shard_files = list(prefix.parent.glob(prefix.name + '*.shard*.h5'))
        epoch_ids = [int(re.search(r'epoch_(\d+)', str(f)).group(1)) for f in shard_files]
        num_epochs = max(epoch_ids) + 1
        return num_epochs

    @classmethod
    def get_epoch_shards_info(cls, shard_file_prefix: Path):
        shard_files = list(shard_file_prefix.parent.glob(shard_file_prefix.name + '.shard*.h5'))
        shard_ids = [int(re.search(r'shard(\d+)', str(f)).group(1)) for f in shard_files]
        shard_num = max(shard_ids) + 1

        cum_size = 0
        for shard_file in shard_files:
            print(shard_file)
            shard_size = cls.get_shard_size(shard_file)
            cum_size += shard_size

        return {
            'shard_num': shard_num,
            'total_size': cum_size
        }

    def only_table(self, example: Dict):
        if not self.config.context_first:
            raise NotImplementedError
        al = example['sequence_a_length']
        input_ids = np.copy(example['token_ids'])
        # split
        context, table = copy.deepcopy(input_ids[:al]), copy.deepcopy(input_ids[al - 1:])
        table[0] = context[0]
        # adjust masked
        only_keep = [i >= al - 1 for i in example['masked_lm_positions']]
        masked_lm_positions = [example['masked_lm_positions'][i] - al + 1 for i in range(len(only_keep)) if only_keep[i]]
        masked_lm_label_ids = [example['masked_lm_label_ids'][i] for i in range(len(only_keep)) if only_keep[i]]
        example['sequence_a_length'] = len(table)
        example['token_ids'] = table
        example['masked_lm_positions'] = masked_lm_positions
        example['masked_lm_label_ids'] = masked_lm_label_ids

    def add_for_contrastive(self, example: Dict, concat: bool = False, same_first_token: bool = False):
        if not self.config.context_first:
            raise NotImplementedError
        al = example['sequence_a_length']
        raw_input_ids = np.copy(example['token_ids'])
        masked_lm_positions = example['masked_lm_positions']
        masked_lm_label_ids = example['masked_lm_label_ids']
        raw_input_ids[masked_lm_positions] = masked_lm_label_ids
        if not concat:
            # assume there is a sep in between and make sure that the sep is included in both
            # use deepcopy because they have overlappings
            context, table = copy.deepcopy(raw_input_ids[:al]), copy.deepcopy(raw_input_ids[al - 1:])
            if 'column_token_to_column_id' in example:
                table_ct2ci = example['column_token_to_column_id'][al - 1:]
        else:
            # assume there is a sep in between and make sure that the sep is included in context
            context, table = raw_input_ids[:al], raw_input_ids[al:]
            if 'column_token_to_column_id' in example:
                table_ct2ci = example['column_token_to_column_id'][al:]
            example['contrastive_concat'] = True
        if same_first_token:
            table[0] = context[0]
        example['context_token_ids'] = context
        example['table_token_ids'] = table
        if 'column_token_to_column_id' in example:
            example['table_column_token_to_column_id'] = table_ct2ci
        if 'context_token_to_mention_id' in example:
            example['context_context_token_to_mention_id'] = example['context_token_to_mention_id'][:al]
        assert len(example['context_token_ids']) == al, 'context length inconsistent with sequence_a_length'

    def load_epoch(self, file_prefix: Path, shard_num: int, valid_indices: Set = None):
        examples = []
        idx = -1
        for shard_id in range(shard_num):
            file_name = file_prefix.with_suffix(f'.shard{shard_id}.h5')
            if file_name.exists():
                data = h5py.File(str(file_name), 'r', rdcc_nbytes=1024 * 1024 * 2048)
            else:
                file_name = file_name.with_suffix('.bin')
                data = torch.load(str(file_name))

            sequences = data['sequences']
            segment_a_lengths = data['segment_a_lengths']
            sequence_offsets = data['sequence_offsets']
            masked_lm_positions = data['masked_lm_positions']
            masked_lm_label_ids = data['masked_lm_label_ids']
            masked_lm_offsets = data['masked_lm_offsets']
            is_positives = data['is_positives'] if 'is_positives' in data else None
            target_sequences = data['target_sequences'] if 'target_sequences' in data else None
            target_sequence_offsets = data['target_sequence_offsets'] if 'target_sequence_offsets' in data else None
            column_token_to_column_id = data['column_token_to_column_id'] if 'column_token_to_column_id' in data else None
            context_token_to_mention_id = data['context_token_to_mention_id'] if 'context_token_to_mention_id' in data else None
            mentions_cells = data['mentions_cells'] if 'mentions_cells' in data else None
            mentions_cells_offsets = data['mentions_cells_offsets'] if 'mentions_cells_offsets' in data else None

            shard_size = len(segment_a_lengths)

            for i in range(shard_size):
                idx += 1

                if valid_indices and idx not in valid_indices:
                    continue

                example = {'idx': idx}

                seq_begin, seq_end = sequence_offsets[i]
                example['token_ids'] = sequences[seq_begin: seq_end]
                if column_token_to_column_id is not None:
                    example['column_token_to_column_id'] = column_token_to_column_id[seq_begin:seq_end]
                if context_token_to_mention_id is not None:
                    example['context_token_to_mention_id'] = context_token_to_mention_id[seq_begin:seq_end]
                if mentions_cells is not None:
                    ms_begin, ms_end = mentions_cells_offsets[i]
                    example['mentions_cells'] = mentions_cells[ms_begin:ms_end]
                    example['num_cells'] = max(np.max(example['column_token_to_column_id']) + 1, 0)
                    example['num_mentions'] = max(np.max(example['context_token_to_mention_id']) + 1, 0)
                if target_sequence_offsets is not None:
                    tgt_begin, tgt_end = target_sequence_offsets[i]
                    example['target_token_ids'] = target_sequences[tgt_begin:tgt_end]

                seq_a_length = segment_a_lengths[i]
                example['sequence_a_length'] = seq_a_length

                mlm_begin, mlm_end = masked_lm_offsets[i]
                example['masked_lm_positions'] = masked_lm_positions[mlm_begin:mlm_end]
                example['masked_lm_label_ids'] = masked_lm_label_ids[mlm_begin:mlm_end]

                example['is_positive'] = is_positives[i] if is_positives is not None else 1  # default to 1

                if self.config.only_table:
                    self.only_table(example)

                obj = self.config.objective_function
                if 'contrastive' in obj or \
                  'table2text' in obj or \
                  'text2table' in obj or \
                  'separate-bin' in obj or \
                  'contrast-span' in obj or \
                  'split' in obj:  # only for split table/context purpose
                    # use cls for the first position for seq2seq objective
                    self.add_for_contrastive(example, same_first_token=True)
                elif 'contrast-concat' in obj or 'nsp' in obj:
                    self.add_for_contrastive(example, concat=True)

                examples.append(example)

            if isinstance(data, h5py.File):
                data.close()

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def collate(examples, pad_id: int = 0, sep_id: int = 102, max_allow_len: int = 512, mention_cell_neg_count: int = 5):  # TODO: add model specific param
        batch_size = len(examples)
        max_len = max(len(e['token_ids']) for e in examples)
        has_contrastive = True
        has_contrastive_concat = True
        has_is_positive = True
        has_target = True
        has_ct2ci = True
        has_ct2mi = True
        has_mention_cell = True
        for e in examples:  # use the first one to check
            if 'context_token_ids' not in e or 'contrastive_concat' in e:
                has_contrastive = False
            if 'context_token_ids' not in e or 'contrastive_concat' not in e:
                has_contrastive_concat = False
            if 'is_positive' not in e:
                has_is_positive = False
            if 'target_token_ids' not in e:
                has_target = False
            if 'column_token_to_column_id' not in e:
                has_ct2ci = False
            if 'context_token_to_mention_id' not in e:
                has_ct2mi = False
            if 'mentions_cells' not in e:
                has_mention_cell = False

        if has_target:
            max_target_len = max(len(e['target_token_ids']) for e in examples)
        if has_contrastive or has_contrastive_concat:
            max_context_len = max(len(e['context_token_ids']) for e in examples)
            max_table_len = max(len(e['table_token_ids']) for e in examples)

        idx_array = np.full((batch_size,), dtype=np.int, fill_value=0)
        input_array = np.full((batch_size, max_len), dtype=np.int, fill_value=pad_id)
        mask_array = np.zeros((batch_size, max_len), dtype=np.bool)
        segment_array = np.zeros((batch_size, max_len), dtype=np.bool)
        if has_ct2ci:
            column_token_to_column_id = np.full((batch_size, max_len), dtype=np.int, fill_value=-1)
        if has_ct2mi:
            context_token_to_mention_id = np.full((batch_size, max_len), dtype=np.int, fill_value=-1)
        if has_mention_cell:
            max_num_cells = np.max([example['num_cells'] for example in examples])
            max_num_mentions = np.max([example['num_mentions'] for example in examples])
            pos_mentions_cells = []
            neg_mentions_cells = []
        if has_target:
            lm_label_array = np.full((batch_size, max_target_len), dtype=np.int, fill_value=-1)
            target_input_array = np.full((batch_size, max_target_len), dtype=np.int, fill_value=pad_id)
            is_mlm_array = np.full((batch_size,), dtype=np.bool, fill_value=True)
        else:
            lm_label_array = np.full((batch_size, max_len), dtype=np.int, fill_value=-1)
        if has_contrastive:
            context_input_array = np.full((batch_size, max_context_len), dtype=np.int, fill_value=pad_id)
            table_input_array = np.full((batch_size, max_table_len), dtype=np.int, fill_value=pad_id)
            context_mask_array = np.zeros((batch_size, max_context_len), dtype=np.bool)
            table_mask_array = np.zeros((batch_size, max_table_len), dtype=np.bool)
            context_segment_array = np.zeros((batch_size, max_context_len), dtype=np.bool)
            table_segment_array = np.zeros((batch_size, max_table_len), dtype=np.bool)
            if has_ct2ci:
                table_column_token_to_column_id = np.full((batch_size, max_table_len), dtype=np.int, fill_value=-1)
            if has_ct2mi:
                context_context_token_to_mention_id = np.full((batch_size, max_context_len), dtype=np.int, fill_value=-1)
        elif has_contrastive_concat:
            d_bs = batch_size * batch_size
            max_len = min(max_context_len + max_table_len, max_allow_len)  # might exceed max allowed length
            concat_input_array = np.full((d_bs, max_len), dtype=np.int, fill_value=pad_id)
            concat_mask_array = np.zeros((d_bs, max_len), dtype=np.bool)
            concat_segment_array = np.zeros((d_bs, max_len), dtype=np.bool)
            context_mask_array = np.zeros((d_bs, max_len), dtype=np.bool)
            table_mask_array = np.zeros((d_bs, max_len), dtype=np.bool)
            concat_labels = np.diag(np.ones(batch_size)).reshape(-1)

        for e_id, example in enumerate(examples):
            idx_array[e_id] = example['idx']

            token_ids = example['token_ids']
            # print(tokenizer.convert_ids_to_tokens(token_ids))
            # assert tokenizer.convert_ids_to_tokens([token_ids[0]]) == ['[CLS]'] and \
            #        tokenizer.convert_ids_to_tokens([token_ids[-1]]) == ['[SEP]']

            masked_label_ids = example['masked_lm_label_ids']
            masked_lm_positions = example['masked_lm_positions']

            input_array[e_id, :len(token_ids)] = token_ids
            mask_array[e_id, :len(token_ids)] = 1
            segment_array[e_id, example['sequence_a_length']:] = 1
            if not has_is_positive or example['is_positive']:  # only generate mlm labels for positive examples
                if not has_target or len(masked_lm_positions) > 0:
                    lm_label_array[e_id, masked_lm_positions] = masked_label_ids  # masked tokens as labels
                else:
                    lm_label_array[e_id, :len(example['target_token_ids'])] = example['target_token_ids']  # target as labels
                    is_mlm_array[e_id] = False

            if has_ct2ci:
                ct2ci = example['column_token_to_column_id']
                column_token_to_column_id[e_id, :len(ct2ci)] = ct2ci
            if has_ct2mi:
                ct2mi = example['context_token_to_mention_id']
                context_token_to_mention_id[e_id, :len(ct2mi)] = ct2mi
            if has_mention_cell:
                for mid, cid in example['mentions_cells']:
                    _mid = e_id * max_num_mentions + mid
                    _cid = e_id * max_num_cells + cid
                    pos_mentions_cells.append((_mid, _cid))
                    # neg cells from the same table
                    _mention_cell_neg_count = min(mention_cell_neg_count, example['num_cells'])
                    for neg_cid in np.random.choice(example['num_cells'], _mention_cell_neg_count, replace=False):
                        if neg_cid == cid:
                            continue
                        _cid = e_id * max_num_cells + neg_cid
                        neg_mentions_cells.append((_mid, _cid))
                    # neg cells from other tables in the same batch
                    for _ in range(mention_cell_neg_count):
                        _eids = list(set(np.random.choice(len(examples), min(2, len(examples)), replace=False)) - {e_id})  # sample a table
                        if len(_eids) <= 0:
                            continue
                        _eid = _eids[0]
                        _example = examples[_eid]
                        if _example['num_cells'] <= 0:
                            continue
                        neg_cid = np.random.choice(_example['num_cells'], 1)[0]  # sample a cell
                        _cid = _eid * max_num_cells + neg_cid
                        neg_mentions_cells.append((_mid, _cid))
            if has_target:
                target_input_array[e_id, :len(example['target_token_ids'])] = example['target_token_ids']
            if has_contrastive:
                context_token_ids = example['context_token_ids']
                table_token_ids = example['table_token_ids']
                context_input_array[e_id, :len(context_token_ids)] = context_token_ids
                table_input_array[e_id, :len(table_token_ids)] = table_token_ids
                context_mask_array[e_id, :len(context_token_ids)] = 1
                table_mask_array[e_id, :len(table_token_ids)] = 1
                if has_ct2ci:
                    tct2ci = example['table_column_token_to_column_id']
                    table_column_token_to_column_id[e_id, :len(tct2ci)] = tct2ci
                if has_ct2mi:
                    cct2mi = example['context_context_token_to_mention_id']
                    context_context_token_to_mention_id[e_id, :len(cct2mi)] = cct2mi
            elif has_contrastive_concat:
                context_token_ids = example['context_token_ids']
                for e_id2, example2 in enumerate(examples):
                    table_token_ids = example2['table_token_ids']
                    ind = e_id * batch_size + e_id2
                    concat_input_array[ind, :len(context_token_ids)] = context_token_ids
                    if len(context_token_ids) + len(table_token_ids) > max_allow_len:
                        concat_input_array[ind, len(context_token_ids):] = table_token_ids[:max_allow_len - len(context_token_ids)]
                        concat_input_array[ind, -1] = sep_id
                    else:
                        concat_input_array[ind, len(context_token_ids):len(context_token_ids) + len(table_token_ids)] = table_token_ids
                    concat_mask_array[ind, :len(context_token_ids) + len(table_token_ids)] = 1
                    concat_segment_array[ind, len(context_token_ids):] = 1
                    context_mask_array[ind, 0] = 1
                    table_mask_array[ind, len(context_token_ids) - 1] = 1  # sep is in context

        result = {
            'idx': torch.tensor(idx_array),
            'input_ids': torch.tensor(input_array.astype(np.int64)),
            'attention_mask': torch.tensor(mask_array.astype(np.int64)),
            'token_type_ids': torch.tensor(segment_array.astype(np.int64)),
            'masked_lm_labels': torch.tensor(lm_label_array.astype(np.int64)),
            'sample_size': (lm_label_array != -1).sum()
        }
        if has_target:
            result['target_input_ids'] = torch.tensor(target_input_array.astype(np.int64))
            result['is_mlm'] = torch.tensor(is_mlm_array)
        if has_ct2ci:
            result['column_token_to_column_id'] = torch.tensor(column_token_to_column_id.astype(np.int64))
        if has_ct2mi:
            result['context_token_to_mention_id'] = torch.tensor(context_token_to_mention_id.astype(np.int64))
        if has_mention_cell:
            result['pos_mentions_cells'] = torch.tensor(pos_mentions_cells)
            result['neg_mentions_cells'] = torch.tensor(neg_mentions_cells)

        if has_contrastive:
            result['context_input_ids'] = torch.tensor(context_input_array.astype(np.int64))
            result['table_input_ids'] = torch.tensor(table_input_array.astype(np.int64))
            result['context_attention_mask'] = torch.tensor(context_mask_array.astype(np.int64))
            result['table_attention_mask'] = torch.tensor(table_mask_array.astype(np.int64))
            result['context_token_type_ids'] = torch.tensor(context_segment_array.astype(np.int64))
            result['table_token_type_ids'] = torch.tensor(table_segment_array.astype(np.int64))
            if has_ct2ci:
                result['table_column_token_to_column_id'] = torch.tensor(table_column_token_to_column_id.astype(np.int64))
            if has_ct2mi:
                result['context_context_token_to_mention_id'] = torch.tensor(context_context_token_to_mention_id.astype(np.int64))
        elif has_contrastive_concat:
            result['concat_input_ids'] = torch.tensor(concat_input_array.astype(np.int64))
            result['concat_attention_mask'] = torch.tensor(concat_mask_array.astype(np.int64))
            result['concat_token_type_ids'] = torch.tensor(concat_segment_array.astype(np.int64))
            result['context_mask'] = torch.tensor(context_mask_array)
            result['table_mask'] = torch.tensor(table_mask_array)
            result['concat_labels'] = torch.tensor(concat_labels.astype(np.int64))
        if has_is_positive:
            result['is_positives'] = torch.tensor([int(e['is_positive']) for e in examples])
        return result


class Example(object):
    def __init__(self, uuid, header: List[Column], context: Tuple[List, List],
                 context_mentions: Tuple[Union[None, List], Union[None, List]]=(None, None),
                 column_data=None, column_data_used=None, is_positive=True,
                 answer_coordinates: List[Tuple[int, int]]=None, answers: List[str]=None, sql: str=None, **kwargs):
        self.uuid = uuid
        self.header = header
        self.context = context
        self.context_mentions = [[[] for _ in range(len(context[0]))] if context_mentions[0] is None else context_mentions[0],
                                 [[] for _ in range(len(context[1]))] if context_mentions[1] is None else context_mentions[1]]
        self.column_data = column_data
        self.column_data_used = column_data_used
        self.is_positive = is_positive
        self.answer_coordinates = answer_coordinates
        self.answers = answers
        self.sql = sql

        for key, val in kwargs.items():
            setattr(self, key, val)

    def highlight_table(self,
                        tokenizer,
                        highlight_parts: str = 'all',
                        highlight_template: str = '* {}'):
        assert highlight_parts in {'all', 'data', 'header'}
        if highlight_parts in {'data', 'all'}:  # highlight table data
            col2hl_rows: Dict[int, Set[int]] = defaultdict(set)
            for row_idx, col_idx in self.column_data_used:
                col2hl_rows[col_idx].add(row_idx)
            new_column_data: List[List[str]] = []
            for col_idx, column in enumerate(self.column_data):
                new_column_data.append([])
                for row_idx, cell in enumerate(column):
                    if row_idx in col2hl_rows[col_idx]:
                        cell = highlight_template.format(cell)
                    new_column_data[-1].append(cell)
            self.column_data = new_column_data

        if highlight_parts in {'header', 'all'}:  # hightlight table header
            for h in self.header:
                if h.used:
                    h.name = highlight_template.format(h.name)
                    h.name_tokens = tokenizer.tokenize(h.name)

    def only_keep_highlighted(self):
        # modify table data
        keep_rows: Set[int] = set()
        for row_idx, col_idx in self.column_data_used:
            keep_rows.add(row_idx)
        new_column_data: List[List[str]] = []
        for col_idx, column in enumerate(self.column_data):
            new_column_data.append([])
            for row_idx, cell in enumerate(column):
                if row_idx not in keep_rows:
                    continue
                new_column_data[-1].append(cell)
        # remove highted cell info because this is now stale
        self.column_data_used = None
        self.column_data = new_column_data

    @staticmethod
    def shuffle_table(example: Dict):
        data = example['table']['data']
        header = example['table']['header']
        data_used = example['table']['data_used'] if 'data_used' in example['table'] else example['table']['used_data']
        if len(data) <= 0:
            return
        num_rows = len(data)
        num_cols = len(data[0])
        data_used = [(r, c) for r, c in data_used if r < num_rows and c < num_cols]  # remove out-of-bound indices
        if len(data_used) <= 1:
            return

        # get used rows/columns
        row_idxs, col_idxs = list(zip(*data_used))
        row_idxs = list(set(row_idxs))
        col_idxs = list(set(col_idxs))
        # shuffle
        random.shuffle(row_idxs)
        random.shuffle(col_idxs)

        full_row_idxs = []
        full_col_idxs = []
        row_old2new: Dict[int, int] = {}
        col_old2new: Dict[int, int] = {}
        ind = 0
        for i in range(num_rows):
            if i in row_idxs:
                full_row_idxs.append(row_idxs[ind])
                row_old2new[i] = row_idxs[ind]
                ind += 1
            else:
                full_row_idxs.append(i)
        ind = 0
        for i in range(num_cols):
            if i in col_idxs:
                full_col_idxs.append(col_idxs[ind])
                col_old2new[i] = col_idxs[ind]
                ind += 1
            else:
                full_col_idxs.append(i)

        new_header = []
        for j in range(num_cols):
            new_header.append(header[full_col_idxs[j]])
        new_data = []
        for i in range(num_rows):
            new_data.append([])
            for j in range(num_cols):
                new_data[-1].append(data[full_row_idxs[i]][full_col_idxs[j]])
        new_data_used = [(row_old2new[r], col_old2new[c]) for r, c in data_used]

        example['table']['data'] = new_data
        example['table']['header'] = new_header
        if 'used_data' in example['table']:
            del example['table']['used_data']
        example['table']['data_used'] = new_data_used

    def serialize(self):
        example = {
            'uuid': self.uuid,
            'source': self.source,
            'context': self.context,
            'context_mentions': self.context_mentions,
            'column_data': self.column_data,
            'column_data_used': self.column_data_used,
            'is_positive': self.is_positive,
            'header': [x.to_dict() for x in self.header],
            'answer_coordinates': self.answer_coordinates,
            'answers': self.answers,
            'sql': self.sql,
        }

        return example

    def get_table(self):
        num_columns = len(self.header)
        num_rows = len(self.column_data[0])
        row_data = []
        for row_id in range(num_rows):
            row = [self.column_data[i][row_id] for i in range(num_columns)]
            row_data.append(row)

        table = Table(self.uuid, header=self.header, data=row_data)

        return table

    @classmethod
    def from_serialized(cls, data) -> 'Example':
        header = [Column(**x) for x in data['header']]
        data['header'] = header
        return Example(**data)

    @staticmethod
    def overlap(s1: int, e1: int, s2: int, e2: int):  # inclusive, exclusive
        # 0 -> overlap
        # 1 -> the first passed the second
        # -1 -> the first not reached second
        if s1 >= e2:
            return 1
        if e1 <= s2:
            return -1
        return 0

    @staticmethod
    def char_index2token_index(offsets: List[Tuple[int, int]],
                               char_mentions: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]],
                               added_prefix_space: bool) -> List[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
        char_adjust = -1 if added_prefix_space else 0
        token_mentions: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = [None for _ in range(len(char_mentions))]
        midx = tid = 0
        while tid < len(offsets) and midx < len(char_mentions):
            start, end = offsets[tid]
            start = start + char_adjust
            end = end + char_adjust
            status = None
            while midx < len(char_mentions):
                (ms, me), cells = char_mentions[midx]
                status = Example.overlap(start, end, ms, me)
                if status != 1:
                    break
                midx += 1
            if status == 0:  # overlap
                if token_mentions[midx] is None:
                    token_mentions[midx] = ((tid, tid + 1), cells)
                else:
                    token_mentions[midx] = ((token_mentions[midx][0][0], tid + 1), token_mentions[midx][1])
                if char_mentions[midx][0][1] < end:
                    midx += 1
                else:
                    tid += 1
            else:  # tid behind
                tid += 1
        location2cells: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        for tm in token_mentions:
            if tm is None:
                #logging.warning(f'char2token index error {offsets} {char_mentions}')
                continue
            if tm[0] not in location2cells:
                location2cells[tm[0]] = set(tm[1])
            else:  # multiple mentions might be from the same token
                location2cells[tm[0]] = location2cells[tm[0]] | set(tm[1])
        token_mentions = [(k, sorted(list(location2cells[k]))) for k in sorted(location2cells.keys())]  # TODO: avoid overlap between mentions?
        return token_mentions

    @staticmethod
    def mention_postprocess(mentions: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]]):
        # dedup, sort, remove overlap
        de_overlap = []
        max_prev = -1
        for (s, e), cells in sorted(mentions, key=lambda x: x[0]):
            if s >= e:
                continue
            if s < max_prev:
                continue
            de_overlap.append(((s, e), list(map(tuple, cells))))
            max_prev = max(max_prev, e)
        return de_overlap

    @classmethod
    def from_dict(cls, entry: Dict, tokenizer: Optional[BertTokenizer], tokenizer_fast: Optional[BertTokenizerFast]=None, suffix=None) -> 'Example':
        def _get_data_source():
            if entry['uuid'].startswith('wiki'):
                return 'wiki'
            if entry['uuid'].startswith('totto_') or entry['uuid'].startswith('wsql_') or entry['uuid'].startswith('tablefact_'):
                return 'grappa'
            return 'common_crawl'

        source = _get_data_source()

        header_entry = entry['header'] if source == 'wiki' else entry['table']['header']
        header: List[Column] = []
        column_data = []
        for col in header_entry:
            sample_value = col['sample_value']['value']
            if tokenizer:
                name_tokens = tokenizer.tokenize(col['name'])
            else: name_tokens = None
            used = col['used'] if 'used' in col else False
            value_used = col['value_used'] if 'value_used' in col else used  # fall back to used if not exist
            column = Column(col['name'],
                            col['type'],
                            sample_value,
                            name_tokens=name_tokens,
                            used=used,
                            value_used=value_used)
            header.append(column)

        if source == 'wiki':
            for row in entry['data'][1:]:
                for col_id, (tag, cell_val) in enumerate(row):
                    if col_id >= len(column_data):
                        column_data.append([])

                    column_data[col_id].append(cell_val)
        else:
            for row in entry['table']['data']:
                for col_id, (cell_val) in enumerate(row):
                    if col_id >= len(column_data):
                        column_data.append([])

                    column_data[col_id].append(cell_val)

        # TODO: modify files to avoid this postprocess
        if 'data_used' in entry['table']:
            column_data_used = sorted(set(map(tuple, entry['table']['data_used'])))
        elif 'used_data' in entry['table']:
            column_data_used = sorted(set(map(tuple, entry['table']['used_data'])))
        else:
            column_data_used = []

        context_before = []
        context_before_offsets = []
        context_after = []
        context_after_offsets = []

        aps = False
        if source == 'wiki':
            for para in entry['context_before']:
                for sent in para:
                    if tokenizer:
                        sent = tokenizer.tokenize(sent)

                    context_before.append(sent)

            caption = entry['caption']
            if caption:
                if tokenizer:
                    caption = tokenizer.tokenize(entry['caption'])

                context_before.append(caption)
        else:
            for sent in entry['context_before']:
                offsets = []
                if tokenizer_fast:
                    sent_fast = tokenizer_fast(sent, add_special_tokens=False, return_offsets_mapping=True)
                    offsets = sent_fast['offset_mapping']
                    if 'added_prefix_space' in sent_fast: aps = sent_fast['added_prefix_space']
                if tokenizer:
                    raw_sent = sent
                    sent = tokenizer.tokenize(sent)
                    if tokenizer_fast and len(sent_fast['input_ids']) != len(sent):
                        sent = tokenizer.convert_ids_to_tokens(sent_fast['input_ids'])
                        logging.warning(f"tokenizer results inconsistent {raw_sent} {sent} {sent_fast['input_ids']}")
                context_before.append(sent)
                context_before_offsets.append(offsets)

            for sent in entry['context_after']:
                offsets = []
                if tokenizer_fast:
                    sent_fast = tokenizer_fast(sent, add_special_tokens=False, return_offsets_mapping=True)
                    offsets = sent_fast['offset_mapping']
                    if 'added_prefix_space' in sent_fast: aps = sent_fast['added_prefix_space']
                if tokenizer:
                    sent = tokenizer.tokenize(sent)
                    if tokenizer_fast:
                        assert len(sent_fast['input_ids']) == len(sent), 'tokenizer results inconsistent'
                context_after.append(sent)
                context_after_offsets.append(offsets)

        cbm = cam = None
        if 'context_before_mentions' in entry:
            cbm = entry['context_before_mentions']
            if 'context_before_mentions_cells' in entry:
                cbm = [list(zip(single_cbm, single_cbmc)) for single_cbm, single_cbmc in zip(cbm, entry['context_before_mentions_cells'])]
            else:
                cbm = [[(m, []) for m in single_cbm] for single_cbm in cbm]
            assert len(cbm) == len(context_before_offsets)
            # TODO: modify files to avoid this postprocess
            if tokenizer is not None or tokenizer_fast is not None:
                cbm = [Example.char_index2token_index(off, Example.mention_postprocess(cm), added_prefix_space=aps)
                    for off, cm in zip(context_before_offsets, cbm)]
            else:
                cbm = [Example.mention_postprocess(cm) for cm in cbm]
        # TODO: add after

        uuid = entry['uuid']
        is_positive = entry['is_positive'] if 'is_positive' in entry else True

        answer_coordinates = entry['answer_coordinates'] if 'answer_coordinates' in entry else None
        answers = entry['answers'] if 'answers' in entry else None
        sql = entry['sql'] if 'sql' in entry else None

        metadata = entry['metadata'] if 'metadata' in entry else None

        return cls(uuid, header,
                   (context_before, context_after),
                   (cbm, cam),
                   column_data=column_data,
                   column_data_used=column_data_used,
                   source=source,
                   is_positive=is_positive,
                   answer_coordinates=answer_coordinates,
                   answers=answers,
                   sql=sql,
                   metadata=metadata)


class TableDatabase:
    def __init__(self, tokenizer, backend='redis', num_workers=None):
        self.tokenizer = tokenizer
        self.backend = backend

        if self.backend == 'redis':
            self.restore_redis_client()
            self.client.flushall(asynchronous=False)

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
            if num_workers > 20:
                num_workers -= 2

        self.num_workers = num_workers
        self._cur_index = multiprocessing.Value('i', 0)

    def restore_redis_client(self):
        self.client = redis.Redis(host='localhost', port=6379, db=0)

    @staticmethod
    def __load_process_zmq(file, num_workers):
        context = zmq.Context()
        job_sender = context.socket(zmq.PUSH)
        # job_sender.setsockopt(zmq.LINGER, -1)
        job_sender.bind("tcp://127.0.0.1:5557")

        controller = context.socket(zmq.PUB)
        controller.bind('tcp://127.0.0.1:5558')

        # wait for sometime to let all workers to connect
        time.sleep(5)

        cnt = 0
        with file.open() as f:
            for line in f:
                cnt += 1
                job_sender.send_string(line)
                # if cnt % 10000 == 0:
                #     print(f'read {cnt} examples')
                #     sys.stdout.flush()

        controller.send_string('kill')
        # print('Reader count:', cnt)

        job_sender.close()
        controller.close()
        context.destroy()
        # while True:
        #     job_sender.send_string('')
        #     time.sleep(0.1)

        # while True:
        #     time.sleep(1)

    @staticmethod
    def __example_worker_process_zmq(tokenizer, db):
        context = zmq.Context()
        job_receiver = context.socket(zmq.PULL)
        # job_receiver.setsockopt(zmq.LINGER, -1)
        job_receiver.connect("tcp://127.0.0.1:5557")

        controller = context.socket(zmq.SUB)
        controller.connect("tcp://127.0.0.1:5558")
        controller.setsockopt(zmq.SUBSCRIBE, b"")

        poller = zmq.Poller()
        poller.register(job_receiver, zmq.POLLIN)
        poller.register(controller, zmq.POLLIN)

        cache_client = redis.Redis(host='localhost', port=6379, db=0)
        buffer_size = 20000

        def _add_to_cache():
            if buffer:
                with db._cur_index.get_lock():
                    index_end = db._cur_index.value + len(buffer)
                    db._cur_index.value = index_end
                index_start = index_end - len(buffer)
                values = {str(i): val for i, val in zip(range(index_start, index_end), buffer)}
                cache_client.mset(values)
                del buffer[:]

        cnt = 0
        buffer = []
        can_exit = False
        while True:
            triggered = False
            socks = dict(poller.poll(timeout=2000))

            if socks.get(job_receiver) == zmq.POLLIN:
                triggered = True
                job = job_receiver.recv_string()
                if job:
                    cnt += 1
                    # print(cnt)
                    example = Example.from_dict(ujson.loads(job), tokenizer, suffix=None)

                    if TableDatabase.is_valid_example(example):
                        data = example.serialize()
                        buffer.append(msgpack.packb(data, use_bin_type=True))

                    if len(buffer) >= buffer_size:
                        _add_to_cache()

            # else:
            #     job_receiver.close()
            #     _add_to_cache()
            #     break

            if socks.get(controller) == zmq.POLLIN:
                triggered = True
                print(controller.recv_string())
                can_exit = True

            # timeout
            # print(socks)
            if not socks and can_exit:
                print('Processor exit...')
                break

            if socks and not triggered:
                print(socks)

        _add_to_cache()
        job_receiver.close()
        controller.close()
        context.destroy()

    @classmethod
    def from_jsonl(
        cls,
        file_path: Path,
        tokenizer: Optional[BertTokenizer] = None,
        tokenizer_fast: Optional[BertTokenizerFast] = None,
        backend='redis',
        num_workers=None,
        indices=None,
        skip_column_name_longer_than: int=10,
        not_skip_empty_column_name: bool=False,
        only_keep_highlighted_rows: bool = False,
        highlight_table: str = None,
    ) -> 'TableDatabase':
        file_path = Path(file_path)

        db = cls(backend=backend, num_workers=num_workers, tokenizer=tokenizer)

        if backend == 'redis':
            assert indices is None
            db.load_data_to_redis(file_path)
        elif backend == 'memory':
            example_store = dict()
            if indices: indices = set(indices)

            with file_path.open() as f:
                for idx, json_line in enumerate(tqdm(f, desc=f'Loading Tables from {str(file_path)}', unit='entries', file=sys.stdout)):
                    if indices and idx not in indices:
                        continue

                    example = Example.from_dict(
                        ujson.loads(json_line),
                        tokenizer,
                        tokenizer_fast=tokenizer_fast,
                        suffix=None
                    )
                    if highlight_table:
                        example.highlight_table(tokenizer, highlight_parts=highlight_table)
                    if only_keep_highlighted_rows:
                        example.only_keep_highlighted()

                    if TableDatabase.is_valid_example(
                            example,
                            skip_column_name_longer_than=skip_column_name_longer_than,
                            not_skip_empty_column_name=not_skip_empty_column_name):
                        example_store[idx] = example
                        
            db.__example_store = example_store

        return db
    
    def to_jsonl(self, file_path):
        with open(file_path, 'w') as fout:
            for example in tqdm(self):
                context = example.context[0][0]
                assert type(context) is str and len(context), f'#{context}#'
                mentions = [m[0] for m in example.context_mentions[0][0]]
                answers = example.answers
                table = {
                    'header': [h.name for h in example.header],
                    'rows': [[example.column_data[c][r] for c in range(len(example.column_data))] for r in range(len(example.column_data[0]))],
                }
                for h in table['header']:
                    assert type(h) is str, f'#{h}#{type(h)}'
                assert len(table['rows'])
                for row in table['rows']:
                    assert len(row) == len(table['header'])
                    for cell in row:
                        assert type(cell) is str, f'#{cell}#{type(cell)}'
                fout.write(json.dumps({'context': context, 'mentions': mentions, 'table': table, 'answers': answers}) + '\n')

    def load_data_to_redis(self, file_path: Path):
        reader = multiprocessing.Process(target=self.__load_process_zmq, args=(file_path, self.num_workers),
                                         daemon=True)

        workers = []
        for _ in range(self.num_workers):
            worker = multiprocessing.Process(target=self.__example_worker_process_zmq,
                                             args=(self.tokenizer, self),
                                             daemon=True)
            worker.start()
            workers.append(worker)

        while any(not worker.is_alive() for worker in workers):
            time.sleep(0.1)

        reader.start()

        stop_count = 0
        db_size = 0
        with tqdm(desc=f"Loading Tables from {str(file_path)}", unit=" entries", file=sys.stdout) as pbar:
            while True:
                cur_db_size = len(self)
                pbar.update(cur_db_size - db_size)
                db_size = cur_db_size

                all_worker_finished = all(not w.is_alive() for w in workers)
                if all_worker_finished:
                    print(f'all workers stoped!')
                    break

                time.sleep(1)

        for worker in workers:
            worker.join()
        reader.terminate()

    def __len__(self):
        if self.backend == 'redis':
            return self._cur_index.value
        elif self.backend == 'memory':
            return len(self.__example_store)
        else:
            raise RuntimeError()

    def __contains__(self, item):
        assert self.backend == 'memory'
        return item in self.__example_store

    def __getitem__(self, item) -> Example:
        if self.backend == 'redis':
            result = self.client.get(str(item))
            if result is None:
                raise IndexError(item)
    
            example = Example.from_serialized(msgpack.unpackb(result, raw=False))
        elif self.backend == 'memory':
            example = self.__example_store[item]
        else:
            raise RuntimeError()

        return example

    def __iter__(self) -> Iterator[Example]:
        if self.backend == 'redis':
            for i in range(len(self)):
                yield self[i]
        else:
            for example in self.__example_store.values():
                yield example

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.backend == 'redis':
            print('Flushing all entries in cache')
            self.client.flushall()

    @classmethod
    def is_valid_example(cls,
                         example,
                         skip_column_name_longer_than: int = 10,
                         not_skip_empty_column_name: bool = False):
        # TODO: move this to preprocess pre-processing
        if skip_column_name_longer_than > 0:
            if any(len(col.name.split(' ')) > skip_column_name_longer_than for col in example.header):
                return False

        if not not_skip_empty_column_name and any(len(col.name_tokens) == 0 for col in example.header):
            return False

        return True
