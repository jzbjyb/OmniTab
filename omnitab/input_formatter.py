#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil
from random import choice, shuffle, sample, random, randint
from typing import List, Callable, Dict, Any, Union, Set, Tuple
import copy
from collections import defaultdict
import numpy as np

from omnitab.utils import BertTokenizer
from omnitab.config import TableBertConfig
from omnitab.dataset import Example
from omnitab.table import Column, Table

trim_count = {'total': 0, 'trim': 0, 'mask_no_enough': 0}
tablededup2count: Dict[int, int] = defaultdict(lambda: 0)


class TableBertBertInputFormatter(object):
    def __init__(self, config: TableBertConfig, tokenizer: BertTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.tokenizer_fast = config.tokenizer_fast_cls.from_pretrained(config.base_model_name)
        if hasattr(self.tokenizer, 'vocab'):
            self.vocab_list = list(self.tokenizer.vocab.keys())
        elif hasattr(self.tokenizer, 'get_vocab'):
            self.vocab_list = list(self.tokenizer.get_vocab().keys())
        else:
            raise Exception('cannot find vocab for the tokenizer {}'.format(self.tokenizer))

class TableTooLongError(ValueError):
    pass


class VanillaTableBertInputFormatter(TableBertBertInputFormatter):
    def get_cell_input(
        self,
        column: Column,
        cell_value: List[str],
        token_offset: int = 0,
        cell_input_template: List[str] = None,
    ):
        input = []
        span_map = {
            'first_token': (token_offset, token_offset + 1)
        }
        cell_input_template = cell_input_template or self.config.cell_input_template
        assert type(cell_input_template) is list

        for token in cell_input_template:
            start_token_abs_position = len(input) + token_offset
            if token == 'column':
                name_tokens = column.name_tokens if self.config.max_column_len is None else column.name_tokens[:self.config.max_column_len]
                span_map['column_name'] = (start_token_abs_position, start_token_abs_position + len(name_tokens))
                input.extend(name_tokens)
            elif token == 'value':
                value_tokens = cell_value if self.config.max_cell_len is None else cell_value[:self.config.max_cell_len]
                span_map['value'] = (start_token_abs_position, start_token_abs_position + len(value_tokens))
                input.extend(value_tokens)
            elif token == 'type':
                span_map['type'] = (start_token_abs_position, start_token_abs_position + 1)
                input.append(column.type)
            else:
                span_map.setdefault('other_tokens', []).append(start_token_abs_position)
                input.append(token)

        span_map['whole_span'] = (token_offset, token_offset + len(input))

        return input, span_map

    def get_cell_input_columnwise(
        self,
        cell_index: List[str],
        cell_value: List[str],
        token_offset: int = 0,
        cell_input_template: List[str] = None,
    ):
        input = []
        span_map = {
            'first_token': (token_offset, token_offset + 1)
        }
        cell_input_template = cell_input_template or self.config.cell_input_template

        for token in cell_input_template:
            start_token_abs_position = len(input) + token_offset
            if token == 'index':
                index_tokens = cell_index[:self.config.max_column_len]
                span_map['index'] = (start_token_abs_position, start_token_abs_position + len(index_tokens))
                input.extend(index_tokens)
            elif token == 'value':
                value_tokens = cell_value if self.config.max_cell_len is None else cell_value[:self.config.max_cell_len]
                span_map['value'] = (start_token_abs_position, start_token_abs_position + len(value_tokens))
                input.extend(value_tokens)
            else:
                span_map.setdefault('other_tokens', []).append(start_token_abs_position)
                input.append(token)
        span_map['whole_span'] = (token_offset, token_offset + len(input))

        return input, span_map

    def get_input(self, context: List[str], table: Table, additional_rows: List[List[Any]] = [],
                  trim_long_table: Union[None, int] = 0, shuffle: bool = False,
                  cell_input_template: List[str] = None):
        row_data = [
            column.sample_value_tokens
            for column in table.header
        ]

        return self.get_row_input(context, table.header, row_data, additional_rows,
                                  trim_long_table=trim_long_table, shuffle=shuffle,
                                  cell_input_template=cell_input_template)

    def concate_cells(self,
                      header: List,
                      row_data: List,
                      table_tokens_start_idx: int,
                      trim_long_table: int,
                      max_table_token_length: int,
                      cell_input_template: List[str] = None,
                      column_delimiters: List[str] = None,
                      multi_row_split: List[bool] = None,
                      column_wise: bool = False):
        row_input_tokens = []
        column_token_span_maps = []
        column_start_idx = table_tokens_start_idx
        column_delimiters = column_delimiters or [self.config.column_delimiter]
        column_delimiters_first = [self.config.column_delimiter_first] or column_delimiters
        row_delimiters = [self.config.row_delimiter] if self.config.row_delimiter is not None else []

        col_id = 0
        prev_is_mrs = True
        for col_id, column in enumerate(header):
            value_tokens = row_data[col_id]
            is_mrs = False if multi_row_split is None else multi_row_split[col_id]

            if column_wise:
                column_input_tokens, token_span_map = self.get_cell_input_columnwise(
                    column,
                    value_tokens,
                    token_offset=column_start_idx,
                    cell_input_template=cell_input_template
                )
                column_input_tokens.extend(row_delimiters if is_mrs else column_delimiters)
            else:
                column_input_tokens, token_span_map = self.get_cell_input(
                    column,
                    value_tokens,
                    token_offset=column_start_idx,
                    cell_input_template=cell_input_template
                )
                cd = column_delimiters_first if prev_is_mrs else column_delimiters
                column_input_tokens.extend(row_delimiters if is_mrs else cd)

            prev_is_mrs = is_mrs
            early_stop = False
            if trim_long_table is not None:
                trim_count['total'] += 1
                if len(row_input_tokens) + len(column_input_tokens) > max_table_token_length:
                    trim_count['trim'] += 1
                    valid_column_input_token_len = max_table_token_length - len(row_input_tokens)
                    column_input_tokens = column_input_tokens[:valid_column_input_token_len]
                    end_index = column_start_idx + len(column_input_tokens)
                    keys_to_delete = []
                    for key in token_span_map:
                        if key in {'column_name', 'type', 'value', 'whole_span', 'index'}:
                            span_start_idx, span_end_idx = token_span_map[key]
                            if span_start_idx < end_index < span_end_idx:
                                token_span_map[key] = (span_start_idx, end_index)
                            elif end_index <= span_start_idx:
                                keys_to_delete.append(key)
                        elif key == 'other_tokens':
                            old_positions = token_span_map[key]
                            new_positions = [idx for idx in old_positions if idx < end_index]
                            if not new_positions:
                                keys_to_delete.append(key)

                    for key in keys_to_delete:
                        del token_span_map[key]

                    # nothing left, we just skip this cell and break
                    if len(token_span_map) == 0:
                        break

                    early_stop = True
                elif len(row_input_tokens) + len(column_input_tokens) == max_table_token_length:
                    early_stop = True
            elif len(row_input_tokens) + len(column_input_tokens) > max_table_token_length:
                break

            row_input_tokens.extend(column_input_tokens)
            column_start_idx = column_start_idx + len(column_input_tokens)
            column_token_span_maps.append(token_span_map)

            if early_stop: break

        # delete delimiters at the end
        for deli in [column_delimiters, column_delimiters_first, row_delimiters]:
            if len(deli) == 1 and row_input_tokens[-1] == deli[0]:
                del row_input_tokens[-1]
            elif len(deli) > 1 and ''.join(row_input_tokens[-len(deli):]) == ''.join(deli):
                row_input_tokens = row_input_tokens[:-len(deli)]

        return row_input_tokens, column_token_span_maps, col_id

    def iter_table_data(self, header: List[Column], rows: List[List], column_wise: bool) -> Tuple[List, List, List]:
        index_cells = []
        value_cells = []
        multi_split = []
        if column_wise:
            for col_idx in range(len(header)):
                index_cells.append(self.tokenizer.tokenize('HEADER'))
                value_cells.append(header[col_idx].name_tokens)
                multi_split.append(False)
                for row_idx in range(len(rows)):
                    index_cells.append(self.tokenizer.tokenize(f'ROW{row_idx}'))
                    value_cells.append(rows[row_idx][col_idx])
                    multi_split.append(row_idx == len(rows) - 1)
        else:
            for col_idx in range(len(header)):
                index_cells.append(self.tokenizer.tokenize('HEADER'))
                value_cells.append(header[col_idx].name_tokens)
                multi_split.append(col_idx == len(header) - 1)
            for row_idx in range(len(rows)):
                for col_idx in range(len(header)):
                    index_cells.append(self.tokenizer.tokenize(f'ROW{row_idx}'))
                    value_cells.append(rows[row_idx][col_idx])
                    multi_split.append(col_idx == len(header) - 1)
        return index_cells, value_cells, multi_split

    def get_row_input(self,
                      context: List[str],
                      header: List[Column],
                      row_data: List[Any],
                      additional_rows: List[List[Any]] = [],
                      trim_long_table: Union[None, int] = 0,
                      shuffle: bool = False,
                      cell_input_template: List[str] = None):  # none means not trim
        column_wise = self.config.column_wise
        use_row_data = self.config.top_row_count == 0 and len(row_data) > 0
        max_total_len = trim_long_table or TableBertConfig.MAX_SOURCE_LEN
        skip_sep_in_middle = self.config.skip_sep_in_middle
        num_sp_tokens = 1 if skip_sep_in_middle else 2

        additional_rows = copy.deepcopy(additional_rows)  # we will modify it in place

        if self.config.table_linearization == 'tapex':
            if use_row_data:
                raise Exception('tapex linearization is only used on raw table data instead of sampled data')
            for i, row in enumerate(additional_rows):  # add row index at the beginning of each row
                row.insert(0, self.tokenizer.tokenize(f'row {i + 1}'))
            # header is the first row
            additional_rows.insert(0, [self.tokenizer.tokenize('col')] + [h.name_tokens for h in header])
            header = [Column.get_dummy()] + header  # the dummy header is for the first index element

        if self.config.context_first:
            table_tokens_start_idx = len(context) + num_sp_tokens  # account for cls and sep
            # account for cls and sep, and the ending sep
            max_table_token_length = max_total_len - len(context) - num_sp_tokens - 1
        else:
            table_tokens_start_idx = 1  # account for starting cls
            # account for cls and sep, and the ending sep
            max_table_token_length = max_total_len - len(context) - num_sp_tokens - 1

        if column_wise and len(additional_rows) > 0:
            # get max #rows that can fit into the max length which is necessary for column_wise iteration
            index_cells, value_cells, multi_split = self.iter_table_data(header, additional_rows, column_wise=False)
            _, _, col_id = self.concate_cells(
                index_cells, value_cells,
                table_tokens_start_idx=table_tokens_start_idx,
                trim_long_table=trim_long_table, max_table_token_length=max_table_token_length,
                cell_input_template=cell_input_template, multi_row_split=multi_split, column_wise=True)
            num_fit_rows = max((col_id + 1) // len(header) - 1, 0)  # remove the header row
            additional_rows = additional_rows[:num_fit_rows]

        # generate table tokens
        if column_wise:
            if use_row_data and shuffle and len(additional_rows) > 0:
                additional_rows.insert(randint(0, len(additional_rows)), row_data)
                combined_rows = additional_rows
            else:
                combined_rows = [row_data] + additional_rows if use_row_data else additional_rows
            index_cells, value_cells, multi_split = self.iter_table_data(header, combined_rows, column_wise=True)
            row_input_tokens, column_token_span_maps, col_id = self.concate_cells(
                index_cells, value_cells, table_tokens_start_idx=table_tokens_start_idx, trim_long_table=trim_long_table,
                max_table_token_length=max_table_token_length, cell_input_template=cell_input_template,
                multi_row_split=multi_split, column_wise=True)
        else:
            ext_header = header * (len(additional_rows) + int(use_row_data))
            multi_row_split = ([False] * (len(header) - 1) + [True]) * (len(additional_rows) + int(use_row_data))
            if use_row_data and shuffle and len(additional_rows) > 0:
                # try to see how many rows can fit into the max length
                _, _, col_id = self.concate_cells(
                    header * len(additional_rows), [i for r in additional_rows for i in r], table_tokens_start_idx=table_tokens_start_idx,
                    trim_long_table=trim_long_table, max_table_token_length=max_table_token_length, cell_input_template=cell_input_template, multi_row_split=multi_row_split)
                num_fit_rows = col_id // len(header)
                additional_rows.insert(randint(0, max(num_fit_rows - 1, 0)), row_data)
                ext_row_data = [i for r in additional_rows for i in r]
            else:
                ext_row_data = (row_data if use_row_data else []) + [i for r in additional_rows for i in r]
            row_input_tokens, column_token_span_maps, col_id = self.concate_cells(
                ext_header, ext_row_data, table_tokens_start_idx=table_tokens_start_idx, trim_long_table=trim_long_table,
                max_table_token_length=max_table_token_length, cell_input_template=cell_input_template, multi_row_split=multi_row_split)

        # it is possible that the first cell to too long and cannot fit into `max_table_token_length`
        # we need to discard this sample
        if len(row_input_tokens) == 0:
            raise TableTooLongError()

        middle_sep = [] if skip_sep_in_middle else [self.config.sep_token]
        if self.config.context_first:
            sequence = [self.config.cls_token] + context + middle_sep + row_input_tokens + [self.config.sep_token]
            segment_a_length = len(context) + num_sp_tokens
            context_span = (0, num_sp_tokens - 1 + len(context))
        else:
            sequence = [self.config.cls_token] + row_input_tokens + middle_sep + context + [self.config.sep_token]
            segment_a_length = len(row_input_tokens) + num_sp_tokens
            context_span = (len(row_input_tokens) + num_sp_tokens - 1, len(row_input_tokens) + num_sp_tokens - 1 + 1 + len(context) + 1)

        # get column_token_to_column_id
        column_token_to_column_id = np.full(len(sequence), dtype=np.int, fill_value=-1)  # init with -1
        span_key = self.config.column_repr_dpr

        for col_id, span in enumerate(column_token_span_maps):
            if span_key not in span: break  # table overflow
            col_start, col_end = span[span_key]
            column_token_to_column_id[col_start:col_end] = col_id

        instance = {
            'tokens': sequence,
            #'token_ids': self.tokenizer.convert_tokens_to_ids(sequence),
            'segment_a_length': segment_a_length,
            # 'segment_ids': segment_ids,
            'column_spans': column_token_span_maps,
            'column_token_to_column_id': column_token_to_column_id.tolist(),
            'context_length': 1 + len(context),  # beginning cls/sep + input question
            'context_span': context_span,
            # 'context_token_indices': context_token_indices
        }

        return instance

    def get_context_token_to_mention_id(self,
                                        context: List[str],
                                        context_mentions: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]],
                                        size: int,
                                        max_num_cells: int = None,
                                        row_size: int = None):
        if not self.config.context_first:
            raise NotImplementedError
        offset = 1  # cls token
        context_token_to_mention_id = np.full(size, dtype=np.int, fill_value=-1)  # init with -1
        mentions_cells: List[Tuple[int, int]] = []
        for idx, ((start, end), cells) in enumerate(context_mentions):
            if end > len(context):
                continue
            context_token_to_mention_id[start + offset:end + offset] = idx
            if max_num_cells is not None:
                for row_idx, col_idx in cells:
                    cell_idx = row_idx * row_size + col_idx
                    if cell_idx >= max_num_cells:
                        continue
                    mentions_cells.append((idx, cell_idx))
        return context_token_to_mention_id.tolist(), mentions_cells

    def get_multiple_rows(self, example, keep_num_rows: int, exclude: Set[int]={}, use_sample: bool = True, skip_empty: bool = True):
        additional_rows = []
        if len(example.column_data) <= 0 or keep_num_rows <= 0:  # no data or no sample
            return additional_rows

        # get valid rows
        num_rows = len(example.column_data[0])
        valid_rows = []
        for row_idx in range(num_rows):
            if row_idx in exclude:
                continue
            if not skip_empty:
                valid_rows.append(row_idx)
            else:
                valid = True
                for col_idx, column in enumerate(example.header):
                    val = example.column_data[col_idx][row_idx]
                    if val is None or len(val) == 0:
                        valid = False
                        break
                if valid:
                    valid_rows.append(row_idx)

        if use_sample:  # sample rows
            sampled_rows = sample(valid_rows, min(len(valid_rows), keep_num_rows))
        else:  # use the top rows
            sampled_rows = valid_rows[:keep_num_rows]
        additional_rows = [[] for _ in range(len(sampled_rows))]
        for col_idx, column in enumerate(example.header):
            for i, row_idx in enumerate(sampled_rows):
                additional_rows[i].append(self.tokenizer.tokenize(example.column_data[col_idx][row_idx]))
        return additional_rows

    def get_a_row(self, example: Example):
        additional_rows = []
        answer = None
        if 'qa' in self.config.seq2seq_format:  # use the first answer
            answer = example.answers[0]

        if 'firstansrow' in self.config.seq2seq_format:  # sample from other rows
            ans_coord = example.answer_coordinates[0] if len(example.answer_coordinates) > 0 else None
            exclude = {} if ans_coord is None else {ans_coord[0]}
            if self.config.additional_row_count:
                additional_rows = self.get_multiple_rows(example, self.config.additional_row_count, exclude=exclude, use_sample=True)
        elif self.config.additional_row_count:
            raise NotImplementedError
        elif self.config.top_row_count:
            additional_rows = self.get_multiple_rows(example, self.config.top_row_count, use_sample=False, skip_empty=False)

        for col_idx, column in enumerate(example.header):
            if 'firstansrow' in self.config.seq2seq_format and ans_coord is not None:
                # use the row that contains the first answer
                # if ans_coord does not exist, go to the following conditions
                sampled_value = example.column_data[col_idx][ans_coord[0]]
            elif self.config.use_sampled_value:
                sampled_value = column.sample_value
            else:
                col_values = example.column_data[col_idx]
                col_values = [val for val in col_values if val is not None and len(val) > 0]
                sampled_value = choice(col_values) if len(col_values) > 0 else ''

            if sampled_value is None or len(sampled_value) <= 0:
                # use a special symbol if this column is empty
                sampled_value = '-'

            # print('chosen value', sampled_value)
            sampled_value_tokens = self.tokenizer.tokenize(sampled_value)
            column.sample_value_tokens = sampled_value_tokens

        return additional_rows, answer

    def verify_mention_cell(self, tokens, context_token_to_mention_id, column_token_to_column_id, mentions_cells, debug: bool = False, **kwargs):
        if debug:
            cid2tokens: Dict[int, List[int]] = defaultdict(list)
            mid2tokens: Dict[int, List[int]] = defaultdict(list)
            for t, cid, mid in zip(tokens, column_token_to_column_id, context_token_to_mention_id):
                if cid != -1:
                    cid2tokens[cid].append(t)
                if mid != -1:
                    mid2tokens[mid].append(t)
            cid2tokens = {k: self.tokenizer_fast.decode(v) for k, v in cid2tokens.items()}
            mid2tokens = {k: self.tokenizer_fast.decode(v) for k, v in mid2tokens.items()}
            for mid, cid in mentions_cells:
                print(f'{mid} {mid2tokens[mid]}\t\t{cid} {cid2tokens[cid]}')
        assert len(context_token_to_mention_id) == len(tokens)
        assert len(column_token_to_column_id) == len(tokens)
        _context_token_to_mention_id = set(context_token_to_mention_id)
        _column_token_to_column_id = set(column_token_to_column_id)
        for mid, cid in mentions_cells:
            assert mid in _context_token_to_mention_id and cid in _column_token_to_column_id

    def get_pretraining_instances_from_example(
        self, example: Example,
        context_sampler: Callable
    ):
        instances = []
        context_iter = context_sampler(
            example, self.config.max_context_len, context_sample_strategy=self.config.context_sample_strategy)
        seq2seq_format = self.config.seq2seq_format

        for context, context_mentions in context_iter:
            # row_num = len(example.column_data)
            # sampled_row_id = choice(list(range(row_num)))

            additional_rows, answer = self.get_a_row(example)

            if seq2seq_format is None:
                instance = self.create_pretraining_instance(context, example.header, additional_rows)
                instance['source'] = example.source
                instances.append(instance)
            else:
                if 'mlm' in seq2seq_format or 'bart-mask' in seq2seq_format:
                    if 'mlm' in seq2seq_format:  # target is identical to source (which probably contains masks)
                        instance = self.create_pretraining_instance(context, example.header, additional_rows)
                        instance['target_token_ids'] = instance['token_ids']
                    elif 'bart-mask' in seq2seq_format:  # target is identical to source but masked tokens are re-filled
                        instance = self.create_pretraining_instance(context, example.header, additional_rows, single_mask=True)
                        instance['target_token_ids'] = instance['raw_token_ids']
                    instance['source'] = example.source
                    if 'bart-mask' not in seq2seq_format:
                        instance['context_token_to_mention_id'], instance['mentions_cells'] = \
                            self.get_context_token_to_mention_id(
                                context,
                                context_mentions,
                                size=len(instance['tokens']),
                                max_num_cells=np.max(instance['column_token_to_column_id']) + 1,
                                row_size=len(example.column_data))
                        self.verify_mention_cell(
                            instance['token_ids'],
                            instance['context_token_to_mention_id'],
                            instance['column_token_to_column_id'],
                            instance['mentions_cells'], debug=False)
                    instances.append(instance)
                if 'single' in seq2seq_format:
                    instances.extend(self.create_seq2seq_instances(context, example.header))
                if 'qa_firstansrow' in seq2seq_format:
                    instances.extend(self.create_qa_instances(context, example.header, answer, additional_rows))
                if 'qa_allrow' in seq2seq_format:
                    instances.extend(self.create_qa_allrow_instances(context, example.header, answer, additional_rows))
                if 'qa_tapex' in seq2seq_format:
                    instances.extend(self.create_qa_tapex_instances(context, example.header, answer, additional_rows))
                if 'totto' in seq2seq_format:
                    instances.extend(self.create_data2text_instances(context, example.header,
                                                                     metadata=example.metadata,
                                                                     additional_rows=additional_rows))
                if 'data2text' in seq2seq_format:
                    instances.extend(self.create_data2text_instances(context, example.header, additional_rows=additional_rows))
                if 'clean-text' in seq2seq_format:
                    instances.extend(self.create_clean_text_instances(context, example.header,
                                                                      metadata=example.metadata,
                                                                      additional_rows=additional_rows))
                if 'clean-fake-text' in seq2seq_format:
                    instances.extend(self.create_clean_text_instances(context, example.header,
                                                                      additional_rows=additional_rows, fake=True))
                if 'bidirection' in seq2seq_format:
                    instances.extend(self.create_bidirection_instances(example, context, example.header, additional_rows=additional_rows))

                if 'sqltable2nl' in seq2seq_format:
                    instances.extend(self.create_sqltable2nl_instances(example.metadata, example.header, additional_rows=additional_rows))
                elif 'sql2nl' in seq2seq_format:
                    instances.extend(self.create_sql2nl_instances(example.metadata))
                elif 'sql' in seq2seq_format:
                    instances.extend(self.create_sql_instances(context, example.header, example.sql))

                if 'cell-filling-mask' in seq2seq_format:
                    instance = self.create_cf_mask_instance(context, example.header)
                    instance['target_token_ids'] = instance['token_ids']
                    instances.append(instance)
                if 'cell-filling-gen' in seq2seq_format:
                    instances.append(self.create_cf_gen_instance(context, example.header))
                if 'schema-augmentation-mask' in seq2seq_format:
                    instance = self.create_sa_mask_instance(context, example.header)
                    instance['target_token_ids'] = instance['token_ids']
                    instances.append(instance)
                if 'schema-augmentation-gen' in seq2seq_format:
                    instances.append(self.create_sa_gen_instance(context, example.header))
                if 'mention-context' in seq2seq_format:
                    instances.extend(self.create_mention_context_instances(context, context_mentions, example, additional_rows))
                if 'salient-mask' in seq2seq_format:
                    instances.extend(self.create_salient_mask_instances(context, context_mentions, example, additional_rows))
                if 'mention-table' in seq2seq_format:
                    instances.extend(self.create_mention_table_instances(context, example, additional_rows))
                if 'mention-dedup-table' in seq2seq_format:
                    instances.extend(self.create_mention_table_instances(context, example, additional_rows, dedup=True))
                if 'table-row' in seq2seq_format:
                    instances.extend(self.create_table_row_instances(context, example, additional_rows))

            stop = False
            for fm in {'mlm', 'qa', 'sql', 'sql2nl', 'sqltable2nl', 'cell-filling', 'schema-augmentation', 'clean-text', 'bidirection'}:
                # for these formats, do not iterative over context
                if fm in seq2seq_format:
                    stop = True
                    break
            if stop:
                break

        return instances

    def create_table_row_instances(self, context, example: Example, additional_rows: List[List[Any]]):
        mtl = TableBertConfig.MAX_TARGET_LEN
        instances = []
        if len(additional_rows) <= 0:
            return instances
        table = Table('fake_table', example.header)
        for row_idx in sample(range(len(additional_rows)), min(len(additional_rows), self.config.max_num_mention_per_example)):
            target_row = additional_rows[row_idx]
            other_rows = additional_rows[:row_idx] + [[self.config.mask_token for _ in target_row]] + additional_rows[row_idx + 1:]
            instance = self.get_input(context, table, other_rows)
            target = self.concate_cells(
                example.header, target_row, table_tokens_start_idx=0,
                trim_long_table=0, max_table_token_length=mtl - 2)[0]
            target = [self.config.cls_token] + target[:mtl - 2] + [self.config.sep_token]
            instance = {
                'tokens': instance['tokens'],
                'token_ids': self.tokenizer.convert_tokens_to_ids(instance['tokens']),
                'target_tokens': target,
                'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
                'segment_a_length': instance['segment_a_length'],
                'masked_lm_positions': [],
                'masked_lm_labels': [],
                'masked_lm_label_ids': [],
                'info': None
            }
            instances.append(instance)
        return instances

    def create_mention_context_instances(self,
                                         context: List[str],
                                         context_mentions: List[Tuple[int, int]],
                                         example: Example,
                                         additional_rows: List[List[Any]]):
        mtl = TableBertConfig.MAX_TARGET_LEN
        instances = []
        if len(context_mentions) <= 0:
            return instances
        table = Table('fake_table', example.header)
        for (start, end), _ in sample(context_mentions, min(len(context_mentions), self.config.max_num_mention_per_example)):
            _context = context[:start] + [self.config.mask_token] + context[end:]
            instance = self.get_input(_context, table, additional_rows)
            target = [self.config.cls_token] + context[start:end][:mtl - 2] + [self.config.sep_token]
            instance = {
                'tokens': instance['tokens'],
                'token_ids': self.tokenizer.convert_tokens_to_ids(instance['tokens']),
                'target_tokens': target,
                'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
                'segment_a_length': instance['segment_a_length'],
                'masked_lm_positions': [],
                'masked_lm_labels': [],
                'masked_lm_label_ids': [],
                'info': None
            }
            instances.append(instance)
        return instances

    def create_salient_mask_instances(self,
                                      context: List[str],
                                      context_mentions: List[Tuple[int, int]],
                                      example: Example,
                                      additional_rows: List[List[Any]]):
        mtl = TableBertConfig.MAX_TARGET_LEN
        instances = []
        if len(context_mentions) <= 0:
            return instances
        table = Table('fake_table', example.header)

        # mask all mentions
        assert len(context) <= mtl - 2, 'context is truncated as target, so masked mentions might be out of bound'
        masked_context = copy.deepcopy(context)
        masked_lm_positions: List[int] = []
        masked_lm_labels: List[str] = []
        for (start, end), _ in context_mentions:
            for i in range(start, end):
                masked_lm_positions.append(i + 1)  # add one to acount for the cls token
                masked_lm_labels.append(context[i])
                if i == start:
                    masked_context[start] = self.config.mask_token
                else:
                    masked_context[i] = None

        # single mask
        masked_context = [t for t in masked_context if t is not None]
        masked_inst = self.get_input(masked_context, table, additional_rows)

        # use raw token sequence as target
        target = [self.config.cls_token] + context[:mtl - 2] + [self.config.sep_token]
        masked_lm_positions = masked_lm_positions
        masked_lm_labels = masked_lm_labels
        instance = {
            'tokens': masked_inst['tokens'],
            'token_ids': self.tokenizer.convert_tokens_to_ids(masked_inst['tokens']),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': masked_inst['segment_a_length'],
            'masked_lm_positions': masked_lm_positions,
            'masked_lm_labels': masked_lm_labels,
            'masked_lm_label_ids': self.tokenizer.convert_tokens_to_ids(masked_lm_labels),
            'info': None
        }
        instances.append(instance)
        return instances

    def create_mention_table_instances(self, context, example: Example, additional_rows: List[List[Any]], dedup: bool = False):
        mtl = TableBertConfig.MAX_TARGET_LEN
        instances = []
        column_data_used = example.column_data_used
        if len(column_data_used) <= 0:
            return instances
        table = Table('fake_table', example.header)
        for row_idx, col_idx in sample(column_data_used, min(len(column_data_used), self.config.max_num_mention_per_example)):
            if row_idx >= len(additional_rows):
                continue
            if col_idx >= len(additional_rows[row_idx]):
                continue
            value = additional_rows[row_idx][col_idx]
            additional_rows[row_idx][col_idx] = [self.config.mask_token]
            if dedup:
                others: List[Tuple[int, int]] = []
                for ri in range(len(additional_rows)):
                    for ci in range(len(additional_rows[ri])):
                        if '@'.join(value) == '@'.join(additional_rows[ri][ci]):
                            additional_rows[ri][ci] = [self.config.mask_token]
                            others.append((ri, ci))
                tablededup2count[len(others)] += 1
            instance = self.get_input(context, table, additional_rows)
            # recover it
            additional_rows[row_idx][col_idx] = value
            if dedup:
                for ri, ci in others:
                    additional_rows[ri][ci] = value
            target = [self.config.cls_token] + value[:mtl - 2] + [self.config.sep_token]
            instance = {
                'tokens': instance['tokens'],
                'token_ids': self.tokenizer.convert_tokens_to_ids(instance['tokens']),
                'target_tokens': target,
                'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
                'segment_a_length': instance['segment_a_length'],
                'masked_lm_positions': [],
                'masked_lm_labels': [],
                'masked_lm_label_ids': [],
                'info': None
            }
            instances.append(instance)
        return instances

    def _add_single_head(self, raw_tokens: List[str], toadd: List[str], target: List[str], seq_a_len: int):
        need_to_remove = len(raw_tokens) + len(toadd) + 1 - TableBertConfig.MAX_SOURCE_LEN  # sep token
        if need_to_remove > 0:  # overflow
            assert len(raw_tokens) >= need_to_remove + 2, 'the raw input is too short to be removed'  # sep and cls token
            tokens = raw_tokens[:-(need_to_remove + 1)]  # also remove the last sep token
            if tokens[-1] != self.config.sep_token:
                tokens.append(self.config.sep_token)
            tokens = tokens + toadd + [self.config.sep_token]
        else:
            tokens = raw_tokens + toadd + [self.config.sep_token]
        instance = {
            'tokens': tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }
        return instance

    def create_seq2seq_instances(self, context, header: List[Column]):
        mtl = TableBertConfig.MAX_TARGET_LEN
        instances = []
        seq2seqf = self.config.seq2seq_format
        for hi, single_head in enumerate(header):
            if not single_head.used and not single_head.value_used:
                continue
            if len(header) > 1:  # has remain headers
                remain_table = Table('fake_table', header[:hi] + header[hi + 1:])
                input_instance = self.get_input(context, remain_table)
                raw_tokens = input_instance['tokens']
                seq_a_len = input_instance['segment_a_length']
            else:
                raw_tokens = [self.config.cls_token] + context + [self.config.sep_token]
                seq_a_len = len(context) + 2
            if 'single-c2v' in seq2seqf and single_head.value_used:
                cell_value = single_head.sample_value_tokens[:self.config.max_cell_len]
                target_tokens = [self.config.cls_token] + cell_value[:mtl - 2] + [self.config.sep_token]
                ctokens = self.get_cell_input(single_head, cell_value, cell_input_template=['column', '|', 'type', '|'])[0]
                instances.append(self._add_single_head(raw_tokens, ctokens, target_tokens, seq_a_len))
            if 'single-v2c' in seq2seqf and (single_head.value_used or single_head.used):
                cell_value = single_head.sample_value_tokens[:self.config.max_cell_len]
                vtokens = self.get_cell_input(single_head, cell_value, cell_input_template=['|', 'value'])[0]
                ctokens = self.get_cell_input(single_head, cell_value, cell_input_template=['column', '|', 'type'])[0]
                target_tokens = [self.config.cls_token] + ctokens[:mtl - 2] + [self.config.sep_token]
                instances.append(self._add_single_head(raw_tokens, vtokens, target_tokens, seq_a_len))
        return instances

    def create_qa_instances(self, context, header: List[Column], answer: str, additional_rows: List[List[Any]] = []):
        mtl = TableBertConfig.MAX_TARGET_LEN
        # context + table without mask
        table = Table('fake_table', header)
        instance = self.get_input(context, table, additional_rows, shuffle=True)
        tokens = instance['tokens']
        seq_a_len = instance['segment_a_length']
        # answer as target
        target = [self.config.cls_token] + self.tokenizer.tokenize(answer)[:mtl - 2] + [self.config.sep_token]
        instance = {
            'tokens': tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }
        return [instance]

    def create_qa_tapex_instances(self, context, header: List[Column], answer: str, additional_rows: List[List[Any]] = []):
        mtl = TableBertConfig.MAX_TARGET_LEN
        table = Table('fake_table', header)
        instance = self.get_input(context, table, additional_rows)
        tokens = instance['tokens']
        seq_a_len = instance['segment_a_length']
        # answer as target
        target = [self.config.cls_token] + self.tokenizer.tokenize(answer)[:mtl - 2] + [self.config.sep_token]
        instance = {
            'tokens': tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }
        return [instance]

    def create_data2text_instances(self,
                                   context: List[str],
                                   header: List[Column],
                                   metadata: Dict[str, str] = None,
                                   additional_rows: List[List[Any]] = [],
                                   use_metadata_context: bool = False):
        if metadata is not None:  # totto metadata
            metastr: str = ' / '.join([metadata['page_title'], metadata['section_title'], metadata['section_text']]).strip()
            metastr: List[str] = self.tokenizer.tokenize(metastr)
        else:
            metastr: List[str] = []
        mtl = TableBertConfig.MAX_TARGET_LEN
        table = Table('fake_table', header)
        instance = self.get_input(metastr, table, additional_rows)
        tokens = instance['tokens']
        seq_a_len = instance['segment_a_length']
        # context as target
        if use_metadata_context:
            context = self.tokenizer.tokenize(metadata['sentence_annotations'][0]['original_sentence'])
        target = [self.config.cls_token] + context[:mtl - 2] + [self.config.sep_token]
        instance = {
            'tokens': tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }
        return [instance]

    def create_clean_text_instances(self,
                                    context: List[str],
                                    header: List[Column],
                                    metadata: Dict[str, str] = None,
                                    additional_rows: List[List[Any]] = [],
                                    fake: bool = False):
        mtl = TableBertConfig.MAX_TARGET_LEN
        instances = []
        if fake:
            table = Table('fake_table', header)
            instance = self.get_input(context, table, additional_rows)
            target = [self.config.cls_token] + context[:mtl - 2] + [self.config.sep_token]
            tokens = instance['tokens']
            seq_a_len = instance['segment_a_length']
            instance = {
                'tokens': tokens,
                'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
                'target_tokens': target,
                'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
                'segment_a_length': seq_a_len,
                'masked_lm_positions': [],
                'masked_lm_labels': [],
                'masked_lm_label_ids': [],
                'info': None
            }
            instances.append(instance)
            return instances
        for sa in metadata['sentence_annotations']:
            noisy_text: List[str] = self.tokenizer.tokenize(sa['original_sentence'])
            clean_text: List[str] = self.tokenizer.tokenize(sa['final_sentence'])
            table = Table('fake_table', header)
            instance = self.get_input(noisy_text, table, additional_rows)
            target = [self.config.cls_token] + clean_text[:mtl - 2] + [self.config.sep_token]
            tokens = instance['tokens']
            seq_a_len = instance['segment_a_length']
            instance = {
                'tokens': tokens,
                'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
                'target_tokens': target,
                'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
                'segment_a_length': seq_a_len,
                'masked_lm_positions': [],
                'masked_lm_labels': [],
                'masked_lm_label_ids': [],
                'info': None
            }
            instances.append(instance)
        return instances

    def create_bidirection_instances(self,
                                     example: Example,
                                     context: List[str],
                                     header: List[Column],
                                     additional_rows: List[List[Any]] = []):
        mtl = TableBertConfig.MAX_TARGET_LEN
        # table (with highlights) -> text
        table = Table('fake_table', header)
        instance = self.get_input([], table, additional_rows)
        tokens = instance['tokens']
        seq_a_len = instance['segment_a_length']
        target = [self.config.cls_token] + context[:mtl - 2] + [self.config.sep_token]
        instance_table2text = {
            'tokens': tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }

        # text + table (exclude highlights) -> highlights
        target_cells: List[str] = []
        target_cell_ind: int = 1
        _additional_rows = copy.deepcopy(additional_rows)
        for row_idx, col_idx in sorted(example.column_data_used):  # predict cells in normal order
            if row_idx < len(_additional_rows) and col_idx < len(_additional_rows[row_idx]):
                cell: str = self.tokenizer.convert_tokens_to_string(_additional_rows[row_idx][col_idx]).strip()
                assert cell.startswith('*')  # TODO: use argument
                cell = cell.lstrip('*').strip()  # remove the highlight mark
                target_cells.extend(self.tokenizer.tokenize(f'({target_cell_ind}) {cell}'))  # append sentinel tokens and cell tokens
                _additional_rows[row_idx][col_idx] = self.tokenizer.tokenize(f'({target_cell_ind})')
                target_cell_ind += 1

        table = Table('fake_table', header)
        instance = self.get_input(context, table, _additional_rows)
        tokens = instance['tokens']
        seq_a_len = instance['segment_a_length']
        target = [self.config.cls_token] + target_cells[:mtl - 2] + [self.config.sep_token]
        instance_text2table = {
            'tokens': tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }
        return [instance_table2text, instance_text2table]

    def create_qa_allrow_instances(self, context, header: List[Column], answer: str, additional_rows: List[List[Any]] = []):
        mtl = TableBertConfig.MAX_TARGET_LEN
        # context + table without mask
        table = Table('fake_table', header)
        instance = self.get_input(context, table, additional_rows)
        tokens = instance['tokens']
        seq_a_len = instance['segment_a_length']
        # answer as target
        target = [self.config.cls_token] + self.tokenizer.tokenize(answer)[:mtl - 2] + [self.config.sep_token]
        instance = {
            'tokens': tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }
        return [instance]

    def create_sql2nl_instances(self, metadata: Dict):
        msl = TableBertConfig.MAX_SOURCE_LEN
        mtl = TableBertConfig.MAX_TARGET_LEN
        sql: List[str] = self.tokenizer.tokenize(metadata['sql'].strip())
        nl: List[str] = self.tokenizer.tokenize(metadata['nl'].strip())
        source = [self.config.cls_token] + sql[:msl - 2] + [self.config.sep_token]
        target = [self.config.cls_token] + nl[:mtl - 2] + [self.config.sep_token]
        instance = {
            'tokens': source,
            'token_ids': self.tokenizer.convert_tokens_to_ids(source),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': len(source),
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }
        return [instance]

    def create_sqltable2nl_instances(self, metadata: Dict, header: List[Column], additional_rows: List[List[Any]] = []):
        mtl = TableBertConfig.MAX_TARGET_LEN
        sql: List[str] = self.tokenizer.tokenize(metadata['sql'].strip())
        nl: List[str] = self.tokenizer.tokenize(metadata['nl'].strip())
        table = Table('fake_table', header)
        instance = self.get_input(sql, table, additional_rows)
        tokens = instance['tokens']
        seq_a_len = instance['segment_a_length']
        target = [self.config.cls_token] + nl[:mtl - 2] + [self.config.sep_token]
        instance = {
            'tokens': tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }
        return [instance]

    def create_sql_instances(self, context, header: List[Column], sql: str):
        mtl = TableBertConfig.MAX_TARGET_LEN
        # context + table without mask
        table = Table('fake_table', header)
        instance = self.get_input(context, table)
        tokens = instance['tokens']
        seq_a_len = instance['segment_a_length']
        # sql as target
        target = [self.config.cls_token] + self.tokenizer.tokenize(sql)[:mtl - 2] + [self.config.sep_token]
        instance = {
            'tokens': tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'target_tokens': target,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }
        return [instance]

    def create_sa_gen_instance(self, context, header: List[Column]):
        mtl = TableBertConfig.MAX_TARGET_LEN
        max_schema_token_length = mtl - 2  # cls and sep token
        cds = self.config.multi_decode_sep_tokens
        schema_tokens = self.concate_cells(
            header, [[] for _ in range(len(header))],
            table_tokens_start_idx=0, trim_long_table=0, max_table_token_length=max_schema_token_length,
            cell_input_template=['column', '|', 'type'], column_delimiters=cds)[0]
        schema_tokens = [self.config.cls_token] + schema_tokens + [self.config.sep_token]

        context_tokens = [self.config.cls_token] + context + [self.config.sep_token]
        seq_a_len = len(context_tokens)

        return {
            'tokens': context_tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(context_tokens),
            'target_tokens': schema_tokens,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(schema_tokens),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }

    def create_cf_gen_instance(self, context, header: List[Column]):
        mtl = TableBertConfig.MAX_TARGET_LEN
        # get target using object columns (odd)
        target_cells: List[List] = []
        for col_idx, cell in enumerate(header):
            if col_idx % 2 == 0:
                continue
            target_cells.append(cell.sample_value_tokens[:self.config.max_cell_len])
            cell.sample_value_tokens = []  # skip it in the source

        table = Table('fake_table', header)
        input_instance = self.get_input(context, table)
        source_tokens = input_instance['tokens']
        seq_a_len = input_instance['segment_a_length']

        num_columns_used = len(input_instance['column_spans']) // 2
        target_tokens = [self.config.cls_token]
        target_cells = target_cells[:num_columns_used]
        for i, tc in enumerate(target_cells):
            target_tokens.extend(tc)
            if i < len(target_cells) - 1:
                target_tokens.extend(self.config.multi_decode_sep_tokens)
        target_tokens.append(self.config.sep_token)
        target_tokens = target_tokens[:mtl]
        if len(target_tokens) > 0 and target_tokens[-1] != self.config.sep_token:
            target_tokens[-1] = self.config.sep_token

        return {
            'tokens': source_tokens,
            'token_ids': self.tokenizer.convert_tokens_to_ids(source_tokens),
            'target_tokens': target_tokens,
            'target_token_ids': self.tokenizer.convert_tokens_to_ids(target_tokens),
            'segment_a_length': seq_a_len,
            'masked_lm_positions': [],
            'masked_lm_labels': [],
            'masked_lm_label_ids': [],
            'info': None
        }

    def create_sa_mask_instance(self, context, header: List[Column]):
        table = Table('fake_table', header)
        input_instance = self.get_input(context, table, cell_input_template=['column', '|', 'type'])
        column_spans = input_instance['column_spans']

        assert self.config.masked_column_prob == 1.0, \
            'generating schema augmentation examples requires masked_column_prob=1.0'
        column_candidate_indices = [
            (
                list(range(*span['column_name']) if 'column_name' in span else []) +
                list(range(*span['type']) if 'type' in span else [])
            )
            for col_id, span in enumerate(column_spans)]
        context_candidate_indices = []

        masked_sequence, masked_lm_positions, masked_lm_labels, info = self.create_masked_lm_predictions(
            input_instance['tokens'], context_candidate_indices, column_candidate_indices)
        info['num_columns'] = len(header)

        instance = {
            "tokens": masked_sequence,
            "token_ids": self.tokenizer.convert_tokens_to_ids(masked_sequence),
            "segment_a_length": input_instance['segment_a_length'],
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels,
            "masked_lm_label_ids": self.tokenizer.convert_tokens_to_ids(masked_lm_labels),
            "info": info
        }

        return instance

    def _recover_data_from_header(self, header: List[Column], interval: int):
        real_header = header[:interval]
        rows: List[List] = []
        for i, h in enumerate(header):
            if i % interval == 0:
                rows.append([])
            rows[-1].append(h.sample_value_tokens)
        return real_header, rows

    def create_cf_mask_instance_columnwise(self, context, header: List[Column]):
        header, rows = self._recover_data_from_header(header, interval=2)
        input_instance = self.get_row_input(context, header, [], rows)
        column_spans = input_instance['column_spans']

        assert self.config.mask_value and self.config.masked_column_prob == 1.0, \
            'generating cell filling examples requires mask_value=True and masked_column_prob=1.0'
        # assume: the second column (later half) is value, and the first of it is a header
        column_candidate_indices = [
            list(range(*span['value']) if ('value' in span and col_id > len(column_spans) // 2) else [])
            for col_id, span in enumerate(column_spans)]
        context_candidate_indices = []
        masked_sequence, masked_lm_positions, masked_lm_labels, info = self.create_masked_lm_predictions(
            input_instance['tokens'], context_candidate_indices, column_candidate_indices)
        info['num_columns'] = len(header)

        instance = {
            "tokens": masked_sequence,
            "token_ids": self.tokenizer.convert_tokens_to_ids(masked_sequence),
            "segment_a_length": input_instance['segment_a_length'],
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels,
            "masked_lm_label_ids": self.tokenizer.convert_tokens_to_ids(masked_lm_labels),
            "info": info
        }

        return instance

    def create_cf_mask_instance(self, context, header: List[Column]):
        if self.config.column_wise:
            header, rows = self._recover_data_from_header(header, interval=2)
            input_instance = self.get_row_input(context, header, [], rows)
            column_spans = input_instance['column_spans']
            assert self.config.mask_value and self.config.masked_column_prob == 1.0, \
                'generating cell filling examples requires mask_value=True and masked_column_prob=1.0'
            # assume: the second column (later half) is value, and the first of it is a header
            column_candidate_indices = [
                list(range(*span['value']) if ('value' in span and col_id > len(column_spans) // 2) else [])
                for col_id, span in enumerate(column_spans)]
        else:
            table = Table('fake_table', header)
            input_instance = self.get_input(context, table)
            column_spans = input_instance['column_spans']
            assert self.config.mask_value and self.config.masked_column_prob == 1.0, \
                'generating cell filling examples requires mask_value=True and masked_column_prob=1.0'
            # assume: the odd ones are object columns, the even ones are subject columns
            column_candidate_indices = [
                list(range(*span['value']) if ('value' in span and col_id % 2 == 1) else [])
                for col_id, span in enumerate(column_spans)]

        context_candidate_indices = []
        masked_sequence, masked_lm_positions, masked_lm_labels, info = self.create_masked_lm_predictions(
            input_instance['tokens'], context_candidate_indices, column_candidate_indices)
        info['num_columns'] = len(header)

        instance = {
            "tokens": masked_sequence,
            "token_ids": self.tokenizer.convert_tokens_to_ids(masked_sequence),
            "segment_a_length": input_instance['segment_a_length'],
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels,
            "masked_lm_label_ids": self.tokenizer.convert_tokens_to_ids(masked_lm_labels),
            "info": info
        }
        return instance

    def create_pretraining_instance(self,
                                    context,
                                    header,
                                    additional_rows: List[List[Any]] = [],
                                    single_mask: bool = False):
        table = Table('fake_table', header)
        input_instance = self.get_input(context, table, additional_rows)  # core format function
        column_spans = input_instance['column_spans']

        column_candidate_indices = [
            (
                list(range(*span['column_name']) if 'column_name' in span else []) +
                list(range(*span['type']) if 'type' in span else []) +
                list(range(*span['index']) if 'index' in span else []) +
                (
                    span['other_tokens']
                    if random() < 0.01 and 'other_tokens' in span
                    else []
                )
            )
            for col_id, span
            in enumerate(column_spans)  # only mask the first row and used columns if specified
            if not self.config.mask_used_column_prob or header[col_id].used
        ]
        if self.config.mask_value:
            column_candidate_indices_value = [
                list(range(*span['value']) if 'value' in span else [])
                for col_id, span
                in enumerate(column_spans)  # only mask the first row and used columns if specified
                if not self.config.mask_used_column_prob or header[col_id].used
            ]
            assert len(column_candidate_indices_value) == len(column_candidate_indices), 'candidate indices length inconsistent'
            if self.config.mask_value_column_separate:
                column_candidate_indices = column_candidate_indices + column_candidate_indices_value
            else:
                column_candidate_indices = [ci + civ for ci, civ in zip(column_candidate_indices, column_candidate_indices_value)]

        masked_column_prob = None
        if self.config.mask_used_column_prob:
            masked_column_prob = min(self.config.masked_column_prob * len(column_spans) / (len(column_candidate_indices) or 1), 1.0)

        context_candidate_indices = (
            list(range(*input_instance['context_span']))[1:]
            if self.config.context_first else
            list(range(*input_instance['context_span']))[:-1]
        )

        raw_tokens = input_instance['tokens']
        masked_sequence, masked_lm_positions, masked_lm_labels, info = self.create_masked_lm_predictions(
            raw_tokens, context_candidate_indices, column_candidate_indices, masked_column_prob=masked_column_prob
        )

        if single_mask:  # only keep a single mask for consecutive mask tokens
            _masked_sequence: List[str] = []
            for i in range(len(masked_sequence)):
                if i > 0 and masked_sequence[i - 1] == masked_sequence[i] == self.config.mask_token:
                    continue
                _masked_sequence.append(masked_sequence[i])
            masked_sequence = _masked_sequence

        info['num_columns'] = len(header)

        instance = {
            "tokens": masked_sequence,
            "token_ids": self.tokenizer.convert_tokens_to_ids(masked_sequence),
            "raw_tokens": raw_tokens,
            "raw_token_ids": self.tokenizer.convert_tokens_to_ids(raw_tokens),
            "column_token_to_column_id": input_instance["column_token_to_column_id"],
            "segment_a_length": input_instance['segment_a_length'],
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels,
            "masked_lm_label_ids": self.tokenizer.convert_tokens_to_ids(masked_lm_labels),
            "info": info
        }

        return instance

    def create_masked_lm_predictions(self,
                                     tokens,
                                     context_indices,
                                     column_indices,
                                     masked_column_prob=None):
        table_mask_strategy = self.config.table_mask_strategy
        masked_column_prob = masked_column_prob or self.config.masked_column_prob

        info = dict()
        info['num_maskable_column_tokens'] = sum(len(token_ids) for token_ids in column_indices)

        if table_mask_strategy == 'column_token':
            column_indices = [i for l in column_indices for i in l]
            num_column_tokens_to_mask = min(self.config.max_predictions_per_seq,
                                            max(2, int(len(column_indices) * masked_column_prob)))
            shuffle(column_indices)
            masked_column_token_indices = sorted(sample(column_indices, num_column_tokens_to_mask))
        elif table_mask_strategy == 'column':
            num_maskable_columns = len(column_indices)
            num_column_to_mask = max(1, ceil(num_maskable_columns * masked_column_prob))
            columns_to_mask = sorted(sample(list(range(num_maskable_columns)), num_column_to_mask))
            shuffle(columns_to_mask)
            num_column_tokens_to_mask = sum(len(column_indices[i]) for i in columns_to_mask)
            masked_column_token_indices = [idx for col in columns_to_mask for idx in column_indices[col]]

            info['num_masked_columns'] = num_column_to_mask
        else:
            raise RuntimeError('unknown mode!')

        max_context_token_to_mask = self.config.max_predictions_per_seq - num_column_tokens_to_mask
        num_context_tokens_to_mask = min(max_context_token_to_mask, min(
            len(context_indices), max(1, int(len(context_indices) * self.config.masked_context_prob))))
        if num_context_tokens_to_mask == max_context_token_to_mask:
            trim_count['mask_no_enough'] += 1

        if num_context_tokens_to_mask > 0:
            # if num_context_tokens_to_mask < 0 or num_context_tokens_to_mask > len(context_indices):
            #     for col_id in columns_to_mask:
            #         print([tokens[i] for i in column_indices[col_id]])
            #     print(num_context_tokens_to_mask, num_column_tokens_to_mask)
            shuffle(context_indices)
            masked_context_token_indices = sorted(sample(context_indices, num_context_tokens_to_mask))
            masked_indices = sorted(masked_context_token_indices + masked_column_token_indices)
        else:
            masked_indices = masked_column_token_indices
        assert len(set(masked_column_token_indices)) == len(masked_column_token_indices), 'duplicate indicies'

        masked_token_labels = []
        masked_indices_rm_oob = []

        tokens_with_mask = copy.deepcopy(tokens)
        for index in masked_indices:
            if self.config.not_strict_mask and index >= len(tokens_with_mask):
                continue
            if not self.config.use_electra:  # BERT style masking
                # 80% of the time, replace with mask
                if random() < 0.8:
                    masked_token = self.config.mask_token
                else:
                    # 10% of the time, keep original
                    if random() < 0.5:
                        masked_token = tokens_with_mask[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = choice(self.vocab_list)
            else:  # ELECTRA style masking
                if random() < 0.85:  # 85% of the time, replace with mask
                    masked_token = self.config.mask_token
                else:  # 15% of the time, keep original
                    masked_token = tokens_with_mask[index]
            masked_token_labels.append(tokens_with_mask[index])
            masked_indices_rm_oob.append(index)
            # Once we've saved the true label for that token, we can overwrite it with the masked version
            tokens_with_mask[index] = masked_token

        info.update({
            'num_column_tokens_to_mask': num_column_tokens_to_mask,
            'num_context_tokens_to_mask': num_context_tokens_to_mask,
        })

        return tokens_with_mask, masked_indices_rm_oob, masked_token_labels, info

    def remove_unecessary_instance_entries(self, instance: Dict):
        del instance['tokens']
        del instance['masked_lm_labels']
        del instance['info']


if __name__ == '__main__':
    config = TableBertConfig()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_formatter = VanillaTableBertInputFormatter(config, tokenizer)

    header = []
    for i in range(1000):
        header.append(
            Column(
                name='test',
                type='text',
                name_tokens=['test'] * 3,
                sample_value='ha ha ha yay',
                sample_value_tokens=['ha', 'ha', 'ha', 'yay']
            )
        )

    print(
        input_formatter.get_row_input(
            context='12 213 5 345 23 234'.split(),
            header=header,
            row_data=[col.sample_value_tokens for col in header]
        )
    )
