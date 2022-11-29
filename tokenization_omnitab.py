from typing import Union, Optional, List, Dict, Tuple, Any, Set
import copy
import random
import numpy as np
import torch
from transformers import TapexTokenizer, BartTokenizerFast
from transformers.utils import logging
from transformers.file_utils import PaddingStrategy, TensorType
from transformers.tokenization_utils_base import TruncationStrategy, TextInput, BatchEncoding
import pandas as pd


logger = logging.get_logger(__name__)


class IndexedRowTableLinearizeOffset:
    def process_table(self, table_content: Dict, row_sep: str = ' ') -> Tuple[str, Dict]:
        assert 'header' in table_content and 'rows' in table_content, 'incorret format'
        offset = 0
        table_offset = {'header': [], 'rows': []}
        # process header
        eles, inds = self.process_header(table_content['header'])
        for ele, ind in zip(eles, inds):
            if ind is not None:
                table_offset['header'].append((offset, offset + len(ele)))
            offset += len(ele)
        if len(eles):  # add sep if there is a header
            eles.append(row_sep)
            inds.append(None)
            offset += len(row_sep)
        # process rows
        for i, row_example in enumerate(table_content['rows']):
            _eles, _inds = self.process_row(row_example, row_index=i)
            table_offset['rows'].append([])
            for ele, ind in zip(_eles, _inds):
                if ind is not None:
                    table_offset['rows'][-1].append((offset, offset + len(ele)))
                offset += len(ele)
            eles.extend(_eles)
            inds.extend(_inds)
            if i < len(table_content['rows']) - 1:
                eles.append(row_sep)
                inds.append(None)
                offset += len(row_sep)
        assert len(eles) == len(inds), 'inconsistent length'
        linearized_table = ''.join(eles)
        return linearized_table, table_offset

    def process_header(self, headers: List) -> Tuple[List[str], List[int]]:
        if len(headers) <= 0:
            return [], []
        elements = ['col : '] + sum([[h, ' | '] if i < len(headers) - 1 else [h] for i, h in enumerate(headers)], [])
        indices = [None] + sum([[i, None] if i < len(headers) - 1 else [i] for i, _ in enumerate(headers)], [])
        return elements, indices

    def process_row(self, row: List, row_index: int) -> Tuple[List[str], List[Tuple[int, int]]]:
        elements = [f'row {row_index + 1} : ']  # zero-based
        indices = [None]
        for col_index, cell_value in enumerate(row):
            if isinstance(cell_value, str):
                elements.append(cell_value)
                indices.append((row_index, col_index))
            else:
                elements.append(str(cell_value))
                indices.append((row_index, col_index))
            if col_index < len(row) - 1:
                elements.append(' | ')
                indices.append(None)
        return elements, indices


class OmnitabTokenizer(TapexTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fast_tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')  # used to get offset
        self.fast_tokenize = lambda inputs: fast_tokenizer(inputs, add_special_tokens=False, return_offsets_mapping=True)
        self.table_linearize_offset = IndexedRowTableLinearizeOffset()  # track the linearization

    def truncate(
        self,
        query: Union[str, List[str]],
        table: Union["pd.DataFrame", List["pd.DataFrame"]],
        max_query_length: Union[int, List[int]],
        max_length: Union[int, List[int]]
    ):
        # first truncate query based on max_query_length
        query, query_len = self.truncate_query(query, max_query_length=max_query_length)

        # then truncate table based on query and max_length
        if type(max_length) is list and type(query_len) is list:
            assert len(max_length) == len(query_len), 'inconsistent length'
            max_table_length = [ml - ql for ml, ql in zip(max_length, query_len)]
        elif type(max_length) is list and type(query_len) is not list:
            max_table_length = [ml - query_len for ml in max_length]
        elif type(max_length) is not list and type(query_len) is list:
            max_table_length = [max_length - ql for ql in query_len]
        else:
            max_table_length = max_length - query_len
        table = self.truncate_table(table, max_table_length=max_table_length)

        return query, table

    def truncate_table(
        self,
        table: Union["pd.DataFrame", List["pd.DataFrame"]],
        max_table_length: Union[int, List[int]],
    ):
        is_single = type(table) is not list
        tables = [table] if is_single else table
        max_table_lengths = [max_table_length] * len(tables) if type(max_table_length) is not list else max_table_length
        assert len(tables) == len(max_table_lengths), 'inconsistent length'

        truncated_tables: List["pd.DataFrame"] = []
        for tab, mtl in zip(tables, max_table_lengths):
            assert mtl > 0, 'truncation length should be positive'
            tab_content = {'header': list(tab.columns), 'rows': [list(row.values) for i, row in tab.iterrows()]}
            # truncate cells
            self.truncate_table_header_cells(tab_content)
            self.truncate_table_cells(tab_content, question=None, answer=None)
            # remove rows
            self.remove_rows(tab_content, max_length=mtl)
            truncated_tables.append(pd.DataFrame.from_records(tab_content['rows'], columns=tab_content['header']))

        return (truncated_tables[0] if is_single else truncated_tables)

    def truncate_query(
        self,
        query: Union[str, List[str]],
        max_query_length: Union[int, List[int]],
    ):
        is_single = type(query) is not list
        queries = [query] if is_single else query
        max_query_lengths = [max_query_length] * len(queries) if type(max_query_length) is not list else max_query_length
        assert len(queries) == len(max_query_lengths), 'inconsistent length'

        # encode using fast tokenizer
        encoded = self.fast_tokenize(queries)

        # truncate
        truncated_queries: List[str] = []
        len_truncated_queries: List[int] = []
        for qry, mql, ids, offsets in zip(queries, max_query_lengths, encoded['input_ids'], encoded['offset_mapping']):
            assert mql > 0, 'truncation length should be positive'
            if len(ids) > mql:  # need truncation
                offset = offsets[mql - 1][-1]
                truncated_queries.append(qry[:offset])
                len_truncated_queries.append(mql)
            else:
                truncated_queries.append(qry)
                len_truncated_queries.append(len(ids))

        return (truncated_queries[0] if is_single else truncated_queries), (len_truncated_queries[0] if is_single else len_truncated_queries)

    def truncate_table_header_cells(self, table_content: Dict):
        for i, h in enumerate(table_content['header']):
            truncated_h = self.truncate_cell(h) or h
            table_content['header'][i] = truncated_h

    def truncate_cell(self, cell_value: Union[int, float, str]):
        if isinstance(cell_value, int) or isinstance(cell_value, float):
            return cell_value
        if cell_value.strip() != '':
            # make sure there is at least one leading space because cell values always have a leading space in the final linearized format
            add_prefix_space = len(cell_value) and not cell_value[0].isspace()
            try_tokens = self.tokenize(cell_value, add_prefix_space=True) if add_prefix_space else self.tokenize(cell_value)
            if len(try_tokens) >= self.max_cell_length:
                retain_tokens = try_tokens[:self.max_cell_length]
                retain_cell_value = self.convert_tokens_to_string(retain_tokens)
                retain_cell_value = retain_cell_value[1:] if add_prefix_space else retain_cell_value
                return retain_cell_value
            else:
                return None
        else:
            return cell_value

    def remove_rows(
        self,
        table_content: Dict,
        max_length: int
    ):
        # TODO: in the original OmniTab code truncation is performed at cell-level instead of row-level
        remain_token_len = max_length

        # first examine header
        value_string = self.table_linearize.process_header(table_content['header'])
        value_token_len = len(self.tokenize(value_string))
        if value_token_len > remain_token_len:
            raise ValueError('since the header is too long, the entire table is truncated')
        remain_token_len -= value_token_len

        # then examine rows
        maximum_keep_rows = 0
        for ind, row_example in enumerate(table_content['rows']):
            value_string = self.table_linearize.process_row(row_example, ind + 1)
            value_token_len = len(self.tokenize(value_string))
            if value_token_len > remain_token_len:
                break
            remain_token_len -= value_token_len
            maximum_keep_rows += 1
        del table_content['rows'][maximum_keep_rows:]

    def replace_special_tokens(
        self,
        input_ids: Union[List, torch.Tensor, np.ndarray],
        special_tokens_mask: Union[List, torch.Tensor, np.ndarray],
        replace_with: int = None,
    ):
        is_list = False
        if type(input_ids) is list:
            is_list = True
            input_ids = torch.tensor(input_ids)
            special_tokens_mask = torch.tensor(special_tokens_mask)
        input_ids[special_tokens_mask.eq(1)] = replace_with or self.pad_token_id
        if is_list:
            input_ids = input_ids.tolist()
        return input_ids

    def _encode_plus(
        self,
        table: "pd.DataFrame",
        query: Optional[TextInput] = None,
        answer: Optional[str] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        query_mask: List[Tuple[int, int]] = None,
        table_mask: "pd.DataFrame" = None,
        mask_kwargs: Dict[str, Any] = None,
        **kwargs
    ) -> BatchEncoding:
        if query_mask is None and table_mask is None:  # call the original function
            return super()._encode_plus(
                table=table,
                query=query,
                answer=answer,
                add_special_tokens=add_special_tokens,
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs)

        text, mask_spans = self.prepare_table_query_with_mask(query, table, query_mask, table_mask)

        if self.do_lower_case:
            _text = text.lower()
            assert len(text) == len(_text), 'offsets in masks might be invalid'
            text = _text

        # tokenize
        encoded = self.fast_tokenize(text)
        token_ids = encoded['input_ids']
        token_offsets = encoded['offset_mapping']

        # convert char-based spans to token-based spans
        mask_spans = self.char2token(token_offsets, mask_spans)
        valid_mask_spans = [span for span in mask_spans if span is not None]
        if len(valid_mask_spans) != len(mask_spans):
            logger.warning('there are masked spans not overlopping with any tokens')

        label_token_ids, masked_token_ids, replace_unmasked_tokens_with_pad = self.perform_masking(token_ids, valid_mask_spans, **(mask_kwargs or {}))

        source = self.prepare_for_model(
            ids=masked_token_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )
        target = self.prepare_for_model(
            ids=token_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )
        label = self.prepare_for_model(
            ids=label_token_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=True,
            return_length=return_length,
            verbose=verbose,
        )
        if replace_unmasked_tokens_with_pad:  # convert added special tokens to pad
            label['input_ids'] = self.replace_special_tokens(label['input_ids'], label['special_tokens_mask'], replace_with=self.pad_token_id)
        source['target_input_ids'] = target['input_ids']
        source['label_input_ids'] = label['input_ids']
        return source

    def _batch_prepare_for_model(
        self,
        table: Union["pd.DataFrame", List["pd.DataFrame"]],
        query: Optional[Union[TextInput, List[TextInput]]] = None,
        answer: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        query_mask: Union[List[Tuple], List[List[Tuple]]] = None,
        table_mask: Union["pd.DataFrame", List["pd.DataFrame"]] = None,
        mask_kwargs: Dict[str, Any] = None,
        **kwargs
    ) -> BatchEncoding:
        if query_mask is None and table_mask is None:  # call the original function
            return super()._batch_prepare_for_model(
                table=table,
                query=query,
                answer=answer,
                add_special_tokens=add_special_tokens,
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                verbose=verbose)

        batch_outputs = {}
        if answer is None:
            answer = [None] * len(table)
        if table_mask is None:
            table_mask = [None] * len(table)
        if query_mask is None:
            query_mask = [None] * len(table)
        for _table, _query, _answer, _table_mask, _query_mask in zip(table, query, answer, table_mask, query_mask):
            outputs = self._encode_plus(
                table=_table,
                query=_query,
                answer=_answer,
                add_special_tokens=add_special_tokens,
                padding_strategy=PaddingStrategy.DO_NOT_PAD,
                truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,
                return_tensors=None,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=False,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                verbose=verbose,
                query_mask=_query_mask,
                table_mask=_table_mask,
                mask_kwargs=mask_kwargs,
                **kwargs,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        has_target = False
        if 'target_input_ids' in batch_outputs:
            has_target = True
            target = batch_outputs['target_input_ids']
            label = batch_outputs['label_input_ids']
            del batch_outputs['target_input_ids']
            del batch_outputs['label_input_ids']
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
        if has_target:
            target_batch_outputs = self.pad(
                {'input_ids': target},
                padding=padding_strategy.value,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            label_batch_outputs = self.pad(
                {'input_ids': label},
                padding=padding_strategy.value,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            for k, v in target_batch_outputs.items():
                batch_outputs[f'target_{k}'] = v
            for k, v in label_batch_outputs.items():
                batch_outputs[f'label_{k}'] = v

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    def _batch_encode_plus(
        self,
        table: Union["pd.DataFrame", List["pd.DataFrame"]],
        query: Optional[List[TextInput]] = None,
        answer: Optional[List[str]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        if isinstance(table, pd.DataFrame) and isinstance(query, (list, tuple)):
            # single table, many queries case
            # duplicate table for every query
            table = [table] * len(query)
        if isinstance(table, (list, tuple)) and isinstance(query, str):
            # many tables, single query case
            # duplicate query for every table
            query = [query] * len(table)

        batch_outputs = self._batch_prepare_for_model(
            table=table,
            query=query,
            answer=answer,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
            **kwargs
        )

        return BatchEncoding(batch_outputs)

    def prepare_table_query_with_mask(
        self,
        query: TextInput,
        table: pd.DataFrame,
        query_mask: List[Tuple[int, int]] = None,
        table_mask: pd.DataFrame = None,
    ):
        # linearize table
        tab = {'header': list(table.columns), 'rows': [list(row.values) for i, row in table.iterrows()]}
        linearized_table, tab_offset = self.table_linearize_offset.process_table(tab)

        # concat query and table and shift table_offset
        separator = ' ' if query and linearized_table else ''
        concat_input = query + separator + linearized_table

        # clean query masks
        concat_mask: List[Tuple[int, int]] = query_mask or []
        concat_mask = self.clean_spans(concat_mask, max_length=len(query))

        # create table masks
        shift = len(query + separator)
        if table_mask is not None:
            tab_mask = {'header': list(table_mask.columns), 'rows': [list(row.values) for i, row in table_mask.iterrows()]}
            assert len(tab_mask['header']) == len(tab_offset['header']), 'inconsistent length'
            for mask, offset in zip(tab_mask['header'], tab_offset['header']):
                if mask:
                    concat_mask.append((shift + offset[0], shift + offset[1]))
            assert len(tab_mask['rows']) == len(tab_offset['rows']), 'inconsistent length'
            for ri in range(len(tab_mask['rows'])):
                for mask, offset in zip(tab_mask['rows'][ri], tab_offset['rows'][ri]):
                    if mask:
                        concat_mask.append((shift + offset[0], shift + offset[1]))

        # clean all masks
        concat_mask = self.clean_spans(concat_mask, max_length=len(concat_input))

        return concat_input, concat_mask

    def clean_spans(
        self,
        spans: List[Tuple[int, int]],
        max_length: int = None
    ):
        spans = sorted(spans)
        prev = -1
        valid_indices: List[int] = []
        for i, (start, end) in enumerate(spans):
            valid = True
            if start >= end:  # empty span
                valid = False
            elif start < prev:  # overlap with previous span
                valid = False
            elif max_length and end > max_length:  # overflow
                valid = False
            if valid:
                valid_indices.append(i)
                prev = end
        spans = [spans[i] for i in valid_indices]
        return spans

    @classmethod
    def span_overlap(cls, s1: int, e1: int, s2: int, e2: int):  # inclusive, exclusive
        # 0 -> overlap
        # 1 -> the first passed the second
        # -1 -> the second passed the first
        if s1 >= e2:
            return 1
        if e1 <= s2:
            return -1
        return 0

    @classmethod
    def char2token(
        cls,
        tokens: List[Tuple[int, int]],
        mentions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Convert mentions from char index to token index.
        Assume tokens and mentions contain sorted valid spans without overlap.
        """
        mentions_in_tok: List[Tuple[int, int]] = [None] * len(mentions)  # none indicates the mention doesn't overlap with any token
        midx = tidx = 0
        while tidx < len(tokens) and midx < len(mentions):
            t_start, t_end = tokens[tidx]
            while midx < len(mentions) and cls.span_overlap(t_start, t_end, *mentions[midx]) == 1:  # skip mentions before the token
                midx += 1
            if midx >= len(mentions):
                break
            m_start, m_end = mentions[midx]
            status = cls.span_overlap(t_start, t_end, m_start, m_end)
            if status == 0:  # overlap
                if mentions_in_tok[midx] is None:  # start a new mention
                    mentions_in_tok[midx] = (tidx, tidx + 1)
                else:  # extend an old mention
                    mentions_in_tok[midx] = (mentions_in_tok[midx][0], tidx + 1)
                if m_end < t_end:  # skip the current mention because it won't be used in the future
                    midx += 1
                else:  # skip the current token because it won't be used in the future
                    tidx += 1
            elif status == -1:  # the current mention pass the current token
                tidx += 1
        return mentions_in_tok

    def get_mask_token_id(
        self,
        token_id: int,
        mask_strategy: str = 'mask'
    ):
        assert mask_strategy in {'mask', 'bert'}
        if mask_strategy == 'mask':
            return self.mask_token_id
        if mask_strategy == 'bert':
            r = random.random()
            if r < 0.8:  # 80% replace with the mask token
                return self.mask_token_id
            elif r < 0.9:  # 10% keep the original token
                return token_id
            else:  # 10% replace with a random token
                return random.randint(4, 50260)  # don't sample from special tokens

    def perform_masking(
        self,
        token_ids: List[int],
        mask_spans: List[Tuple[int, int]],
        mask_granularity: str = 'span',
        mask_strategy: str = 'mask',
        merge_consecutive_masks: bool = False,
        replace_unmasked_tokens_with_pad: bool = True,
    ) -> Tuple[List[int], List[int]]:
        assert mask_granularity in {'span', 'token'}

        # mask
        masked_token_ids: List[int] = copy.deepcopy(token_ids)
        masked_positions: Set[int] = set()
        if mask_granularity == 'span':  # only mask once for each span
            for start, end in mask_spans:
                masked_token_ids[start] = self.get_mask_token_id(masked_token_ids[start], mask_strategy=mask_strategy)
                masked_positions.add(start)
                for i in range(start + 1, end):
                    masked_token_ids[i] = None
                    masked_positions.add(i)
        elif mask_granularity == 'token':  # mask all tokens in each span
            for start, end in mask_spans:
                for i in range(start, end):
                    masked_token_ids[i] = self.get_mask_token_id(masked_token_ids[i], mask_strategy=mask_strategy)
                    masked_positions.add(i)
        else:
            raise ValueError

        # clean masked token ids
        _masked_token_ids: List[int] = []
        prev_tid = None
        for i, tid in enumerate(masked_token_ids):
            if tid is None:
                continue
            if merge_consecutive_masks and (tid == prev_tid == self.mask_token_id):  # merge consecutive masks
                continue
            _masked_token_ids.append(tid)
            prev_tid = tid
        masked_token_ids = _masked_token_ids

        label_token_ids: List[int] = copy.deepcopy(token_ids)
        if replace_unmasked_tokens_with_pad:  # replace unmasked tokens with the pad token
            for i in range(len(label_token_ids)):
                if i not in masked_positions:
                    label_token_ids[i] = self.pad_token_id

        assert len(masked_token_ids) <= len(token_ids) == len(label_token_ids), 'masking caused an unexpected increasing of #tokens'
        return label_token_ids, masked_token_ids, replace_unmasked_tokens_with_pad
