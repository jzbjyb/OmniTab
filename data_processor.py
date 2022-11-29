from typing import List, Tuple, Dict
import logging
import math
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets, Dataset
from tokenization_omnitab import OmnitabTokenizer


logger = logging.getLogger(__name__)


class OmnitabPretrainDataset:
    def __init__(self, root_dir: str):
        self.nat_file = f'{root_dir}/natural.jsonl'
        self.syn_file = f'{root_dir}/synthetic.jsonl'
        self.sql_file = f'{root_dir}/sql.jsonl'
        self.load_data()

    def load_data(self):
        self.nat_data = load_dataset('json', data_files=self.nat_file)['train']
        self.syn_data = load_dataset('json', data_files=self.syn_file)['train']
        self.sql_data = load_dataset('json', data_files=self.sql_file)['train']


class TableQAProcessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 1024,
        answer_sep: str = ', ',  # answer separator
        question_lower_case: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.answer_sep = answer_sep
        self.question_lower_case = question_lower_case
        pass

    def process(
        self,
        dataset: Dataset,
        is_training: bool = False,
        max_target_length: int = 1024,
        num_proc: int = 1,
        overwrite_cache: bool = False,
    ):
        def _process(examples):
            questions = [question.lower() if self.question_lower_case else question for question in examples['question']]
            tables = examples['table']
            tables = [pd.DataFrame.from_records(table['rows'], columns=table['header']) for table in tables]
            answers = examples['answers']

            # IMPORTANT: we cannot pass by answers during evaluation, answers passed during training are used to
            # truncate large tables in the train set!
            model_inputs = self.tokenizer(
                table=tables,
                query=questions,
                answer=answers if is_training else None,
                max_length=self.max_source_length,
                padding=False,
                truncation=True,
            )

            labels = self.tokenizer(
                answer=[self.answer_sep.join(answer) for answer in answers],
                max_length=max_target_length,
                padding=False,
                truncation=True,
            )

            model_inputs['labels'] = labels['input_ids']

            return model_inputs

        dataset = dataset.map(
            _process,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            load_from_cache_file=not overwrite_cache,
            desc=f'processing table-QA data',
        )
        return dataset


class PretrainProcessor:
    def __init__(
        self,
        table_mask_prob: float = 0.2,  # probablity of masking an element in tables
        context_mask_prob: float = 0.15,  # probability of masking a token in contexts
        max_num_mask_tokens: int = 200,  # max number of masked tokens
        max_source_length: int = 1024,  # max length of source
        max_context_length: int = 128,  # max length of context (in source)
        answer_sep: str = ', '  # answer separator
    ):
        self.table_mask_prob = table_mask_prob
        self.context_mask_prob = context_mask_prob
        self.max_num_mask_tokens = max_num_mask_tokens
        self.max_source_length = max_source_length
        self.max_context_length = max_context_length
        # Since the format used in truncation might not be exactly the same as the final format (due to special tokens and tokenization sensitivity),
        # the final length might be longer than the estimation in truncation. Therefore we keep 5 safe positions to make sure the final output is strictly no longer than max_source_length
        self.num_safe_tokens = 5
        assert self.max_source_length > self.num_safe_tokens and self.max_source_length - self.num_safe_tokens >= self.max_context_length, 'max length is too short'
        self.max_target_length = max_source_length  # source and target are similar for masking
        self.answer_sep = answer_sep
        self.tokenizer: OmnitabTokenizer = OmnitabTokenizer.from_pretrained('neulab/omnitab-large')
        self.add_prefix_space = lambda text: ((' ' + text) if not text[0].isspace() else text, not text[0].isspace())

    def qa(
        self,
        contexts: List[str],
        tables: List[pd.DataFrame],
        answers: List[List[str]],
    ):
        # truncate
        contexts, tables = self.tokenizer.truncate(
            query=contexts,
            table=tables,
            max_query_length=self.max_context_length,
            max_length=self.max_source_length - self.num_safe_tokens)

        # generate source
        outputs = self.tokenizer(
            table=tables,
            query=contexts,
            answer=None,
            max_length=self.max_source_length,
            padding=False,
            truncation=True,
        )

        # generate target
        targets = self.tokenizer(
            answer=[self.answer_sep.join(answer) for answer in answers],
            max_length=self.max_target_length,
            padding=False,
            truncation=True,
        )
        for k, v in targets.items():
            outputs[f'target_{k}'] = v
            outputs[f'label_{k}'] = v

        # check lengths
        for source, target, label in zip(outputs['input_ids'], outputs['target_input_ids'], outputs['label_input_ids']):
            assert len(source) <= self.max_source_length and len(target) == len(label) <= self.max_target_length, \
                f'#tokens exceeds max_length {self.max_source_length} or inconsistent source={len(source)}, target={len(target)}, label={len(label)}'

        return outputs

    def salient_masking(
        self,
        contexts: List[str],
        tables: List[pd.DataFrame],
        ctx_mentions: List[List[Tuple[int, int]]],
    ):
        # truncate
        contexts, tables = self.tokenizer.truncate(
            query=contexts,
            table=tables,
            max_query_length=self.max_context_length,
            max_length=self.max_source_length - self.num_safe_tokens)

        # remove overflow mentions
        assert len(contexts) == len(ctx_mentions), 'inconsistent length'
        for i, (context, ctx_mention) in enumerate(zip(contexts, ctx_mentions)):
            ctx_mentions[i] = [(s, e) for s, e in ctx_mention if s < len(context) and e <= len(context)]

        # masking
        mask_kwargs = {
            'mask_granularity': 'span',
            'mask_strategy': 'mask',
            'merge_consecutive_masks': False,
            'replace_unmasked_tokens_with_pad': True
        }
        outputs = self.tokenizer(
            table=tables,
            query=contexts,
            query_mask=ctx_mentions,
            max_length=self.max_source_length,
            padding=False,
            truncation=True,
            mask_kwargs=mask_kwargs)

        # check lengths
        for source, target, label in zip(outputs['input_ids'], outputs['target_input_ids'], outputs['label_input_ids']):
            assert len(source) <= len(target) and len(target) == len(label) <= self.max_source_length, \
                f'#tokens exceeds max_length {self.max_source_length} or inconsistent source={len(source)}, target={len(target)}, label={len(label)}'

        return outputs

    def random_masking(
        self,
        contexts: List[str],
        tables: List[pd.DataFrame]
    ):
        # truncate
        contexts, tables = self.tokenizer.truncate(
            query=contexts,
            table=tables,
            max_query_length=self.max_context_length,
            max_length=self.max_source_length - self.num_safe_tokens)

        # generate random masks
        _contexts: List[str] = []
        _tables: List[pd.DataFrame] = []
        context_masks: List[List[Tuple[int, int]]] = []
        table_masks: List[pd.DataFrame] = []
        for context, table in zip(contexts, tables):
            # first mask table elements
            table_row = table.values
            num_col = len(table.columns)
            num_row = len(table)
            num_eles = num_col * (num_row + 1)
            num_eles_to_mask = max(1, math.ceil(num_eles * self.table_mask_prob))  # at least mask 1 element

            if num_eles < num_eles_to_mask:
                logger.warning('skip because there is no table elements to mask')
                continue

            mask_inds = np.random.choice(num_eles, num_eles_to_mask, replace=False)

            col_mask = [False] * num_col
            row_mask = [[False] * num_col for _ in range(num_row)]
            eles_to_be_masked: List[str] = []
            for i in mask_inds:
                if i < num_col:  # mask eles in header
                    col_mask[i] = True
                    eles_to_be_masked.append(table.columns[i])
                else:  # mask eles in row
                    row_ind = ((i - num_col) // num_col, (i - num_col) % num_col)
                    row_mask[row_ind[0]][row_ind[1]] = True
                    eles_to_be_masked.append(table_row[row_ind])

            table_mask = pd.DataFrame.from_records(row_mask, columns=col_mask)

            # second mask context
            # compute the num of mask tokens uesd by the table
            context_mask: List[Tuple[int, int]] = []
            num_remain_masks = self.max_num_mask_tokens - len(self.tokenizer.tokenize(' '.join(eles_to_be_masked)))
            if num_remain_masks > 0:
                tokens: List[Tuple[int, int]] = self.tokenizer.fast_tokenize(context)['offset_mapping']
                num_tokens_to_mask = min(num_remain_masks, math.ceil(len(tokens) * self.context_mask_prob))
                if num_tokens_to_mask > 0:
                    mask_inds = sorted(np.random.choice(len(tokens), num_tokens_to_mask, replace=False))
                    # merge the char-based index of consecutive tokens
                    for i, mi in enumerate(mask_inds):
                        if i > 0 and mi == mask_inds[i - 1] + 1:  # consecutive
                            context_mask[-1] = (context_mask[-1][0], tokens[mi][1])
                        else:
                            context_mask.append(tokens[mi])

            _contexts.append(context)
            _tables.append(table)
            context_masks.append(context_mask)
            table_masks.append(table_mask)

        contexts = _contexts
        tables = _tables

        # masking
        mask_kwargs = {
            'mask_granularity': 'token',
            'mask_strategy': 'bert',
            'merge_consecutive_masks': True,
            'replace_unmasked_tokens_with_pad': True
        }
        outputs = self.tokenizer(
            table=tables,
            query=contexts,
            table_mask=table_masks,
            query_mask=context_masks,
            max_length=self.max_source_length,
            padding=False,
            truncation=True,
            mask_kwargs=mask_kwargs)

        # check lengths
        for source, target, label in zip(outputs['input_ids'], outputs['target_input_ids'], outputs['label_input_ids']):
            assert len(source) <= len(target) and len(target) == len(label) <= self.max_source_length, \
                f'#tokens exceeds max_length {self.max_source_length} or inconsistent source={len(source)}, target={len(target)}, label={len(label)}'

        return outputs

    def process_single_dataset(
        self,
        dataset: Dataset,
        methods: List[str],
        num_proc: int = 1,
        overwrite_cache: bool = False,
    ):
        def _process(examples):
            contexts: List[str] = examples['context']
            tables: List[pd.DataFrame] = [pd.DataFrame.from_records(table['rows'], columns=table['header']) for table in examples['table']]
            ctx_mentions: List[List[Tuple[int, int]]] = examples['mentions']
            answers: List[List[str]] = examples['answers']

            # add a prefix space to contexts and answers
            for i, (context, mentions) in enumerate(zip(contexts, ctx_mentions)):
                context, added = self.add_prefix_space(context)
                if added:
                    mentions = [(s + 1, e + 1) for s, e in mentions]
                    contexts[i] = context
                    ctx_mentions[i] = mentions
            answers = [([self.add_prefix_space(ans[0])[0]] + ans[1:]) if len(ans) else ans for ans in answers]

            outputs = {
                'input_ids': [],
                'attention_mask': [],
                'target_input_ids': [],
                'target_attention_mask': [],
                'label_input_ids': [],
                'label_attention_mask': [],
            }
            def merge_outputs(_outputs: Dict):
                for k, v in _outputs.items():
                    outputs[k].extend(v)

            # generate pretraining examples using one or multiple methods
            for method in methods:
                if method == 'random_masking':
                    merge_outputs(self.random_masking(contexts, tables))
                elif method == 'salient_masking':
                    merge_outputs(self.salient_masking(contexts, tables, ctx_mentions))
                elif method == 'qa':
                    merge_outputs(self.qa(contexts, tables, answers))
                else:
                    raise ValueError

            # prepare labels
            outputs['labels'] = outputs['label_input_ids']
            del outputs['label_input_ids']
            del outputs['label_attention_mask']

            # prepare decoder_input_ids
            del outputs['target_attention_mask']

            return outputs

        dataset = dataset.map(
            _process,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            load_from_cache_file=not overwrite_cache,
            desc=f'processing data with methods: {methods}')
        return dataset

    def process(
        self,
        dataset: OmnitabPretrainDataset,
        **kwargs
    ):
        nat_data = self.process_single_dataset(dataset.nat_data, methods=['random_masking', 'salient_masking'], **kwargs)
        syn_data = self.process_single_dataset(dataset.syn_data, methods=['qa'], **kwargs)
        sql_data = self.process_single_dataset(dataset.sql_data, methods=['qa'], **kwargs)
        data = concatenate_datasets([nat_data, syn_data, sql_data])
        return data


@dataclass
class DataCollatorWithTargetToBeShifted(DataCollatorForSeq2Seq):
    target_field: str = None

    def pad(self, tensor):
        max_length = max(len(l) for l in tensor)
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        padding_side = self.tokenizer.padding_side
        padded_tensor = []
        for t in tensor:
            remainder = [self.label_pad_token_id] * (max_length - len(t))
            if isinstance(t, list):
                t = ((t + remainder) if padding_side == 'right' else (remainder + t))
            elif padding_side == 'right':
                t = np.concatenate([t, remainder]).astype(np.int64)
            else:
                t = np.concatenate([remainder, t]).astype(np.int64)
            padded_tensor.append(t)

        return padded_tensor

    def __call__(self, features, return_tensors=None):
        has_target = self.target_field in features[0]
        if has_target:
            targets = []
            for example in features:
                targets.append(example[self.target_field])
                del example[self.target_field]
        features = super().__call__(features, return_tensors=return_tensors)
        return_tensors = return_tensors or self.return_tensors

        # prepare decoder_input_ids from targets
        if (
            has_target
            and self.model is not None
            and hasattr(self.model, 'prepare_decoder_input_ids_from_labels')
        ):
            assert return_tensors == 'pt', NotImplemented
            targets = torch.tensor(self.pad(targets)).to(features['input_ids'])
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=targets)
            features['decoder_input_ids'] = decoder_input_ids
            if 'labels' in features:
                assert features['decoder_input_ids'].size() == features['labels'].size()

        return features
