#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import inspect
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, Union, List, Tuple
from enum import Enum

from omnitab.utils import BertTokenizerWrapper, ElectraTokenizer, BertConfig, ElectraConfig, \
    RobertaTokenizer, RobertaConfig, BartConfig, BartTokenizerWrapper
from omnitab.utils import BertTokenizerFast, ElectraTokenizerFast, RobertaTokenizerFast, BartTokenizerFastWrapper


BERT_CONFIGS = {
    'bert-base-uncased': BertConfig(
        vocab_size_or_config_json_file=30522,
        attention_probs_dropout_prob=0.1,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=2,
    )
    # Model config {
    #   "attention_probs_dropout_prob": 0.1,
    #   "hidden_act": "gelu",
    #   "hidden_dropout_prob": 0.1,
    #   "hidden_size": 768,
    #   "initializer_range": 0.02,
    #   "intermediate_size": 3072,
    #   "layer_norm_eps": 1e-12,
    #   "max_position_embeddings": 512,
    #   "num_attention_heads": 12,
    #   "num_hidden_layers": 12,
    #   "type_vocab_size": 2,
    #   "vocab_size": 30522
    # }
    ,
    'bert-large-uncased': BertConfig(
        vocab_size_or_config_json_file=30522,
        attention_probs_dropout_prob=0.1,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        num_attention_heads=16,
        num_hidden_layers=24,
        type_vocab_size=2,
    )
}

ELECTRA_GEN_DIVISOR = {
    'google/electra-small-generator': 4,
    'google/electra-base-generator': 3,
    'google/electra-large-generator': 4
}


class ModelType(Enum):
    BERT = 'bert'
    ELECTRA = 'electra'
    RoBERTa = 'roberta'
    BART = 'bart'


MODEL2TOKENIZER = {
    ModelType.BERT: BertTokenizerWrapper,
    ModelType.ELECTRA: ElectraTokenizer,
    ModelType.RoBERTa: RobertaTokenizer,
    ModelType.BART: BartTokenizerWrapper,
}

MODEL2TOKENIZERFAST = {
    ModelType.BERT: BertTokenizerFast,
    ModelType.ELECTRA: ElectraTokenizerFast,
    ModelType.RoBERTa: RobertaTokenizerFast,
    ModelType.BART: BartTokenizerFastWrapper,
}

MODEL2SEP = {
    ModelType.BERT: '[SEP]',
    ModelType.ELECTRA: '[SEP]',
    ModelType.RoBERTa: '</s>',
    ModelType.BART: '</s>',
}


MODEL2CLS = {
    ModelType.BERT: '[CLS]',
    ModelType.ELECTRA: '[CLS]',
    ModelType.RoBERTa: '<s>',
    ModelType.BART: '<s>',
}


MODEL2MASK = {
    ModelType.BERT: '[MASK]',
    ModelType.ELECTRA: '[MASK]',
    ModelType.RoBERTa: '<mask>',
    ModelType.BART: '<mask>',
}


MODEL2PADID = {
    ModelType.BERT: 0,
    ModelType.ELECTRA: 0,
    ModelType.RoBERTa: 1,
    ModelType.BART: 1,
}

MODEL2PAD = {
    ModelType.BERT: '[PAD]',
    ModelType.ELECTRA: '[PAD]',
    ModelType.RoBERTa: '<pad>',
    ModelType.BART: '<pad>',
}

class TableBertConfig(SimpleNamespace):
    MAX_SOURCE_LEN: int = 512
    MAX_TARGET_LEN: int = 512

    def __init__(
        self,
        base_model_name: str = 'bert-base-uncased',
        load_model_from: str = None,
        column_delimiter: str = '[SEP]',
        column_delimiter_first: str = None,
        skip_sep_in_middle: bool = False,
        row_delimiter: str = '[SEP]',
        context_first: bool = True,
        cell_input_template: str = 'column | type | value',
        column_representation: str = 'mean_pool',
        column_repr_dpr: str = 'whole_span',
        max_cell_len: int = 5,
        max_sequence_len: int = 512,  # TODO: never used
        max_context_len: int = 256,
        masked_context_prob: float = 0.15,
        masked_column_prob: float = 0.2,
        max_predictions_per_seq: int = 100,
        context_sample_strategy: str = 'nearest',
        table_mask_strategy: str = 'column',
        do_lower_case: bool = True,
        objective_function: str = 'mlm',
        contrastive_emb_size: int = 512,
        additional_row_count: int = 0,
        top_row_count: int = 0,
        only_keep_highlighted_rows: bool = False,
        highlight_table: str = None,
        max_num_mention_per_example: int = 0,
        use_sampled_value: bool = False,
        mask_used_column_prob: float = 0.0,
        mask_value: bool = False,
        mask_value_column_separate: bool = False,
        max_column_len: int = None,
        skip_column_name_longer_than: int = None,
        not_skip_empty_column_name: bool = False,
        only_table: bool = False,
        seq2seq_format: str = None,
        table_linearization: str = None,
        multi_decode_sep_token: str = '<|>',
        column_wise: bool = False,
        not_strict_mask: bool = False,
        max_source_len: int = 512,
        max_target_len: int = 512,
        **kwargs
    ):
        super(TableBertConfig, self).__init__()
        # update the class-level value
        TableBertConfig.MAX_SOURCE_LEN = max_source_len
        TableBertConfig.MAX_TARGET_LEN = max_target_len

        self.base_model_name = base_model_name
        self.load_model_from = load_model_from
        self.model_type = self.check_model_type(base_model_name)
        self.use_electra = self.model_type == ModelType.ELECTRA

        self.tokenizer_cls = MODEL2TOKENIZER[self.model_type]
        self.tokenizer_fast_cls = MODEL2TOKENIZERFAST[self.model_type]
        tokenizer = self.tokenizer_cls.from_pretrained(self.base_model_name)

        self.column_delimiter = column_delimiter
        self.column_delimiter_first = column_delimiter_first or self.column_delimiter
        self.column_delimiter: str = self.preprocess_column_delimiter(
            self.column_delimiter, self.model_type, tokenizer)
        self.column_delimiter_first: str = self.preprocess_column_delimiter(
            self.column_delimiter_first, self.model_type, tokenizer)
        self.skip_sep_in_middle = skip_sep_in_middle
        self.row_delimiter = row_delimiter
        if row_delimiter == '[SEP]':  # model-dependent delimiter
            self.row_delimiter = MODEL2SEP[self.model_type]
        elif row_delimiter == 'none':
            self.row_delimiter = None
        else:
            raise NotImplementedError
        self.sep_token = MODEL2SEP[self.model_type]
        self.sep_id = tokenizer.convert_tokens_to_ids([self.sep_token])[0]
        assert multi_decode_sep_token != self.sep_token, \
            'sep token is eos tokens, which should not be used as a symbol for multi answer decoding'
        self.multi_decode_sep_tokens: List[str] = tokenizer.tokenize(multi_decode_sep_token)
        self.cls_token = MODEL2CLS[self.model_type]
        self.mask_token = MODEL2MASK[self.model_type]
        self.pad_id = MODEL2PADID[self.model_type]
        self.context_first = context_first
        self.column_representation = column_representation
        self.column_repr_dpr = column_repr_dpr
        self.objective_function = objective_function
        self.column_wise = column_wise
        assert objective_function in {
            'mlm', 'split_mlm',
            'contrastive', 'contrastive_mlm', 'contrast-concat_mlm', 'nsp_mlm', 'contrast-span_mlm',
            'table2text_mlm', 'text2table_mlm', 'table2text_text2table', 'table2text_text2table_mlm',  # bart
            'binary_mlm', 'separate-bin_mlm',
            'seq2seq',  # bart
        }
        self.contrastive_emb_size = contrastive_emb_size

        self.max_cell_len = max_cell_len  # for cell value
        self.max_column_len = max_column_len  # for column name
        self.max_sequence_len = max_sequence_len
        self.max_context_len = max_context_len

        self.do_lower_case = do_lower_case
        self.not_strict_mask = not_strict_mask

        if isinstance(cell_input_template, str):
            if ' ' in cell_input_template:
                _cell_input_template = []
                for t in cell_input_template.split(' '):
                    if t in {'column', 'value', 'type'}:  # pre-defined keywords
                        _cell_input_template.append(t)
                    else:
                        assert len(tokenizer.tokenize(t)) == 1, 'template should only contain single-piece tokens'
                        _cell_input_template.append(tokenizer.tokenize(t)[0])
                cell_input_template = _cell_input_template
            else:
                cell_input_template = [cell_input_template]  # assume there is only a single element in the template

        self.cell_input_template = cell_input_template

        self.masked_context_prob = masked_context_prob
        self.masked_column_prob = masked_column_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.context_sample_strategy = context_sample_strategy
        self.table_mask_strategy = table_mask_strategy
        self.additional_row_count = additional_row_count
        self.top_row_count = top_row_count
        self.only_keep_highlighted_rows = only_keep_highlighted_rows
        self.highlight_table = highlight_table
        assert additional_row_count >= 0 and top_row_count >= 0
        assert not additional_row_count or not top_row_count, \
            'additional_row_count and top_row_count cannot be non-zero at the same time'
        self.max_num_mention_per_example = max_num_mention_per_example
        self.use_sampled_value = use_sampled_value
        self.mask_used_column_prob = mask_used_column_prob
        assert mask_used_column_prob in {0.0, 1.0}, 'other values are not implemented'
        self.mask_value_column_separate = mask_value_column_separate
        self.mask_value = mask_value or additional_row_count > 0 or mask_value_column_separate
        self.skip_column_name_longer_than = skip_column_name_longer_than
        self.not_skip_empty_column_name = not_skip_empty_column_name
        self.only_table = only_table
        self.seq2seq_format = seq2seq_format
        self.table_linearization = table_linearization

        if not hasattr(self, 'vocab_size_or_config_json_file'):
            if self.model_type == ModelType.BERT:
                bert_config = BERT_CONFIGS[self.base_model_name]
                for k, v in vars(bert_config).items():
                    setattr(self, k, v)
            elif self.model_type == ModelType.ELECTRA:
                assert 'generator' in self.base_model_name, 'use "generator" as the base_model_name when using ELECTRA'
                divisor = ELECTRA_GEN_DIVISOR[self.base_model_name]
                self.disc_config = ElectraConfig.from_pretrained(
                    self.base_model_name.replace('generator', 'discriminator'))
                self.gen_config = ElectraConfig.from_pretrained(self.base_model_name)
                self.gen_config.hidden_size = self.disc_config.hidden_size // divisor
                self.gen_config.num_attention_heads = self.disc_config.num_attention_heads // divisor
                self.gen_config.intermediate_size = self.disc_config.intermediate_size // divisor
            elif self.model_type == ModelType.RoBERTa:
                self.roberta_config = RobertaConfig.from_pretrained(self.base_model_name)
            elif self.model_type == ModelType.BART:
                self.bart_config = BartConfig.from_pretrained(self.base_model_name)
            else:
                raise NotImplementedError

    @classmethod
    def preprocess_column_delimiter(cls, delimiter: str, model_type: ModelType, tokenizer) -> str:
        if delimiter == '[SEP]':  # model-dependent delimiter
            delimiter = MODEL2SEP[model_type]
        else:
            delimiter = tokenizer.tokenize(delimiter)
            assert len(delimiter) == 1, 'column_delimiter should only contain a single word piece'
            delimiter = delimiter[0]
        return delimiter

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument('--base_model_name', type=str, default='bert-base-uncased')

        parser.add_argument('--context_first', dest='context_first', action='store_true')
        parser.add_argument('--table_first', dest='context_first', action='store_false')
        parser.set_defaults(context_first=True)

        parser.add_argument("--column_delimiter", type=str, default='[SEP]', help='Column delimiter')
        parser.add_argument("--column_delimiter_first", type=str, default=None, help='delimiter for the first column (used fro TAPEX)')
        parser.add_argument('--skip_sep_in_middle', action='store_true', help='whether to skip the sep between text and table')
        parser.add_argument("--row_delimiter", type=str, default='[SEP]', help='row delimiter')
        parser.add_argument("--cell_input_template", type=str, default='column | type | value', help='Cell representation')
        parser.add_argument("--column_representation", type=str, default='mean_pool', help='Column representation')
        parser.add_argument('--column_repr_dpr', type=str, default='whole_span', help='which part to use in DPR table index')

        # training specifications
        parser.add_argument("--max_sequence_len", type=int, default=512)
        parser.add_argument("--max_context_len", type=int, default=256)
        parser.add_argument('--max_source_len', type=int, default=512)
        parser.add_argument('--max_target_len', type=int, default=512)

        parser.add_argument("--max_cell_len", type=int, default=5)
        parser.add_argument("--max_column_len", type=int, default=None)
        parser.add_argument("--skip_column_name_longer_than", type=int, default=10)
        parser.add_argument("--not_skip_empty_column_name", action='store_true')

        parser.add_argument("--masked_context_prob", type=float, default=0.15,
                            help="Probability of masking each token for the LM task")
        parser.add_argument("--masked_column_prob", type=float, default=0.20,
                            help="Probability of masking each token for the LM task")
        parser.add_argument("--max_predictions_per_seq", type=int, default=200,
                            help="Maximum number of tokens to mask in each sequence")

        parser.add_argument('--context_sample_strategy', type=str, default='nearest',
                            choices=['nearest', 'concate_and_enumerate'])
        parser.add_argument('--table_mask_strategy', type=str, default='column',
                            choices=['column', 'column_token'])
        parser.add_argument('--additional_row_count', type=int, default=0)
        parser.add_argument('--top_row_count', type=int, default=0)
        parser.add_argument('--only_keep_highlighted_rows', action='store_true')
        parser.add_argument('--highlight_table', type=str, default=None)
        parser.add_argument('--max_num_mention_per_example', type=int, default=5,
                            help='max number of mentions (in either context or table) to use')
        parser.add_argument('--use_sampled_value', action='store_true')
        parser.add_argument('--mask_used_column_prob', type=float, default=0.0, help='probability of only masking used columns')
        parser.add_argument('--mask_value', action='store_true')
        parser.add_argument('--mask_value_column_separate', action='store_true')
        parser.add_argument('--table_linearization', type=str, default='tabert', choices=['tabert', 'tapex'],
                            help='specifies how to linearize tables')
        parser.add_argument('--seq2seq_format', type=str,
                            choices=[None, 'mlm', 'mlm_single-c2v', 'mlm_single-v2c', 'mlm_single-c2v_single-v2c', 'single-c2v_single-v2c',
                                     'qa_firstansrow', 'qa_allrow', 'qa_tapex', 'sql', 'totto',
                                     'cell-filling-mask', 'cell-filling-gen', 'schema-augmentation-mask', 'schema-augmentation-gen',
                                     'mention-context', 'mention-table', 'mlm_mention-context',
                                     'mlm_mention-table', 'mlm_mention-dedup-table', 'mlm_table-row-1',
                                     'bart-mask', 'salient-mask', 'bart-mask_salient-mask', 'bart-mask_mention-context',
                                     'data2text', 'clean-text', 'clean-fake-text', 'bidirection', 'sql2nl', 'sqltable2nl'],
                            help='seq2seq examples for BART-like models')
        parser.add_argument('--column_wise', action='store_true', help='linearize the table by columns')
        parser.add_argument('--not_strict_mask', action='store_true', help='skip errors during masking')
        parser.add_argument("--do_lower_case", action="store_true")
        parser.set_defaults(do_lower_case=True)

        return parser

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **override_args):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        args = json.load(file_path.open())
        override_args = override_args or dict()
        args.update(override_args)
        default_config = cls()
        config_dict = {}
        for key, default_val in vars(default_config).items():
            val = args.get(key, default_val)
            config_dict[key] = val

        # backward compatibility
        if 'column_item_delimiter' in args:
            column_item_delimiter = args['column_item_delimiter']
            cell_input_template = 'column'
            use_value = args.get('use_sample_value', True)
            use_type = args.get('use_type_text', True)

            if use_type:
                cell_input_template += column_item_delimiter + 'type'
            if use_value:
                cell_input_template += column_item_delimiter + 'value'

            config_dict['cell_input_template'] = cell_input_template

        config = cls(**config_dict)

        return config

    @classmethod
    def from_dict(cls, args: Dict):
        return cls(**args)

    @classmethod
    def check_model_type(cls, model_name: str) -> ModelType:
        model_name = model_name.lower()
        if 'electra' in model_name:
            return ModelType.ELECTRA
        if 'roberta' in model_name:
            return ModelType.RoBERTa
        if 'bart' in model_name:
            return ModelType.BART
        return ModelType.BERT

    @classmethod
    def get_special_tokens(cls, model_name: str) -> Tuple[str, str, str]:
        mt = cls.check_model_type(model_name)
        tokenizer = MODEL2TOKENIZER[mt].from_pretrained(model_name)
        sep_token = MODEL2SEP[mt]
        cls_token = MODEL2CLS[mt]
        pad_token = MODEL2PAD[mt]
        return cls_token, sep_token, pad_token

    def with_new_args(self, **updated_args):
        new_config = self.__class__(**vars(self))
        for key, val in updated_args.items():
            setattr(new_config, key, val)

        return new_config

    def save(self, file_path: Path):
        json.dump(vars(self), file_path.open('w'), indent=2, sort_keys=True, default=str)

    def to_log_string(self):
        return json.dumps(vars(self), indent=2, sort_keys=True, default=str)

    def to_dict(self):
        return vars(self)

    def get_default_values_for_parameters(self):
        signature = inspect.signature(self.__init__)

        default_args = OrderedDict(
            (k, v.default)
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        )

        return default_args

    def extract_args(self, kwargs, pop=True):
        arg_dict = {}

        for key, default_val in self.get_default_values_for_parameters().items():
            if key in kwargs:
                val = kwargs.get(key)
                if pop:
                    kwargs.pop(key)

                arg_dict[key] = val

        return arg_dict

    @staticmethod
    def infer_model_class_from_config_dict(config_dict):
        if 'num_vertical_layers' in config_dict:
            from .vertical.vertical_attention_table_bert import VerticalAttentionTableBert
            return VerticalAttentionTableBert

        from .vanilla_table_bert import VanillaTableBert
        return VanillaTableBert

    @staticmethod
    def infer_model_class_from_config_file(config_file):
        config_dict = json.load(open(config_file))
        return TableBertConfig.infer_model_class_from_config_dict(config_dict)
