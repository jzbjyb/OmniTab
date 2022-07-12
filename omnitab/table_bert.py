#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Dict, List, Tuple, Optional, Set
import sys
import json
from pathlib import Path
import logging

import torch
from torch import nn as nn

from omnitab.utils import (
    BertForPreTraining, BertForMaskedLM, BertModel,
    BertTokenizer, BertConfig,
    TransformerVersion, TRANSFORMER_VERSION
)
from omnitab.table import Table
from omnitab.config import TableBertConfig, ModelType


MAX_BERT_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
NEGATIVE_NUMBER = -1e8
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


class TableBertModel(nn.Module):
    CONFIG_CLASS = TableBertConfig

    def __init__(
        self,
        config: TableBertConfig,
        **kwargs
    ):
        nn.Module.__init__(self)

        bert_model: Union[BertForPreTraining, BertModel] = kwargs.pop('bert_model', None)

        if bert_model is not None:
            logging.warning(
                'using `bert_model` to initialize `TableBertModel` is deprecated. '
                'I will still set `self._bert_model` this time.'
            )

        self._bert_model = bert_model
        self.tokenizer = config.tokenizer_cls.from_pretrained(config.base_model_name)
        self.tokenizer_fast = config.tokenizer_fast_cls.from_pretrained(config.base_model_name)
        self.config = config

    @property
    def bert(self) -> BertModel:
        """Return the underlying base BERT model"""
        return getattr(self, 'bert_{}'.format(self.config.model_type.value))()

    @property
    def bert_all(self):
        return getattr(self, 'bert_all_{}'.format(self.config.model_type.value))()

    def bert_bert(self):
        if not hasattr(self, '_bert_model') or getattr(self, '_bert_model') is None:
            raise ValueError('This instance does not have a base BERT model.')
        if hasattr(self._bert_model, 'bert'):
            return self._bert_model.bert
        else:
            return self._bert_model

    def bert_all_bert(self):
        return self.bert_bert()

    def bert_electra(self, return_all: bool = False):
        def _forward(*args, **kwargs):
            if 'output_all_encoded_layers' in kwargs:
                del kwargs['output_all_encoded_layers']
            if 'return_dict' in kwargs:
                del kwargs['return_dict']
            if return_all:
                return self._electra.discriminator(*args, **kwargs, return_dict=True)
            else:
                kwargs['output_hidden_states'] = True
                outputs = self._electra.discriminator(*args, **kwargs, return_dict=True).hidden_states[-1]
                return outputs, None
        return _forward

    def bert_all_electra(self):
        return self.bert_electra(return_all=True)

    def bert_roberta(self, return_all: bool = False):
        def _forward(*args, **kwargs):
            if 'output_all_encoded_layers' in kwargs:
                del kwargs['output_all_encoded_layers']
            if 'return_dict' in kwargs:
                del kwargs['return_dict']
            if 'token_type_ids' in kwargs:
                del kwargs['token_type_ids']
            if return_all:
                return self._roberta(*args, **kwargs, return_dict=True)
            else:
                kwargs['output_hidden_states'] = True
                outputs = self._roberta(*args, **kwargs, return_dict=True).hidden_states[-1]
                return outputs, None
        return _forward

    def bert_all_roberta(self):
        return self.bert_roberta(return_all=True)

    def bert_bart(self, return_all: bool = False):
        def _forward(*args, **kwargs):
            if 'output_all_encoded_layers' in kwargs:
                del kwargs['output_all_encoded_layers']
            if 'return_dict' in kwargs:
                del kwargs['return_dict']
            if 'token_type_ids' in kwargs:
                del kwargs['token_type_ids']
            if return_all:
                return self._bart(*args, **kwargs, return_dict=True)
            else:
                kwargs['output_hidden_states'] = True
                outputs = self._bart(*args, **kwargs, return_dict=True).encoder_last_hidden_state
                return outputs, None
        return _forward

    def bert_all_bart(self):
        return self.bert_bart(return_all=True)

    @property
    def bert_config(self) -> BertConfig:
        if self.config.model_type == ModelType.BERT:
            return self.bert.config
        elif self.config.model_type == ModelType.ELECTRA:
            return self._electra.discriminator.config
        elif self.config.model_type == ModelType.RoBERTa:
            return self._roberta.config
        elif self.config.model_type == ModelType.BART:
            config = self._bart.config
            if not hasattr(config, 'initializer_range'):
                config.initializer_range = config.init_std
            return config
        else:
            raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def output_size(self):
        return self.bert_config.hidden_size

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path],
        config_file: Optional[Union[str, Path]] = None,
        **override_config: Dict
    ):
        if model_path in ('bert-base-uncased', 'bert-large-uncased'):
            from omnitab.vanilla_table_bert import VanillaTableBert, TableBertConfig
            config = TableBertConfig(**override_config)
            model = VanillaTableBert(config)

            return model

        if model_path and isinstance(model_path, str):
            model_path = Path(model_path)

        if config_file is None:
            config_file = model_path.parent / 'tb_config.json'
        elif isinstance(config_file, str):
            config_file = Path(config_file)

        if model_path:
            state_dict = torch.load(str(model_path), map_location='cpu')
        else:
            state_dict = None

        config_dict = json.load(open(config_file))

        if cls == TableBertModel:
            if 'num_vertical_attention_heads' in config_dict:
                from omnitab.vertical.vertical_attention_table_bert import VerticalAttentionTableBert, VerticalAttentionTableBertConfig
                table_bert_cls = VerticalAttentionTableBert
                config_cls = VerticalAttentionTableBertConfig
            else:
                from omnitab.vanilla_table_bert import VanillaTableBert
                table_bert_cls = VanillaTableBert
                config_cls = TableBertConfig
        else:
            table_bert_cls = cls
            config_cls = table_bert_cls.CONFIG_CLASS

        config = config_cls.from_file(config_file, **override_config)
        model = table_bert_cls(config)

        # old table_bert format
        if state_dict is not None:
            # fix the name for weight `cls.predictions.decoder.bias`,
            # to make it compatible with the latest version of `transformers`

            from omnitab.utils import hf_flag
            if hf_flag == 'new':
                old_key_to_new_key_names: List[(str, str)] = []
                for key in state_dict:
                    if key.endswith('.predictions.bias'):
                        old_key_to_new_key_names.append(
                            (
                                key,
                                key.replace('.predictions.bias', '.predictions.decoder.bias')
                            )
                        )

                for old_key, new_key in old_key_to_new_key_names:
                    state_dict[new_key] = state_dict[old_key]

            if not any(key.startswith('_bert_model') for key in state_dict):
                print('warning: loading model from an old version', file=sys.stderr)
                bert_model = BertForMaskedLM.from_pretrained(
                    config.base_model_name,
                    state_dict=state_dict
                )
                model._bert_model = bert_model
            else:
                model.load_state_dict(state_dict, strict=True)

        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
        config: Optional[TableBertConfig] = None,
        state_dict: Optional[Dict] = None,
        **kwargs
    ) -> 'TableBertModel':
        # Avoid cyclic import.
        # TODO: a better way to import these dependencies?
        from omnitab.vertical.vertical_attention_table_bert import (
            VerticalAttentionTableBert,
            VerticalAttentionTableBertConfig
        )
        from omnitab.vanilla_table_bert import VanillaTableBert

        if model_name_or_path in {'bert-base-uncased', 'bert-large-uncased'}:
            config = TableBertConfig(base_model_name=model_name_or_path)
            overriding_config = config.extract_args(kwargs, pop=True)
            if len(overriding_config) > 0:
                config = config.with_new_args(**overriding_config)

            model = VanillaTableBert(config)

            return model

        if not isinstance(config, TableBertConfig):
            if config_file:
                config_file = Path(config_file)
            else:
                assert model_name_or_path, f'model path is None'
                config_file = Path(model_name_or_path).parent / 'tb_config.json'

            assert config_file.exists(), f'Unable to find TaBERT config file at {config_file}'

            # Identify from the json config file whether the model uses vertical self-attention (TaBERT(K>1))
            if cls == TableBertModel and VerticalAttentionTableBertConfig.is_valid_config_file(config_file):
                config_cls = VerticalAttentionTableBertConfig
            else:
                config_cls = TableBertConfig

            config = config_cls.from_file(config_file)

        overriding_config = config.extract_args(kwargs, pop=True)
        if len(overriding_config) > 0:
            config = config.with_new_args(**overriding_config)

        model_kwargs = kwargs

        model_cls = (
            cls    # If the current class is not the base generic class, then we assume the user want to
                   # load a pre-trained instance of that specific model class. Otherwise, we infer the model
                   # class from its config class
            if cls != TableBertModel
            else {
                TableBertConfig.__name__: VanillaTableBert,
                VerticalAttentionTableBertConfig.__name__: VerticalAttentionTableBert
            }[config.__class__.__name__]
        )

        model = model_cls(config, **model_kwargs)

        if state_dict is None:
            state_dict = torch.load(model_name_or_path, map_location="cpu")

        # fix the name for weight `cls.predictions.decoder.bias`,
        # to make it compatible with the latest version of HuggingFace `transformers`
        if TRANSFORMER_VERSION == TransformerVersion.TRANSFORMERS and config.model_type == ModelType.BERT:
            old_key_to_new_key_names: List[(str, str)] = []
            for key in state_dict:
                if key.endswith('.predictions.bias'):
                    old_key_to_new_key_names.append(
                        (
                            key,
                            key.replace('.predictions.bias', '.predictions.decoder.bias')
                        )
                    )

            for old_key, new_key in old_key_to_new_key_names:
                state_dict[new_key] = state_dict[old_key]

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            logging.warning('The state dict is not compatible with the model, '
                            'please check manually to see if this is OK.')
            model.load_state_dict(state_dict, strict=False)

        return model

    def encode(
            self,
            contexts: List[List[str]],
            tables: List[Table],
            **kwargs: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        raise NotImplementedError
