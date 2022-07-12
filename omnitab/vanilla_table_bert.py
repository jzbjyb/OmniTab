#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import sys
from typing import List, Any, Tuple, Dict, Union
import numpy as np
from fairseq import distributed_utils
from tqdm import tqdm
import json
import os
from collections import defaultdict
import torch
from torch.nn import CrossEntropyLoss
from torch_scatter import scatter_max, scatter_mean

from omnitab.utils import BertForPreTraining, BertForMaskedLM, TRANSFORMER_VERSION, TransformerVersion
from omnitab.utils import ElectraForPreTraining, ElectraForMaskedLM, RobertaForMaskedLM, \
    BartForConditionalGeneration, shift_tokens_right
from omnitab.table_bert import TableBertModel
from omnitab.config import TableBertConfig, ModelType
from omnitab.table import Table
from omnitab.input_formatter import VanillaTableBertInputFormatter
from omnitab.electra import ELECTRAModel, ELECTRALoss
from omnitab.contrastive import CLIPLoss


class VanillaTableBert(TableBertModel):
    CONFIG_CLASS = TableBertConfig

    def __init__(
        self,
        config: TableBertConfig,
        **kwargs
    ):
        super(VanillaTableBert, self).__init__(config, **kwargs)

        self.lm_module_name = getattr(self, 'load_{}'.format(config.model_type.value))()  # load model based on model type
        obj = self.config.objective_function
        if 'contrastive' in obj:
            self.contrastive_loss = CLIPLoss(self.output_size, self.config.contrastive_emb_size)
        elif 'contrast-concat' in obj or 'separate-bin' in obj or 'contrast-span' in obj:
            self.contrastive_loss = CLIPLoss(self.output_size, self.config.contrastive_emb_size, is_paired=True)
        elif 'nsp' in obj or 'binary' in obj:
            self.nsp_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        if config.load_model_from is not None:
            print('init from {}'.format(config.load_model_from))
            state_dict = torch.load(config.load_model_from, map_location='cpu')
            try:
                self.load_state_dict(state_dict)
            except:
                print('directly load the LM')
                if self.config.model_type == ModelType.BART:
                    state_dict.pop('lm_head.weight', None)
                getattr(self, self.lm_module_name).load_state_dict(state_dict)
        self.input_formatter = VanillaTableBertInputFormatter(self.config, self.tokenizer)

    def load_bert(self):
        obj = self.config.objective_function
        for loss_fct in ['seq2seq']:
            if loss_fct in obj:
                raise NotImplementedError
        if 'nsp' in obj or 'binary' in obj:
            self._bert_model = BertForPreTraining.from_pretrained(self.config.base_model_name)
        else:
            self._bert_model = BertForMaskedLM.from_pretrained(self.config.base_model_name)
        return '_bert_model'

    def load_electra(self):
        for loss_fct in ['nsp', 'binary', 'seq2seq', 'contrast-span']:
            if loss_fct in self.config.objective_function:
                raise NotImplementedError
        generator = ElectraForMaskedLM.from_pretrained(self.config.base_model_name)
        discriminator = ElectraForPreTraining.from_pretrained(
            self.config.base_model_name.replace('generator', 'discriminator'))
        discriminator.electra.embeddings = generator.electra.embeddings
        generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight
        self._electra = ELECTRAModel(generator, discriminator)
        self._electra_loss = ELECTRALoss()
        return '_electra'

    def load_roberta(self):
        for loss_fct in ['nsp', 'binary', 'separate-bin', 'seq2seq', 'contrast-span']:
            if loss_fct in self.config.objective_function:
                raise NotImplementedError
        self._roberta = RobertaForMaskedLM.from_pretrained(self.config.base_model_name)
        return '_roberta'

    def load_bart(self):
        for loss_fct in ['nsp', 'binary', 'separate-bin', 'contrastive', 'contrast-concat', 'contrast-span']:
            if loss_fct in self.config.objective_function:
                raise NotImplementedError
        self._bart = BartForConditionalGeneration.from_pretrained(self.config.base_model_name)
        return '_bart'

    def forward(self, *args, **kwargs):
        return getattr(self, 'forward_{}'.format(self.config.model_type.value))(*args, **kwargs)

    def forward_bert(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, **kwargs):
        total_loss: torch.Tensor = 0.0
        sample_size = masked_lm_labels.ne(-1).sum().item()
        logging_output = {'sample_size': sample_size}
        obj = self.config.objective_function
        if 'mlm' in obj or 'binary' in obj:
            sequence_output, pooled_output = self._bert_model.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            if type(self._bert_model) is BertForMaskedLM:
                prediction_scores = self._bert_model.cls(sequence_output)
            elif type(self._bert_model) is BertForPreTraining:
                prediction_scores, seq_relationship_score = self._bert_model.cls(sequence_output, pooled_output)
            if 'mlm' in obj:
                loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
                masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), masked_lm_labels.view(-1))
                total_loss += masked_lm_loss
                logging_output['mlm_loss'] = masked_lm_loss.item()
            if 'binary' in obj:
                binary_label = 1 - kwargs['is_positives']  # 0 => next sentence is the continuation, 1 => next sentence is a random sentence
                binary_loss = self.nsp_loss(seq_relationship_score.view(-1, 2), binary_label.view(-1))
                total_loss += binary_loss
        if 'contrastive' in obj or 'separate-bin' in obj:
            # use the representation corresponding to the first token (cls or sep)
            context_repr = self._bert_model.bert(
                kwargs['context_input_ids'], kwargs['context_token_type_ids'], kwargs['context_attention_mask'],
                output_all_encoded_layers=False)[0][:, 0, :]
            table_repr = self._bert_model.bert(
                kwargs['table_input_ids'], kwargs['table_token_type_ids'], kwargs['table_attention_mask'],
                output_all_encoded_layers=False)[0][:, 0, :]
            if 'contrastive' in obj:
                contrastive_loss = self.contrastive_loss(context_repr, table_repr)
            elif 'separate-bin' in obj:
                contrastive_loss = self.contrastive_loss(context_repr, table_repr, labels=kwargs['is_positives'])
            else:
                raise ValueError
            total_loss += contrastive_loss
            logging_output['contrastive_loss'] = contrastive_loss.item()
        if 'contrast-span' in obj and kwargs['pos_mentions_cells'].size(0) > 0:
            mentions_repr = self.extract_span_context_bert(kwargs, field='context')[0]  # (bs, num_mentions, emb_size)
            cells_repr = self.extract_span_context_bert(kwargs, field='table')[0]  # (bs, num_cells, emb_size)
            mentions_repr = mentions_repr.view(-1, mentions_repr.size(-1))
            cells_repr = cells_repr.view(-1, cells_repr.size(-1))
            pos_mc = kwargs['pos_mentions_cells']  # (num_pos_pairs, 2)
            neg_mc = kwargs['neg_mentions_cells']  # (num_neg_pairs, 2)
            all_mentions = torch.cat([mentions_repr[pos_mc[:, 0]], mentions_repr[neg_mc[:, 0]]], 0)  # (num_pos_pairs + num_neg_pairs, emb_size)
            all_cells = torch.cat([cells_repr[pos_mc[:, 1]], cells_repr[neg_mc[:, 1]]], 0)  # (num_pos_pairs + num_neg_pairs, emb_size)
            labels = torch.cat([torch.ones(pos_mc.size(0)), torch.zeros(neg_mc.size(0))], 0).to(mentions_repr.device)
            span_contrastive_loss = self.contrastive_loss(all_mentions, all_cells, labels=labels)
            total_loss += span_contrastive_loss
            logging_output['span_contrastive_loss'] = span_contrastive_loss.item()
        elif 'contrast-span' in obj and kwargs['pos_mentions_cells'].size(0) <= 0:
            logging_output['span_contrastive_loss'] = 0.0
        if 'contrast-concat' in obj:
            # use the representation corresponding to the first token (cls or sep)
            concat_repr, _ = self._bert_model.bert(
                kwargs['concat_input_ids'], kwargs['concat_token_type_ids'], kwargs['concat_attention_mask'], output_all_encoded_layers=False)
            context_repr = concat_repr[kwargs['context_mask'], :]
            table_repr = concat_repr[kwargs['table_mask'], :]
            l, binary_label = CLIPLoss.get_diag_label(context_repr, binary=True)
            contrastive_loss = self.contrastive_loss(context_repr, table_repr, labels=binary_label)
            total_loss += contrastive_loss
        if 'nsp' in obj:
            # only use cls
            sequence_output, pooled_output = self._bert_model.bert(
                kwargs['concat_input_ids'], kwargs['concat_token_type_ids'], kwargs['concat_attention_mask'], output_all_encoded_layers=False)
            _, seq_relationship_score = self._bert_model.cls(sequence_output, pooled_output)
            l, nsp_label = CLIPLoss.get_diag_label(pooled_output, binary=True)
            nsp_label = 1 - nsp_label  # 0 => next sentence is the continuation, 1 => next sentence is a random sentence
            nsp_loss = self.nsp_loss(seq_relationship_score.view(-1, 2), nsp_label.view(-1))
            total_loss += nsp_loss

        logging_output['loss'] = total_loss.item()
        total_loss = total_loss * (sample_size or 1)
        return total_loss, logging_output

    def evaluate_bert(self, batch):
        results: List[Dict] = []
        input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
        repr = self._bert_model.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)[0]
        logits = self._bert_model.cls(repr)
        pred_ids = logits.max(-1)[1]
        gold_ids = batch['masked_lm_labels']
        for tgt_id, pred_id, gold_id in zip(input_ids, pred_ids, gold_ids):
            tgt = self.tokenizer.convert_ids_to_tokens(tgt_id.detach().cpu().numpy())
            pred = self.tokenizer.convert_ids_to_tokens(pred_id.detach().cpu().numpy())
            gold = self.tokenizer.convert_ids_to_tokens(gold_id.detach().cpu().numpy())
            result = {'tgt': tgt, 'pred': pred, 'gold': gold}
            results.append(result)
        return results

    def represent_bert(self, batch):
        context_repr = self._bert_model.bert(
            batch['context_input_ids'], batch['context_token_type_ids'], batch['context_attention_mask'],
            output_all_encoded_layers=False)[0][:, 0, :]
        table_repr = self._bert_model.bert(
            batch['table_input_ids'], batch['table_token_type_ids'], batch['table_attention_mask'],
            output_all_encoded_layers=False)[0][:, 0, :]
        if hasattr(self, 'contrastive_loss'):
            context_repr, table_repr = self.contrastive_loss(context_repr, table_repr, return_repr=True)
        return context_repr, table_repr

    def represent_avg_cell_bert(self, batch):
        # extract span repr
        context_span_repr, context_span_mask = self.extract_span_context_bert(batch, field='context', use_word_emb=False)[:2]
        table_span_repr, table_span_mask = self.extract_span_context_bert(batch, field='table', use_word_emb=False)[:2]
        # avg across spans
        context_repr = context_span_repr.sum(1) / torch.clamp(context_span_mask.sum(1), min=1).unsqueeze(-1)
        table_repr = table_span_repr.sum(1) / torch.clamp(table_span_mask.sum(1), min=1).unsqueeze(-1)
        if hasattr(self, 'contrastive_loss'):
            context_repr, table_repr = self.contrastive_loss(context_repr, table_repr, return_repr=True)
        return context_repr, table_repr

    def get_group_token_ids(self, token_ids, token_to_group_map):
        batchgroup2tokens: Dict[Tuple[int, int], Union[List, str]] = defaultdict(list)
        for b in range(token_ids.size(0)):
            for t in range(token_ids.size(1)):
                g = token_to_group_map[b, t].item()
                if g == -1:
                    continue
                batchgroup2tokens[(b, g)].append(token_ids[b, t])
        _batchgroup2tokens: Dict[Tuple[int, int], str] = defaultdict(lambda: '')
        for b, g in batchgroup2tokens:
            _batchgroup2tokens[(b, g)] = self.tokenizer_fast.decode(batchgroup2tokens[(b, g)])
        return _batchgroup2tokens

    def extract_span_noncontext_bert(self, batch, field: str):
        return self.extract_span_context_bert(batch, field, use_word_emb=True)

    def extract_span_context_bert(self, batch, field: str, use_word_emb: bool = False):
        assert field in {'table', 'context'}
        input_ids = batch[f'{field}_input_ids']
        if field == 'table':
            t2i = batch['table_column_token_to_column_id'] # (batch_size, seq_len)
        elif field == 'context':
            t2i = batch['context_context_token_to_mention_id']  # (batch_size, seq_len)
        else:
            raise ValueError(f'{field} not supported')

        # get representation
        if use_word_emb:
            token_repr = self._bert_model.bert.embeddings(input_ids, batch[f'{field}_token_type_ids'])
        else:
            token_repr = self._bert_model.bert(
                input_ids, batch[f'{field}_token_type_ids'], batch[f'{field}_attention_mask'],
                output_all_encoded_layers=False)[0]

        # get text
        batchgroup2tokens = self.get_group_token_ids(input_ids, t2i)

        # build tensors
        agg_func = scatter_mean
        t2i_mask = t2i.ne(-1)  # (batch_size, seq_len)
        num_spans = torch.clamp(t2i.max(-1)[0] + 1, min=0)  # (batch_size,)
        max_num_spans = int(num_spans.max().item())
        span_mask = torch.arange(max_num_spans).unsqueeze(0).to(t2i.device) < num_spans.unsqueeze(-1)  # (batch_size, seq_len)
        t2i_noneg = t2i * t2i_mask + (~t2i_mask) * num_spans.unsqueeze(-1)  # (batch_size, seq_len)

        # (batch_size, max_num_spans + 1, emb_size)
        span_repr = agg_func(token_repr, t2i_noneg.unsqueeze(-1).expand(
            -1, -1, token_repr.size(-1)), dim=1, dim_size=max_num_spans + 1, fill_value=0).type(token_repr.dtype)

        # remove the last "garbage collection" entry, mask out padding spans
        span_repr = span_repr[:, :-1] * span_mask.unsqueeze(-1)

        return span_repr, span_mask, batchgroup2tokens

    def represent_span_context_bert(self, batch, field: str, use_word_emb: bool = False):
        repre, mask, batchgroup2tokens = self.extract_span_context_bert(batch, field, use_word_emb=use_word_emb)
        if 'contrast-span' in self.config.objective_function:
            if field == 'context':
                repre = self.contrastive_loss.preprocess(repre, self.contrastive_loss.source_projection)
            elif field == 'table':
                repre = self.contrastive_loss.preprocess(repre, self.contrastive_loss.target_projection)
        return repre, mask, batchgroup2tokens

    def represent_span_noncontext_bert(self, batch, field: str):
        return self.represent_span_context_bert(batch, field, use_word_emb=True)

    def forward_electra(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, **kwargs):
        total_loss: torch.Tensor = 0.0
        sample_size = 0
        if 'mlm' in self.config.objective_function:
            gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied = \
                self._electra(input_ids, token_type_ids, attention_mask, masked_lm_labels)
            electra_loss = self._electra_loss(
                (gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied), masked_lm_labels)
            sample_size = masked_lm_labels.ne(-1).sum().item()
            total_loss += electra_loss
        if 'contrastive' in self.config.objective_function:
            # use the representation corresponding to the first token (cls or sep)
            context_repr = self._electra.discriminator.electra(
                kwargs['context_input_ids'],  kwargs['context_attention_mask'], kwargs['context_token_type_ids'])[0][:, 0, :]
            table_repr = self._electra.discriminator.electra(
                kwargs['table_input_ids'], kwargs['table_attention_mask'], kwargs['table_token_type_ids'])[0][:, 0, :]
            contrastive_loss = self.contrastive_loss(context_repr, table_repr)
            total_loss += contrastive_loss
        if 'contrast-concat' in self.config.objective_function:
            # use the representation corresponding to the first token (cls or sep)
            concat_repr = self._electra.discriminator.electra(
                kwargs['concat_input_ids'], kwargs['concat_attention_mask'], kwargs['concat_token_type_ids'])[0]
            context_repr = concat_repr[kwargs['context_mask'], :]
            table_repr = concat_repr[kwargs['table_mask'], :]
            contrastive_loss = self.contrastive_loss(context_repr, table_repr, labels=None)
            total_loss += contrastive_loss

        logging_output = {
            'sample_size': sample_size,
            'loss': total_loss.item()
        }
        total_loss = total_loss * (sample_size or 1)
        return total_loss, logging_output

    def forward_roberta(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, **kwargs):
        total_loss: torch.Tensor = 0.0
        sample_size = 0
        if 'mlm' in self.config.objective_function:
            sequence_logits = self._roberta(input_ids, attention_mask)[0]
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
            masked_lm_loss = loss_fct(sequence_logits.view(-1, sequence_logits.size(-1)), masked_lm_labels.view(-1))
            sample_size = masked_lm_labels.ne(-1).sum().item()
            total_loss += masked_lm_loss
        if 'contrastive' in self.config.objective_function:
            # use the representation corresponding to the first token (cls or sep)
            context_repr = self._roberta.roberta(kwargs['context_input_ids'], kwargs['context_attention_mask'])[0][:, 0, :]
            table_repr = self._roberta.roberta(kwargs['table_input_ids'], kwargs['table_attention_mask'])[0][:, 0, :]
            contrastive_loss = self.contrastive_loss(context_repr, table_repr)
            total_loss += contrastive_loss
        if 'contrast-concat' in self.config.objective_function:
            # use the representation corresponding to the first token (cls or sep)
            concat_repr = self._roberta.roberta(kwargs['concat_input_ids'], kwargs['concat_attention_mask'])[0]
            context_repr = concat_repr[kwargs['context_mask'], :]
            table_repr = concat_repr[kwargs['table_mask'], :]
            contrastive_loss = self.contrastive_loss(context_repr, table_repr, labels=None)
            total_loss += contrastive_loss

        logging_output = {
            'sample_size': sample_size,
            'loss': total_loss.item()
        }
        total_loss = total_loss * (sample_size or 1)
        return total_loss, logging_output

    def forward_bart(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, **kwargs):
        total_loss: torch.Tensor = 0.0
        sample_size = masked_lm_labels.ne(-1).sum().item()
        logging_output = {'sample_size': sample_size}

        if 'mlm' in self.config.objective_function:
            sequence_logits = self._bart(input_ids, attention_mask=attention_mask, return_dict=True).logits
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
            masked_lm_loss = loss_fct(sequence_logits.view(-1, sequence_logits.size(-1)), masked_lm_labels.view(-1))
            total_loss += masked_lm_loss

        for obj in ['text2table', 'table2text']:
            if obj not in self.config.objective_function:
                continue
            # get src, tgt
            if obj == 'text2table':
                src_ids, src_mask = kwargs['context_input_ids'], kwargs['context_attention_mask']
                tgt_ids = kwargs['table_input_ids']
            elif obj == 'table2text':
                src_ids, src_mask = kwargs['table_input_ids'], kwargs['table_attention_mask']
                tgt_ids = kwargs['context_input_ids']
            # compute loss
            decoder_input_ids = shift_tokens_right(tgt_ids, self.config.pad_id)
            logits = self._bart(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False, return_dict=True).logits
            loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_id, reduction='mean')
            seq2seq_loss = loss_fct(logits.view(-1, logits.size(-1)), tgt_ids.view(-1))
            total_loss += seq2seq_loss

        if 'seq2seq' in self.config.objective_function:
            src_ids, src_mask = input_ids, attention_mask
            tgt_ids = kwargs['target_input_ids']
            bs = src_ids.size(0)
            decoder_input_ids = shift_tokens_right(tgt_ids, self.config.pad_id)
            logits = self._bart(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False, return_dict=True).logits

            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            # masked_lm_labels is a combination of real mlm and seq2seq targets
            combined_loss = loss_fct(logits.view(-1, logits.size(-1)), masked_lm_labels.view(-1)).view(bs, -1)  # (bs, seq_len)
            per_example_loss = (combined_loss * masked_lm_labels.ne(-1)).sum(-1)  # (bs)
            combined_loss_avg = combined_loss.sum() / (masked_lm_labels.ne(-1).sum() or 1.0)
            total_loss += combined_loss_avg  # loss is proportional to the number of tokens of each type (mlm or seq2seq)
            # separate mlm and seq2seq loss for logging
            is_mlm = kwargs['is_mlm']  # (bs, )
            mlm_loss_avg = combined_loss[is_mlm].sum() / (masked_lm_labels[is_mlm].ne(-1).sum() or 1.0)
            seq2seq_loss_avg = combined_loss[~is_mlm].sum() / (masked_lm_labels[~is_mlm].ne(-1).sum() or 1.0)
            logging_output['mlm_loss'] = mlm_loss_avg.item()
            logging_output['seq2seq_loss'] = seq2seq_loss_avg.item()

        logging_output['loss'] = total_loss.item()
        total_loss = total_loss * (sample_size or 1)
        if 'return_per_example_loss' in kwargs and kwargs['return_per_example_loss']:
            return per_example_loss, logging_output
        return total_loss, logging_output

    def computeloss_bart(self, batch):
        return self.forward_bart(**batch)[0]

    def evaluate_bart(self, batch):
        results: List[Dict] = []
        src_ids, src_mask = batch['input_ids'], batch['attention_mask']
        tgt_ids = batch['target_input_ids']
        decoder_input_ids = shift_tokens_right(tgt_ids, self.config.pad_id)
        logits = self._bart(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids,
                            use_cache=False, return_dict=True).logits
        pred_ids = logits.max(-1)[1]
        gold_ids = batch['masked_lm_labels']
        for tgt_id, pred_id, gold_id in zip(tgt_ids, pred_ids, gold_ids):
            tgt = self.tokenizer.convert_ids_to_tokens(tgt_id)
            pred = self.tokenizer.convert_ids_to_tokens(pred_id)
            gold = self.tokenizer.convert_ids_to_tokens(gold_id)
            result = {'tgt': tgt, 'pred': pred, 'gold': gold}
            results.append(result)
        return results

    def encode_context_and_table(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_token_indices: torch.Tensor,
        context_token_mask: torch.Tensor,
        column_token_mask: torch.Tensor,
        column_token_to_column_id: torch.Tensor,
        column_mask: torch.Tensor,
        return_bert_encoding: bool = False,
        **kwargs
    ):

        # print('input_ids', input_ids.size(), file=sys.stderr)
        # print('segment_ids', segment_ids.size(), file=sys.stderr)
        # print('attention_mask', attention_mask.size(), file=sys.stderr)
        # print('column_token_mask', column_token_mask.size(), file=sys.stderr)
        # print('column_token_mask', column_token_mask.sum(dim=-1), file=sys.stderr)
        # print('column_token_to_column_id', column_token_to_column_id.size(), file=sys.stderr)
        # print('column_token_to_column_id', column_token_to_column_id.sum(dim=-1), file=sys.stderr)
        # print('column_mask', column_mask.size(), file=sys.stderr)

        kwargs = (
            {}
            if TRANSFORMER_VERSION == TransformerVersion.TRANSFORMERS
            else {'output_all_encoded_layers': False}
        )
        sequence_output, _ = self.bert(
            input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask,
            **kwargs
        )
        # except:
        #     print('!!!!!Exception!!!!!')
        #     datum = (input_ids, segment_ids, attention_mask, question_token_mask,
        #              column_token_mask, column_token_to_column_id, column_mask)
        #     torch.save(datum, 'debug.tensors.bin')
        #     raise

        # gather column representations
        # (batch_size, max_seq_len, encoding_size)
        flattened_column_encoding = sequence_output
        # (batch_size, max_column_size, encoding_size)
        column_encoding = self.get_column_representation(
            flattened_column_encoding,
            column_token_to_column_id,
            column_token_mask,
            column_mask,
            aggregator=self.config.column_representation
        )

        # (batch_size, context_len, encoding_size)
        context_encoding = torch.gather(
            sequence_output,
            dim=1,
            index=context_token_indices.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1)),
        )
        context_encoding = context_encoding * context_token_mask.unsqueeze(-1)

        encoding_info = {}
        if return_bert_encoding:
            encoding_info['bert_encoding'] = sequence_output

        return context_encoding, column_encoding, encoding_info

    @staticmethod
    def get_column_representation(
        flattened_column_encoding: torch.Tensor,
        column_token_to_column_id: torch.Tensor,
        column_token_mask: torch.Tensor,
        column_mask: torch.Tensor,
        aggregator: str = 'mean_pool'
    ) -> torch.Tensor:
        """
        Args:
            flattened_column_encoding: (batch_size, total_column_token_num, encoding_size)
            column_token_to_column_id: (batch_size, total_column_token_num + 1)
            column_mask: (batch_size, max_column_num)
            aggregator: ['mean_pool', 'max_pool', 'first_token']
        Returns:
            column_encoding: (batch_size, max_column_num, encoding_size)
        """

        if aggregator.startswith('max_pool'):
            agg_func = scatter_max
            flattened_column_encoding[column_token_mask == 0] = float('-inf')
        elif aggregator.startswith('mean_pool') or aggregator.startswith('first_token'):
            agg_func = scatter_mean
        else:
            raise ValueError(f'Unknown column representation method {aggregator}')

        max_column_num = column_mask.size(-1)
        # column_token_to_column_id: (batch_size, max_column_num)
        # (batch_size, max_column_size + 1, encoding_size)
        result = agg_func(flattened_column_encoding,
                          column_token_to_column_id.unsqueeze(-1).expand(-1, -1, flattened_column_encoding.size(-1)),
                          dim=1,
                          dim_size=max_column_num + 1)

        # remove the last "garbage collection" entry, mask out padding columns
        result = result[:, :-1] * column_mask.unsqueeze(-1)

        if aggregator == 'max_pool':
            column_encoding = result[0]
        else:
            column_encoding = result

        return column_encoding

    def to_tensor_dict(
        self,
        contexts: List[List[str]],
        tables: List[Table],
        table_specific_tensors=True
    ):
        instances = []
        for e_id, (context, table) in enumerate(zip(contexts, tables)):
            instance = self.input_formatter.get_input(context, table)
            instances.append(instance)

        batch_size = len(contexts)
        max_sequence_len = max(len(x['tokens']) for x in instances)

        # basic tensors
        input_array = np.zeros((batch_size, max_sequence_len), dtype=np.int)
        mask_array = np.zeros((batch_size, max_sequence_len), dtype=np.bool)
        segment_array = np.zeros((batch_size, max_sequence_len), dtype=np.bool)

        # table specific tensors
        if table_specific_tensors:
            max_column_num = max(len(x['column_spans']) for x in instances)
            max_context_len = max(x['context_length'] for x in instances)

            context_token_indices = np.zeros((batch_size, max_context_len), dtype=np.int)
            context_mask = np.zeros((batch_size, max_context_len), dtype=np.bool)
            column_token_mask = np.zeros((batch_size, max_sequence_len), dtype=np.bool)

            # we initialize the mapping with the id of last column as the "garbage collection" entry for reduce ops
            column_token_to_column_id = np.zeros((batch_size, max_sequence_len), dtype=np.int)
            column_token_to_column_id.fill(max_column_num)

            column_mask = np.zeros((batch_size, max_column_num), dtype=np.bool)

            column_span = 'whole_span'
            if 'column_name' in self.config.column_representation:
                column_span = 'column_name'
            elif 'first_token' in self.config.column_representation:
                column_span = 'first_token'

        for i, instance in enumerate(instances):
            token_ids = self.tokenizer.convert_tokens_to_ids(instance['tokens'])

            input_array[i, :len(token_ids)] = token_ids
            segment_array[i, instance['segment_a_length']: len(token_ids)] = 1
            mask_array[i, :len(token_ids)] = 1.

            if table_specific_tensors:
                context_token_indices[i, :instance['context_length']] = list(range(*instance['context_span'])) #instance['context_token_indices']
                context_mask[i, :instance['context_length']] = 1.

                header = tables[i].header
                for col_id, column in enumerate(header):
                    if col_id < len(instance['column_spans']):
                        col_start, col_end = instance['column_spans'][col_id][column_span]

                        column_token_to_column_id[i, col_start: col_end] = col_id
                        column_token_mask[i, col_start: col_end] = 1.
                        column_mask[i, col_id] = 1.

        tensor_dict = {
            'input_ids': torch.tensor(input_array.astype(np.int64)),
            'segment_ids': torch.tensor(segment_array.astype(np.int64)),
            'attention_mask': torch.tensor(mask_array, dtype=torch.float32),
        }

        if table_specific_tensors:
            tensor_dict.update({
                'context_token_indices': torch.tensor(context_token_indices.astype(np.int64)),
                'context_token_mask': torch.tensor(context_mask, dtype=torch.float32),
                'column_token_to_column_id': torch.tensor(column_token_to_column_id.astype(np.int64)),
                'column_token_mask': torch.tensor(column_token_mask, dtype=torch.float32),
                'column_mask': torch.tensor(column_mask, dtype=torch.float32)
            })

        # for instance in instances:
        #     print(instance)

        return tensor_dict, instances

    def encode(
        self,
        contexts: List[List[str]],
        tables: List[Table],
        return_bert_encoding: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        tensor_dict, instances = self.to_tensor_dict(contexts, tables)
        device = next(self.parameters()).device

        for key in tensor_dict.keys():
            tensor_dict[key] = tensor_dict[key].to(device)

        context_encoding, column_encoding, encoding_info = self.encode_context_and_table(
            **tensor_dict,
            return_bert_encoding=return_bert_encoding
        )

        info = {
            'tensor_dict': tensor_dict,
            'instances': instances,
            **encoding_info
        }

        return context_encoding, column_encoding, info

    def validate(self, data_loader, args):
        was_training = self.training
        self.eval()

        logging_info_list = []
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc=f"Evaluation", file=sys.stdout) as pbar:
                for step, batch in enumerate(data_loader):
                    _, logging_info = self(**batch)
                    logging_info_list.append(logging_info)
                    pbar.update(1)

        if was_training:
            self.train()

        stats = {
            k: np.average([x[k] for x in logging_info_list])
            for k in logging_info_list[0]
        } if len(logging_info_list) > 0 else {}

        # handel distributed evaluation
        if args.multi_gpu:
            stats = distributed_utils.all_gather_list(stats)
            stats = {
                k: np.average([x[k] for x in stats])
                for k in stats[0]
            } if len(stats) > 0 else {}

        return stats

    def represent(self, data_loader, args):
        was_training = self.training
        self.eval()

        os.makedirs(args.output_file, exist_ok=True)
        config_file = os.path.join(args.output_file, f'args.json')
        with open(config_file, 'w') as fout:
            json.dump({'data_dir': str(args.data_dir)}, fout)
        output_file = os.path.join(args.output_file, f'repr.{args.global_rank}')
        context_li = []
        table_li = []

        span_overall = {
            'table': {'repr_li': [], 'index_li': [], 'text_li': []},
            'context': {'repr_li': [], 'index_li': [], 'text_li': []}
        }

        with torch.no_grad():
            with tqdm(total=len(data_loader), desc='Representing', file=sys.stdout) as pbar:
                for step, batch in enumerate(data_loader):
                    if args.index_repr == 'whole':
                        c, t = getattr(self, f'represent_{self.config.model_type.value}')(batch)
                        context_li.append(c.detach().cpu().numpy())
                        table_li.append(t.detach().cpu().numpy())
                    elif args.index_repr == 'whole_avg_cell':
                        c, t = getattr(self, f'represent_avg_cell_{self.config.model_type.value}')(batch)
                        context_li.append(c.detach().cpu().numpy())
                        table_li.append(t.detach().cpu().numpy())
                    elif args.index_repr in {'span_context', 'span_noncontext'}:
                        for field in ['table', 'context']:
                            repre, mask, bg2text = getattr(
                                self, f'represent_{args.index_repr}_{self.config.model_type.value}')(batch, field=field)
                            repre = repre.detach().cpu().numpy()
                            mask = mask.detach().cpu().numpy()
                            for b_idx, spans in enumerate(repre):
                                for c_idx, span in enumerate(spans):
                                    if not mask[b_idx, c_idx]:
                                        continue
                                    span_overall[field]['repr_li'].append(span)
                                    # use idx from the dataset builder as identity
                                    span_overall[field]['index_li'].append(batch['idx'][b_idx].item())
                                    span_overall[field]['text_li'].append(bg2text[b_idx, c_idx])
                    else:
                        raise NotImplementedError
                    pbar.update(1)

        if args.index_repr in {'whole', 'whole_avg_cell'}:
            context_li = np.concatenate(context_li, 0)
            table_li = np.concatenate(table_li, 0)
            np.savez(output_file, context=context_li, table=table_li)
        elif args.index_repr in {'span_context', 'span_noncontext'}:
            savekw = {f'{field}_{k}': np.array(span_overall[field][f'{k}_li']) for field in span_overall for k in ['repr', 'index', 'text']}
            np.savez(output_file, **savekw)
        else:
            raise NotImplementedError

        if was_training:
            self.train()

    def evaluate(self, data_loader, args):
        output_file = 'evaluation.jsonl' if args.output_file is None else args.output_file
        output_path = args.output_dir / output_file
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        was_training = self.training
        self.eval()

        with torch.no_grad(), open(output_path, 'w') as fout:
            with tqdm(total=len(data_loader), desc='Evaluation', file=sys.stdout) as pbar:
                for step, batch in enumerate(data_loader):
                    for result in getattr(self, f'evaluate_{self.config.model_type.value}')(batch):
                        fout.write(json.dumps(result) + '\n')
                    pbar.update(1)

        if was_training:
            self.train()

    def computeloss(self, data_loader, args):
        output_file = 'computeloss.jsonl' if args.output_file is None else args.output_file
        if args.multi_gpu:
            output_file += f'.{args.global_rank}'
        output_file = args.output_dir / output_file
        os.makedirs(os.path.dirname(str(output_file)), exist_ok=True)

        was_training = self.training
        self.eval()
        with torch.no_grad(), open(output_file, 'w') as fout:
            with tqdm(total=len(data_loader), desc='Compute Loss', file=sys.stdout) as pbar:
                for step, batch in enumerate(data_loader):
                    batch['return_per_example_loss'] = True
                    results = getattr(self, f'computeloss_{self.config.model_type.value}')(batch)
                    for b_idx, result in enumerate(results.detach().cpu().numpy()):
                        global_idx = batch['idx'][b_idx].item()
                        fout.write(f'{result}\t{global_idx}\n')
                    pbar.update(1)
        if was_training:
            self.train()

    def generate(self, data_loader, args):
        output_file = 'generation.tsv' if args.output_file is None else args.output_file
        if args.multi_gpu:
            output_file += f'.{args.global_rank}'
        output_path = args.output_dir / output_file
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        was_training = self.training
        self.eval()

        nrs = args.num_return_sequences
        sst = nrs > 1
        post_process = lambda x: self.tokenizer.decode(x, skip_special_tokens=sst).replace('\n', '\\n').replace('\t', '\\t')
        with torch.no_grad(), open(output_path, 'w') as fout:
            with tqdm(total=len(data_loader), desc='Generation', file=sys.stdout) as pbar:
                for step, batch in enumerate(data_loader):
                    if args.top_k is not None or args.top_p is not None:  # sampling
                        target_ids = self._bart.generate(
                            batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            num_return_sequences=nrs,
                            do_sample=True,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            min_length=args.min_generate_length,
                            max_length=args.max_generate_length,
                            early_stopping=True)
                    else:
                        target_ids = self._bart.generate(
                            batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            num_beams=args.num_beams,
                            num_return_sequences=nrs,
                            min_length=args.min_generate_length,
                            max_length=args.max_generate_length,
                            early_stopping=True)

                    if args.return_log_prob:
                      batch['return_per_example_loss'] = True
                      is_pad = target_ids.eq(self.tokenizer.pad_token_id).to(target_ids)
                      batch_for_loss = {
                        'idx': torch.repeat_interleave(batch['idx'], nrs, dim=0),
                        'input_ids': torch.repeat_interleave(batch['input_ids'], nrs, dim=0),
                        'attention_mask': torch.repeat_interleave(batch['attention_mask'], nrs, dim=0),
                        'token_type_ids': torch.repeat_interleave(batch['token_type_ids'], nrs, dim=0),
                        'masked_lm_labels': target_ids * (1 - is_pad) - is_pad,  # turn pad id into -1
                        'sample_size': batch['sample_size'],
                        'target_input_ids': target_ids,
                        'is_mlm': torch.repeat_interleave(batch['is_mlm'], nrs, dim=0),
                        'is_positives': torch.repeat_interleave(batch['is_positives'], nrs, dim=0),
                        'return_per_example_loss': True
                      }
                      logprobs = -getattr(self, f'computeloss_{self.config.model_type.value}')(batch_for_loss)  # add negative
                    gold_ids = batch['target_input_ids']
                    for b_idx, (input_id, gold_id) in enumerate(zip(batch['input_ids'], gold_ids)):
                        target_id = target_ids[b_idx * nrs:b_idx * nrs + nrs]
                        source: str = post_process(input_id)
                        preds: List[str] = [post_process(tis) for tis in target_id]
                        preds: str = '\t'.join(preds)
                        gold: str = post_process(gold_id)
                        global_idx = batch['idx'][b_idx].item()
                        if args.return_log_prob:
                          logprob = logprobs[b_idx * nrs:b_idx * nrs + nrs]
                          logprob: str = '\t'.join(map(lambda x: str(x.item()), logprob))
                          fout.write(f'{preds}\t{logprob}\t{gold}\t{source}\t{global_idx}\n')
                        else:
                          fout.write(f'{preds}\t{gold}\t{source}\t{global_idx}\n')
                    pbar.update(1)

        if was_training:
            self.train()
