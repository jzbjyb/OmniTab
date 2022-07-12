#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Callable, Any
import logging
from enum import Enum
import numpy as np
import os
from multiprocessing import Queue, Process
import time


class TransformerVersion(Enum):
    PYTORCH_PRETRAINED_BERT = 0
    TRANSFORMERS = 1


TRANSFORMER_VERSION = None


try:
    if 'USE_TRANSFORMER' in os.environ:
        logging.warning('force to use the new version')
        raise ImportError
    from pytorch_pretrained_bert.modeling import (
        BertForMaskedLM, BertForPreTraining, BertModel,
        BertConfig,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead, BertLayerNorm, gelu
    )
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    class BertTokenizerWrapper(BertTokenizer):
        def convert_ids_to_tokens(self, ids):
            tokens = []
            for i in ids:
                if i < 0:
                    tokens.append(None)
                else:
                    tokens.append(self.ids_to_tokens[i])
            return tokens

    hf_flag = 'old'
    TRANSFORMER_VERSION = TransformerVersion.PYTORCH_PRETRAINED_BERT
    logging.warning('You are using the old version of `pytorch_pretrained_bert`')
except ImportError:
    from transformers.tokenization_bert import BertTokenizer    # noqa
    BertTokenizerWrapper = BertTokenizer
    from transformers.modeling_bert import (    # noqa
        BertForMaskedLM, BertForPreTraining, BertModel,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead
    )
    try:
        from transformers.modeling_bert import BertLayerNorm, gelu
    except ImportError:
        from pytorch_pretrained_bert.modeling import BertLayerNorm, gelu
    from transformers.configuration_bert import BertConfig  # noqa

    hf_flag = 'new'
    TRANSFORMER_VERSION = TransformerVersion.TRANSFORMERS

# BERT
from transformers import BertTokenizerFast

# ELECTRA
from transformers import ElectraConfig, ElectraTokenizer, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining

# RoBERTa
# TODO: set add_prefix_space to True?
from transformers import RobertaConfig, RobertaTokenizer, RobertaTokenizerFast, RobertaForMaskedLM

# BART
from transformers import BartConfig, BartTokenizer, BartTokenizerFast, BartForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right


def compute_mrr(scores: List[float], labels: List[int]):
    return np.mean([1 / (i + 1) for i, (s, r) in enumerate(sorted(zip(scores, labels), key=lambda x: -x[0])) if r])


def get_url(text: str):
  return text[text.find('http'):].rsplit('_', 1)[0]  # remove the suffix number


class BartTokenizerWrapper(BartTokenizer):
    def tokenize(self, *args, **kwargs):
        if 'add_prefix_space' in kwargs:
            del kwargs['add_prefix_space']
        return super(BartTokenizerWrapper, self).tokenize(*args, **kwargs, add_prefix_space=True)  # always add space

class BartTokenizerFastWrapper(BartTokenizerFast):
    def __call__(self, *args, **kwargs):
        if type(args[0]) is not str:
            raise NotImplementedError
        if len(args[0]) > 0 and args[0][0] != ' ':
            args = (' ' + args[0],) + args[1:]  # add_prefix_space
            result = super(BartTokenizerFastWrapper, self).__call__(*args, **kwargs)
            result['added_prefix_space'] = True
        else:
            result = super(BartTokenizerFastWrapper, self).__call__(*args, **kwargs)
        return result


class MultiprocessWrapper(object):
  def __init__(self,
               num_threads: int,
               worker: Callable,
               writer: Callable,
               output_file: str,
               batch_size: int = 1,
               use_gpu: bool = False):
    self.num_threads = num_threads
    self.batch_size = batch_size
    self.use_gpu = use_gpu

    # start processes
    self.input_queue = Queue()
    self.output_queue = Queue()
    self.processes = []
    for pid in range(num_threads):
      if use_gpu:
        p = Process(target=worker, args=(self.input_queue, self.output_queue, pid))
      else:
        p = Process(target=worker, args=(self.input_queue, self.output_queue))
      p.daemon = True
      p.start()
      self.processes.append(p)
    self.write_p = Process(target=writer, args=(output_file, self.output_queue))
    self.write_p.start()

    self.batch: List[Any] = []

  def add_example(self, example: Any):
    self.batch.append(example)
    if len(self.batch) >= self.batch_size:
      self.input_queue.put(self.batch)
      self.batch = []

  def finish(self):
    if len(self.batch) > 0:  # check if there is any remaining examples
      self.input_queue.put(self.batch)
    for _ in self.processes:
      self.input_queue.put('DONE')
    for p in self.processes:
      p.join()
    self.output_queue.put('DONE')
    self.write_p.join()
