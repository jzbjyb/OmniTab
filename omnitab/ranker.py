from typing import List, Tuple, Dict
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from table_bert import TableBertModel, VanillaTableBert
from omnitab.spider import Spider


class Ranker(nn.Module):
    def __init__(self, model: TableBertModel, proj_size: int=512):
        nn.Module.__init__(self)
        self.model = model
        self.context_projection = nn.Parameter(torch.empty(model.output_size, proj_size), requires_grad=True)
        self.table_projection = nn.Parameter(torch.empty(model.output_size, proj_size), requires_grad=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)  # this init is crucial
        nn.init.normal_(self.context_projection, std=model.output_size ** -0.5)
        nn.init.normal_(self.table_projection, std=model.output_size ** -0.5)
        self.tokenizer = model.tokenizer
        self.loss_fc = RankLoss()

    def forward(self, tensor_dict: Dict, labels=None):
        # encode
        question_repr, column_repr, _ = self.model.encode_context_and_table(**tensor_dict,  return_bert_encoding=False)
        # avg
        question_repr = question_repr.mean(1)
        column_repr = column_repr.mean(1)
        # proj
        question_repr = question_repr @ self.context_projection
        column_repr = column_repr @ self.table_projection
        # norm
        question_repr = question_repr / question_repr.norm(dim=-1, keepdim=True)
        column_repr = column_repr / column_repr.norm(dim=-1, keepdim=True)
        # interaction
        logit_scale = self.logit_scale.exp()
        scores = logit_scale * (question_repr * column_repr).sum(-1)
        loss = None
        if labels is not None:
            loss = self.loss_fc(scores, labels)
        return loss, scores


class RankLoss(object):
    def __init__(self):
        self.pointwise = nn.BCEWithLogitsLoss(reduction='mean')

    def __call__(self,
                 scores,  # (batch_size, num_docs) or (batch_size, )
                 labels):  # (batch_size, num_docs) or (batch_size, )
        loss = self.pointwise(scores.view(-1), labels.view(-1).float())
        return loss


class RankSample(object):
    def __init__(self, question: str, table_with_labels: List[Tuple[str, str, int]]):
        self.question = question
        self.table_with_labels = table_with_labels

    @staticmethod
    def load_rank_samples(filename: str, group_by: int):
        samples = []
        prev_q = None
        dwl = []
        with open(filename, 'r') as fin:
            for i, l in enumerate(fin):
                question, db_id, table_name, label = l.strip().split('\t')
                if i % group_by != 0:
                    assert prev_q == question, 'this file "{}" is not grouped by questions'.format(question)
                dwl.append((db_id, table_name, int(label)))
                prev_q = question
                if i % group_by == group_by - 1:
                    samples.append(RankSample(prev_q, dwl))
                    dwl = []
        return samples


class RankDataset(Dataset):
    def __init__(self, sample_file: str, db_file: str, group_by: int, tokenizer):
        self.db2table2column = json.load(open(db_file, 'r'))
        self.samples: List[RankSample] = RankSample.load_rank_samples(sample_file, group_by=group_by)
        self.tokenizer = tokenizer
        self.preprocess('pointwise')

    def preprocess(self, method: str = 'pointwise'):
        return getattr(self, 'preprocess_{}'.format(method))()

    def preprocess_pointwise(self):
        self._examples: List[Tuple] = []
        cache_for_tables = {}
        for sample in self.samples:
            for db_id, table_name, label in sample.table_with_labels:
                q = self.tokenizer.tokenize(sample.question)
                key = '{}|||{}'.format(db_id, table_name)
                if key not in cache_for_tables:
                    table = Spider.convert_to_table(table_name, self.db2table2column[db_id][table_name])
                    cache_for_tables[key] = table.tokenize(self.tokenizer)
                t = cache_for_tables[key]
                self._examples.append((q, t, label))

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        return self._examples[item]

    @staticmethod
    def collate(examples: List, model: VanillaTableBert):
        contexts, tables, labels = list(zip(*examples))
        tensor_dict, _ = model.to_tensor_dict(contexts, tables)
        labels = torch.Tensor(labels).long()
        device = next(model.parameters()).device
        for key in tensor_dict.keys():
            tensor_dict[key] = tensor_dict[key].to(device)
        labels = labels.to(device)
        return tensor_dict, labels
