from typing import List, Dict, Tuple, Set
from collections import defaultdict
from tqdm import tqdm
import os
import numpy as np
import torch
import faiss


class WholeFaiss(object):
    def __init__(self, repr_files: List[str], index_emb_size: int, normalize: bool = True):
        self.index_emb_size = index_emb_size
        self.normalize = normalize
        self.load_faiss(repr_files)

    @staticmethod
    def find_files(repr_file: str):
        repr_files: List[str] = []
        if os.path.exists(repr_file):
            repr_files.append(repr_file)
        else:
            i = 0
            while True:
                found = False
                for fm in [f'{repr_file}.{i}', f'{repr_file}.{i}.npz']:
                    if os.path.exists(fm):
                        repr_files.append(fm)
                        found = True
                        break
                if not found:
                    break
                i += 1
        return repr_files

    def load_faiss(self, repr_files: List[str]):
        context_li = []
        table_li = []
        for repr_file in repr_files:
            repr = np.load(repr_file)
            context_li.append(repr['context'].astype('float32'))
            table_li.append(repr['table'].astype('float32'))
        self.context_emb = np.concatenate(context_li)
        self.table_emb = np.concatenate(table_li)
        if self.normalize:
            self.context_emb = self.context_emb / (np.sqrt((self.context_emb * self.context_emb).sum(-1, keepdims=True)) + 1e-10)
            self.table_emb = self.table_emb / (np.sqrt((self.table_emb * self.table_emb).sum(-1, keepdims=True)) + 1e-10)
        self.emb_size = self.context_emb.shape[1]

    def interact(self, index_name: str, query_name: str, topk: int):
        index_emb = getattr(self, f'{index_name}_emb')
        query_emb = getattr(self, f'{query_name}_emb')
        index = faiss.IndexHNSWFlat(self.emb_size, self.index_emb_size, faiss.METRIC_INNER_PRODUCT)
        print(f'indexing {index_name} with shape {index_emb.shape} ...')
        index.add(index_emb)
        print(f'retrieving ...')
        score_matrix, ind_matrix = index.search(query_emb, topk)
        return score_matrix, ind_matrix


class SpanFaiss(object):
    def __init__(self, index_emb_size: int, cuda: bool):
        self.index_emb_size = index_emb_size
        self.cuda = cuda

    def convert_index(self, num_shards: int, total_count: int):
        raw_ind = list(range(total_count))
        rearranged_ind = []
        for s in range(num_shards):
            rearranged_ind.extend(raw_ind[s::num_shards])
        to_raw_ind = dict(zip(raw_ind, rearranged_ind))
        self.index_index = np.array([to_raw_ind[i] for i in self.index_index])
        self.query_index = np.array([to_raw_ind[i] for i in self.query_index])

    def load_span_faiss(self, repr_files: List[str], index_name: str, query_name: str, normalize: bool = True, reindex_shards: int = None):
        print('loading ...')
        index_emb_li = []
        index_index_li = []
        index_text_li = []
        query_emb_li = []
        query_index_li = []
        query_text_li = []
        for repr_file in repr_files:
            repr = np.load(repr_file, allow_pickle=True)
            index_emb_li.append(repr[f'{index_name}_repr'].astype('float32'))
            index_index_li.append(repr[f'{index_name}_index'])
            index_text_li.append(repr[f'{index_name}_text'])
            query_emb_li.append(repr[f'{query_name}_repr'].astype('float32'))
            query_index_li.append(repr[f'{query_name}_index'])
            query_text_li.append(repr[f'{query_name}_text'])
        self.index_emb = np.concatenate(index_emb_li)
        del index_emb_li[:]
        self.index_emb_size = self.index_emb.shape[1]
        self.index_index = np.concatenate(index_index_li)
        self.index_index_set = set(self.index_index)
        self.index_text = np.concatenate(index_text_li)
        self.query_emb = np.concatenate(query_emb_li)
        del query_emb_li[:]
        self.query_emb_size = self.query_emb.shape[1]
        self.query_index = np.concatenate(query_index_li)
        self.query_index_set = set(self.query_index)
        self.query_text = np.concatenate(query_text_li)
        if normalize:
            self.index_emb = self.index_emb / (np.sqrt((self.index_emb * self.index_emb).sum(-1, keepdims=True)) + 1e-10)
            self.query_emb = self.query_emb / (np.sqrt((self.query_emb * self.query_emb).sum(-1, keepdims=True)) + 1e-10)
        if reindex_shards:
            total_count = np.max(self.index_index) + 1
            assert total_count == 218419, '3merge data only!'
            self.convert_index(reindex_shards, total_count)

    def build_small_index(self, size: int):
        sample_inds = np.random.choice(self.index_emb.shape[0], size, replace=False)
        self.small_index_emb = self.index_emb[sample_inds]
        self.small_index_index = self.index_index[sample_inds]
        self.small_index_text = self.index_text[sample_inds]
        self.small_index = faiss.IndexHNSWFlat(self.small_index_emb.shape[1], self.index_emb_size, faiss.METRIC_INNER_PRODUCT)
        self.small_index.add(self.small_index_emb)

    def get_text_mask(self, text: List[str]):
        mask = [False if type(t) not in {str, np.str_} or t == '' else True for t in text]
        return mask

    def get_subset_query_emb(self, sub_index: List[int]):
        if len(set(sub_index) & self.query_index_set) <= 0:
            return {
                'emb': np.zeros((0, self.query_emb_size)),
                'index': np.zeros(0).astype(int),
                'text': np.zeros(0).astype(str)
            }
        mask = np.isin(self.query_index, sub_index)
        sub_query_emb = self.query_emb[mask]
        sub_query_index = self.query_index[mask]
        sub_query_text = self.query_text[mask]
        sub_mask = self.get_text_mask(sub_query_text)
        sub_query_emb = sub_query_emb[sub_mask]
        sub_query_index = sub_query_index[sub_mask]
        sub_query_text = sub_query_text[sub_mask]
        return {
            'emb': sub_query_emb,
            'index': sub_query_index,
            'text': sub_query_text
        }

    def query_from_subset(self, query_emb: np.ndarray, sub_index: List[int], topk: int = None, use_faiss: bool = True):
        if len(set(sub_index) & self.index_index_set) <= 0:
            return [{} for i in range(len(query_emb))]

        # select
        mask = np.isin(self.index_index, sub_index)
        sub_index_emb = self.index_emb[mask]
        sub_index_index = self.index_index[mask]
        sub_index_text = self.index_text[mask]
        sub_mask = self.get_text_mask(sub_index_text)
        sub_index_emb = sub_index_emb[sub_mask]
        sub_index_index = sub_index_index[sub_mask]
        sub_index_text = sub_index_text[sub_mask]

        if use_faiss:
            # index
            index = faiss.IndexHNSWFlat(sub_index_emb.shape[1], self.index_emb_size, faiss.METRIC_INNER_PRODUCT)
            index.add(sub_index_emb)
            # query
            score_matrix, ind_matrix = index.search(query_emb, topk)
        else:
            if self.cuda:
                query_emb = torch.tensor(query_emb).cuda()
                sub_index_emb = torch.tensor(sub_index_emb).cuda()
                score_matrix = query_emb @ sub_index_emb.T
                if topk is None:
                    topk = score_matrix.shape[1]
                _topk = min(topk, score_matrix.shape[1])
                score_matrix, ind_matrix = torch.topk(score_matrix, _topk, 1)
                score_matrix, ind_matrix = score_matrix.cpu().numpy(), ind_matrix.cpu().numpy()
            else:
                score_matrix = query_emb @ sub_index_emb.T
                ind_matrix = np.argsort(-score_matrix, 1)
                if topk:
                    ind_matrix = ind_matrix[:, :topk]
                score_matrix = np.take_along_axis(score_matrix, ind_matrix, 1)
        li_retid2textscore: List[Dict[int, List[Tuple[str, float]]]] = []
        for i in range(query_emb.shape[0]):
            li_retid2textscore.append(defaultdict(list))
            for id, score in zip(ind_matrix[i], score_matrix[i]):
                rid = sub_index_index[id]
                text = sub_index_text[id]
                li_retid2textscore[-1][rid].append((text, score))
        return li_retid2textscore

    def interact(self, topk: int, reverse: bool = False, after_agg_size: int = 0, max_topk_span: int = 100):
        if reverse:
            index_name, query_name = 'query', 'index'
        else:
            index_name, query_name = 'index', 'query'
        index_emb = getattr(self, f'{index_name}_emb')
        index_index = getattr(self, f'{index_name}_index')
        index_text = getattr(self, f'{index_name}_text')
        query_emb = getattr(self, f'{query_name}_emb')
        query_index = getattr(self, f'{query_name}_index')
        query_text = getattr(self, f'{query_name}_text')

        index_index2count: Dict[int, int] = dict(zip(*np.unique(index_index, return_counts=True)))
        query_index2count: Dict[int, int] = dict(zip(*np.unique(query_index, return_counts=True)))

        index = faiss.IndexHNSWFlat(index_emb.shape[1], self.index_emb_size, faiss.METRIC_INNER_PRODUCT)
        print(f'indexing with shape {index_emb.shape} ...')
        index.add(index_emb)

        print(f'retrieving ...')
        _topk = topk
        if after_agg_size:
            _topk = min(topk * 10, max_topk_span)  # assume that on avg a table has 10 cells retrieved
        score_matrix, ind_matrix = index.search(query_emb, _topk)
        if not after_agg_size:
            return score_matrix, ind_matrix

        print(f'aggregating ...')
        # sum all scores
        query_id2index_id2score: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(lambda: 0))
        query_id2index_id2count: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(lambda: 0))
        #query_id2index_id2textpairs: Dict[int, Dict[int, Set[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(set))
        for i, (scores, inds) in tqdm(enumerate(zip(score_matrix, ind_matrix))):
            qid = query_index[i]
            qtext = query_text[i]
            for score, ind in zip(scores, inds):
                iid = index_index[ind]
                itext = index_text[ind]
                query_id2index_id2score[qid][iid] += score
                query_id2index_id2count[qid][iid] += 1
                #query_id2index_id2textpairs[qid][iid].add((qtext, itext))

        counts: List[int] = []
        agg_score_matrix, agg_ind_matrix = np.zeros((after_agg_size, topk)), np.random.randint(after_agg_size, size=(after_agg_size, topk))

        sparsity_li: List[float] = []
        for qid, iid2score in query_id2index_id2score.items():
            iid2score = sorted(iid2score.items(), key=lambda x: -x[1])[:topk]
            counts.append(len(iid2score))
            iids, scores = zip(*iid2score)
            for iid in iids:
                spa = query_id2index_id2count[qid][iid] / ((query_index2count[qid] * index_index2count[iid]) or 1)
                sparsity_li.append(spa)
            agg_score_matrix[qid][:len(scores)] = scores
            agg_ind_matrix[qid][:len(iids)] = iids
        print(f'for {len(query_id2index_id2score)}/{after_agg_size} queries, '
              f'avg #retrieved items is {np.mean(counts)} with topk={topk}')
        print(f'retrieval sparsity is {np.mean(sparsity_li)}')
        return agg_score_matrix, agg_ind_matrix


class SpanFaissMulti(object):
    def __init__(self, index_emb_size: int, cuda: bool):
        self.index_emb_size = index_emb_size
        self.cuda = cuda

    def load_span_faiss(self, repr_file: str, index_name: str, query_name: str, normalize: bool = True, reindex_shards: int = None, merge: bool = False):
        if merge:
            self.faiss0 = SpanFaiss(self.index_emb_size, self.cuda)
            self.faiss0.load_span_faiss(WholeFaiss.find_files(repr_file), index_name=index_name, query_name=query_name, normalize=normalize, reindex_shards=reindex_shards)
            self.num_index = 1
        else:
            repr_files = WholeFaiss.find_files(repr_file)
            self.num_index = len(repr_files)
            for i, repr_file in enumerate(repr_files):
                setattr(self, f'faiss{i}', SpanFaiss(self.index_emb_size, self.cuda))
                getattr(self, f'faiss{i}').load_span_faiss([repr_file], index_name=index_name, query_name=query_name, normalize=normalize, reindex_shards=reindex_shards)

    def get_subset_query_emb(self, sub_index: List[int]):
        qds = []
        for i in range(self.num_index):
            fi = getattr(self, f'faiss{i}')
            qds.append(fi.get_subset_query_emb(sub_index))
        combined_qd = {k: np.concatenate([qd[k] for qd in qds], 0) for k in qds[0]}
        return combined_qd

    def query_from_subset(self, *args, **kwargs):
        results = []
        for i in range(self.num_index):
            fi = getattr(self, f'faiss{i}')
            results.append(fi.query_from_subset(*args, **kwargs))
        combined = results[0]
        for result in results[1:]:
            assert len(combined) == len(result)
            for i in range(len(combined)):
                for rid, tss in result[i].items():
                    combined[i][rid].extend(tss)
        return combined
