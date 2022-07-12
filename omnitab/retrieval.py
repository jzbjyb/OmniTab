from typing import List, Tuple, Set, Dict, Union
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import string
from typing import Iterator
import json
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
import subprocess
import copy
import sys
import numpy as np


class ESWrapper():
    MAX_QUERY_LEN = 1024
    PUNCT_TO_SPACE = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    def __init__(self, index_name: str=None, timeout: int=300):
        self.es = Elasticsearch(timeout=timeout)
        self.index_name = index_name

    def set_index_name(self, index_name: str):
        self.index_name = index_name

    def lucene_format(self, query_str: str, field: str):
        new_query_str = query_str.translate(self.PUNCT_TO_SPACE)
        new_query_str = new_query_str.replace(' AND ', ' ').replace(' and ', ' ')
        q = '{}:({})'.format(field, new_query_str[:self.MAX_QUERY_LEN])
        return q

    def get_topk(self, query_str: str, field: str, topk: int = 5):
        if len(query_str) <= 0:
            return []
        results = self.es.search(
            index=self.index_name,
            body={'query': {'match': {field: query_str[:self.MAX_QUERY_LEN]}}},
            size=topk)['hits']['hits']
        return [(doc['_source'], doc['_score']) for doc in results]

    def simple_build_index(self, doc_iter: Iterator):
      self.es.indices.delete(index=self.index_name, ignore=[400, 404])
      self.es.indices.create(index=self.index_name, ignore=400)
      for doc in doc_iter:
        assert '_source' in doc
        self.es.index(index=doc['_index'], body=doc['_source'], refresh=True)

    def build_index(self,
                    doc_iter: Iterator,
                    shards: int = 1,
                    replicas: int = 1,
                    verbose: bool = False,
                    refresh: bool = False):
        request_body = {
            'settings': {
                'number_of_shards': shards,
                'number_of_replicas': replicas
            }
        }
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        self.es.indices.create(index=self.index_name, ignore=400)
        e = bulk(self.es, doc_iter, refresh=refresh)
        if verbose:
            print(e)

    def delete_index(self):
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])

    @staticmethod
    def format_text(example):
        return ' '.join(example['context_before'] + example['context_after'])

    @staticmethod
    def format_table(table: Dict, full_table: bool = False):
        if full_table:
            header: str = ' | '.join([c['name'] for c in table['header']])
            data: List[str] = [' | '.join(row) for row in table['data']]
            return '\n'.join([header] + data)
        else:
            return ' & '.join(['{} | {} | {}'.format(
                h['name'], h['type'], h['sample_value']['value']) for h in table['header']])

    def table_text_data_iterator(self, filename: str, split: str = 'train', max_length: int = 10000, full_table: bool = False):
        with open(filename, 'r') as fin:
            for idx, l in tqdm(enumerate(fin)):
                example = json.loads(l)
                text = self.format_text(example)[:max_length]
                table = self.format_table(example['table'], full_table=full_table)[:max_length]
                yield {
                    '_index': self.index_name,
                    '_type': 'document',
                    'id': example['uuid'],
                    'line_idx': idx,
                    'split': split,
                    'text': text,
                    'table': table
                }

    def sentence_iterator(self, filename: str):
        with open(filename, 'r') as fin:
            for idx, l in tqdm(enumerate(fin)):
                l = l.strip()
                yield {
                    '_index': self.index_name,
                    '_type': 'document',
                    'line_idx': idx,
                    'text': l,
                }


def retrieve_part(index_name, filename, topk, full_table, cross_format, two_idx_and_outfile):  # inclusive and exclusive
    start_idx, end_idx, outfile = two_idx_and_outfile
    es = ESWrapper(index_name)
    format_list = lambda l: ' '.join(['{},{}'.format(i, s) for i, s in l])
    with open(filename, 'r') as fin, open(outfile, 'w') as fout:
        for idx, l in tqdm(enumerate(fin), total=end_idx - start_idx, disable=False):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break
            example = json.loads(l)
            text = ESWrapper.format_text(example)
            bytext = [(doc['line_idx'], score) for doc, score in es.get_topk(text, field='text', topk=topk + 1)]
            bytable = []
            if cross_format:  # use text to retrieve table
                bytable = [(doc['line_idx'], score) for doc, score in es.get_topk(text, field='table', topk=topk + 1)]
            elif full_table is not None:  # use table to retrieve table
                table = ESWrapper.format_table(example['table'], full_table=full_table)
                bytable = [(doc['line_idx'], score) for doc, score in es.get_topk(table, field='table', topk=topk + 1)]
            fout.write('{}\t{}\t{}\n'.format(idx, format_list(bytext), format_list(bytable)))


def retrieve(index_name: str, filename: str, output: str, topk: int = 5,
             threads: int = 1, rank: int = 0, world_size: int = 1,
             full_table: Union[bool, None] = False, cross_format: bool = False):
    total_count = int(subprocess.check_output('wc -l {}'.format(filename), shell=True).split()[0])
    bs = total_count // threads
    splits = [(i * bs, (i * bs + bs) if i < threads - 1 else total_count, f'{output}.{i}')
              for i in range(threads)][rank:threads:world_size]
    print('splits:', splits)
    real_threads = len(splits)

    pool = mp.Pool(real_threads)
    pool.map(partial(retrieve_part, index_name, filename, topk, full_table, cross_format), splits)


def fill_to_at_middle(lst: List, fill_to: int):
    if len(lst) == 0:
        raise ValueError()
    if len(lst) > fill_to:
        lst = lst[:fill_to]
    if len(lst) == fill_to:
        return lst
    mid = min(len(lst) // 2 + len(lst) % 2, len(lst))
    lst = lst[:mid] + ([lst[mid - 1]] * (fill_to - len(lst))) + lst[mid:]
    return lst


def load_neg_file(neg_file: str, fill_to: int = None):
    idx2neg: Dict[int, Tuple[List, List]] = {}
    with open(neg_file, 'r') as fin:
        for l in fin:
            idx, bytext, bytable = l.strip().split('\t')
            idx = int(idx)
            # only get idx
            bytext = [int(s.split(',')[0]) for s in bytext.split(' ')]
            bytable = [int(s.split(',')[0]) for s in bytable.split(' ')]
            bytext = [i for i in bytext if i != idx]
            bytable = [i for i in bytable if i != idx]
            if fill_to:
                bytext = fill_to_at_middle(bytext, fill_to=fill_to)
                bytable = fill_to_at_middle(bytable, fill_to=fill_to)
            idx2neg[idx] = (bytext, bytable)
    return idx2neg


def get_from_merge(lst: List[Tuple], count: int, from_bottom: bool = False, filter_set: Set = set()):
    cur = len(lst) - 1 if from_bottom else 0
    off = -1 if from_bottom else 1
    result = set()
    while 0 <= cur < len(lst) and len(result) < count:
        for i in lst[cur]:
            if i in filter_set:
                continue
            result.add(i)
            if len(result) >= count:
                break
        cur += off
    assert len(result) == count
    return list(result)


def combine_negative(data_file: str, neg_file: str, output: str, fill_to: int, num_top_neg: int, num_bottom_neg: int,
                     num_random_neg: int):
    idx2neg = load_neg_file(neg_file, fill_to=fill_to)
    idx2example: Dict[int, Dict] = {}
    with open(data_file, 'r') as fin:
        for idx, l in enumerate(fin):
            idx2example[idx] = json.loads(l)
            idx2example[idx]['is_positive'] = True
    total_num = len(idx2example)
    with open(output, 'w') as fout:
        for idx in tqdm(range(total_num)):
            # pos
            fout.write(json.dumps(idx2example[idx]) + '\n')

            # neg
            bytext, bytable = idx2neg[idx]
            merge = list(zip(bytext, bytable))

            negs = []
            negs += get_from_merge(merge, count=num_top_neg)
            negs += get_from_merge(merge, count=num_bottom_neg, from_bottom=True, filter_set=set(negs))
            rand_negs = np.random.choice(total_num, num_random_neg + len(negs) + 1, replace=False)  # avoid collision
            sn = set(negs)
            rand_negs = [i for i in rand_negs if i != idx and i not in sn][:num_random_neg]
            negs += rand_negs
            assert len(negs) == len(set(negs)), 'different neg methods have collision'
            assert len(negs) == (num_top_neg + num_bottom_neg + num_random_neg), '#neg not enough'

            neg_example = copy.deepcopy(idx2example[idx])
            neg_example['is_positive'] = False
            for i in negs:
                neg_example['uuid'] = idx2example[idx]['uuid'] + '_{}-{}'.format(idx, i)
                neg_example['table'] = idx2example[i]['table']  # replace table
                fout.write(json.dumps(neg_example) + '\n')


def fake_example(target_file: str, ret_file: str, source_file: str, out_file: str, topk: int, dedup: bool = False):
    id2sent = {}
    with open(source_file, 'r') as fin:
        for i, l in enumerate(fin):
            id2sent[i] = l.strip()
    used_ids: Set[int] = set()
    with open(target_file, 'r') as tfin, open(ret_file, 'r') as rfin, open(out_file, 'w') as fout:
        for l in tfin:
            example = json.loads(l)
            example_sent = example['context_before'][0]
            _, bytext = rfin.readline().strip().split('\t')
            count = 0
            for i, idx in enumerate([int(b.split(',')[0]) for b in bytext.split(' ')]):
                sent = id2sent[idx]
                if idx in used_ids:
                    continue
                if sent.lower() == example_sent.lower():
                    continue
                td = {
                    'uuid': f"fakewikisent_{example['uuid']}_{idx}",
                    'table': {'caption': '', 'header': [], 'data': [], 'data_used': [], 'used_header': []},
                    'context_before': [sent],
                    'context_before_mentions': [],
                    'context_after': []
                }
                fout.write(json.dumps(td) + '\n')
                count += 1
                used_ids.add(idx)
                if count >= topk:
                    break


if __name__ == '__main__':
    task = sys.argv[1].split('-')
    use_home = False
    home = '/home/zhengbao' if use_home else ''
    merge3_filename = f'{home}/mnt/root/TaBERT/data/grappa/totto_tablefact_wikisql_train_preprocessed_mention.jsonl'

    if task[0] == 'totto':
        # index and search
        index_name = 'totto'
        filename = 'data/totto_data/train/preprocessed.jsonl'
        neg_file = 'test.tsv'
        topk = 100
        es = ESWrapper(index_name)
        es.build_index(es.table_text_data_iterator(filename))
        retrieve(index_name, filename, neg_file, topk=topk, threads=10)

        # get negative
        final_file = 'test.jsonl'
        combine_negative(filename, neg_file, final_file, topk, 3, 3, 3)

    elif task[0] == '3merge':  # TOTTO, TableFact, WikiSQL
        index_name = '3merge'
        es = ESWrapper(index_name)
        if task[1] == 'index':
            es.build_index(es.table_text_data_iterator(merge3_filename, full_table=True))
        elif task[1] == 'ret':
            topk = 100
            retrieve_output = merge3_filename + f'.ret{topk}'
            retrieve(index_name, merge3_filename, retrieve_output, topk=topk, threads=10, full_table=True)

    elif task[0] == 'tapas_data0':
        index_name = 'tapas_data0'
        es = ESWrapper(index_name)
        if 'index' in task[1]:
            filename = f'{home}/mnt/root/tapas/data/pretrain/train/preprocessed_mention.jsonl.data0'
            es.build_index(es.table_text_data_iterator(filename, full_table=True), shards=5)
        if 'ret' in task[1]:
            topk = 100
            threads, rank, world_size = sys.argv[2:5]
            threads, rank, world_size = int(threads), int(rank), int(world_size)
            filename = f'{home}/mnt/root/tapas/data/pretrain/train/preprocessed_mention.jsonl.data0'
            retrieve_output = filename + f'.{index_name}_ret{topk}'
            retrieve(index_name, filename, retrieve_output, topk=topk,
                     threads=threads, rank=rank, world_size=world_size, full_table=True)

    elif task[0] == 'tapas':
        index_name = 'tapas'
        es = ESWrapper(index_name)
        if 'index' in task[1]:
            filename = f'{home}/mnt/root/tapas/data/pretrain/train/preprocessed.jsonl'
            es.build_index(es.table_text_data_iterator(filename, full_table=True), shards=5)
        if 'ret' in task[1]:
            topk = 10
            threads, rank, world_size = sys.argv[2:5]
            threads, rank, world_size = int(threads), int(rank), int(world_size)
            filename = f'{home}/mnt/root/tapas/data/pretrain/train/preprocessed.jsonl'
            retrieve_output = filename + f'.{index_name}_ret{topk}'
            retrieve(index_name, filename, retrieve_output, topk=topk,
                     threads=threads, rank=rank, world_size=world_size, full_table=True)

    elif task[0] == 'wikipedia':
        index_name = 'wikipedia_sent'
        es = ESWrapper(index_name)
        topk = 100
        wiki_filename = f'{home}/mnt/root/TaBERT/data/wikipedia_sent/wikisent2.txt'
        wiki_ret_filename = merge3_filename + f'.{index_name}_ret{topk}'
        wiki_fake_example_filename = f'{home}/mnt/root/TaBERT/data/wikipedia_sent/wikisent2_3merge_dedup.jsonl'
        if 'index' in task[1]:
            es.build_index(es.sentence_iterator(wiki_filename), shards=5)
        if 'ret' in task[1]:
            retrieve(index_name, merge3_filename, wiki_ret_filename, topk=topk, threads=10, full_table=None)
        if 'fake' in task[1]:
            fake_example(merge3_filename, wiki_ret_filename, wiki_filename, wiki_fake_example_filename, topk=10, dedup=True)
        if 'extend' in task[1]:
            index_name = 'tapas'
            es = ESWrapper(index_name)
            topk = 10
            threads, rank, world_size = sys.argv[2:5]
            threads, rank, world_size = int(threads), int(rank), int(world_size)
            retrieve_output = wiki_fake_example_filename + f'.{index_name}_ret{topk}'
            retrieve(index_name, wiki_fake_example_filename, retrieve_output, topk=topk,
                     threads=threads, rank=rank, world_size=world_size, full_table=None, cross_format=True)
