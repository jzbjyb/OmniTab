import logging
import os
os.environ['WANDB_API_KEY'] = '9caada2c257feff1b6e6a519ad378be3994bc06a'

from typing import List, Dict, Tuple, Set
import functools
import time
from argparse import ArgumentParser
from pathlib import Path
import os
import random
import numpy as np
import json
import copy
from collections import defaultdict
from tqdm import tqdm
import multiprocessing
import itertools
import logging
from timeout_decorator.timeout_decorator import TimeoutError
import faiss
import spacy
from omnitab.totto import Totto
from omnitab.wikisql import WikiSQL
from omnitab.tablefact import TableFact
from omnitab.wikitablequestions import WikiTQ
from omnitab.turl import TurlData
from omnitab.tapas import TapasTables
from omnitab.tapex import Tapex
from omnitab.dataset_utils import BasicDataset
from omnitab.dataset import Example
from omnitab.faiss_utils import SpanFaiss, SpanFaissMulti, WholeFaiss


def tableshuffle(prep_file: str, output_file: str):
    with open(prep_file, 'r') as fin, open(output_file, 'w') as fout:
        for l in tqdm(fin):
            example = json.loads(l)
            Example.shuffle_table(example)
            fout.write(json.dumps(example) + '\n')


def find_other_table(prep_file: str, output_file: str, max_count: int):
    tablecell2ind: Dict[str, List] = defaultdict(list)
    examples: List[Dict] = []
    with open(prep_file, 'r') as fin:
        for eid, l in tqdm(enumerate(fin), desc='build tablecell2ind'):
            example = json.loads(l)
            examples.append(example)
            for row in example['table']['data']:
                for cell in row:
                    cell = cell.strip().lower()
                    if len(tablecell2ind[cell]) > 0 and tablecell2ind[cell][-1] == eid:
                        continue
                    tablecell2ind[cell].append(eid)
    match_counts = []
    used_eids: Set[int] = set()
    with open(output_file, 'w') as fout:
        for eid, example in tqdm(enumerate(examples), desc='find tables'):
            context = example['context_before'][0]
            eid2mentions: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
            all_mentions: Set[Tuple[int, int]] = set()
            for s, e in example['context_before_mentions'][0]:
                all_mentions.add((s, e))
                kw = context[s:e].strip().lower()
                for _eid in random.sample(tablecell2ind[kw], min(len(tablecell2ind[kw]), 1000)):
                    if _eid == eid:
                        continue
                    eid2mentions[_eid].add((s, e))
            if len(eid2mentions) > 0:
                match_eid, mentions = max(eid2mentions.items(), key=lambda x: len(x[1]))
            else:
                match_eid, mentions = (random.randint(0, len(examples) - 1), set())
            used_eids.add(match_eid)
            ne = copy.deepcopy(example)
            ne['table'] = examples[match_eid]['table']
            ne['context_before_mentions'] = [list(mentions) + list(all_mentions - mentions)]
            assert len(ne['context_before_mentions'][0]) == len(all_mentions), f"{ne['context_before_mentions'][0]} {all_mentions}"
            fout.write(f'{json.dumps(ne)}\n')
            match_counts.append(min(max_count, len(mentions)))
        print(np.mean(match_counts))
        print(f'#used_eids {len(used_eids)}')


def _generate_retrieval_data_single(example_lines: List[str], ret_examples_li: List[List[Dict]],
                                    bywhich: str, max_context_len: int, max_num_rows: int, batch_id: int = None, op: str = 'max'):
    assert op in {'max', 'min'}
    examples = []
    for i, (example_line, ret_examples) in enumerate(zip(example_lines, ret_examples_li)):
        example = json.loads(example_line)
        best_match_mentions = None
        best_match_mentions_cells = None
        best_match = None
        for _example in ret_examples:
            if bywhich == 'context':
                context = example['context_before'][0]
                table = _example['table']['data']
            elif bywhich == 'table':
                context = _example['context_before'][0]
                table = example['table']['data']
            else:
                raise NotImplementedError
            if max_context_len:
                context = context[:max_context_len]
            if max_num_rows:
                table = table[:max_num_rows]
            try:
                locations, location2cells = BasicDataset.get_mention_locations(context, table)
                mention_cells: List[List[Tuple[int, int]]] = [location2cells[ml] for ml in locations]
            except TimeoutError:
                print(f'timeout {context} {table}')
                locations = []
                mention_cells = []
            if best_match_mentions is None or \
                    (op == 'max' and len(locations) > len(best_match_mentions)) or \
                    (op == 'min' and len(locations) < len(best_match_mentions)):
                best_match_mentions = locations
                best_match_mentions_cells = mention_cells
                best_match = _example
        if best_match is not None:
            data_used = sorted(list(set(mc for mcs in best_match_mentions_cells for mc in mcs)))
            if bywhich == 'context':
                example['table'] = best_match['table']
                example['table']['data_used'] = data_used
                example['context_before_mentions'] = [best_match_mentions]
                example['context_before_mentions_cells'] = [best_match_mentions_cells]
                if op == 'min':  example['is_positive'] = False
            elif bywhich == 'table':
                example['table']['data_used'] = data_used
                example['context_before'] = best_match['context_before']
                example['context_before_mentions'] = [best_match_mentions]
                example['context_before_mentions_cells'] = [best_match_mentions_cells]
                if op == 'min':  example['is_positive'] = False
        examples.append(example)  # use the original if there is no retrieved examples
    print(f'batch {batch_id} completed')
    return examples


class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def output_example(results, fout, batch_id, timeout, num_mentions):
    _timeout = timeout
    for r in results:
        try:
            for e in r.get(_timeout):
                num_mentions.append(len(e['context_before_mentions'][0]))
                fout.write(json.dumps(e) + '\n')
        except multiprocessing.TimeoutError:
            _timeout = max(_timeout // 4, 60)
            logging.warning(f'batch {batch_id} timeout')


def generate_retrieval_data(retrieval_file: str, target_file: str, source_file: str, output_file: str,
                            bywhich: str, topk: int = 0, botk: int = 0, nthread: int = 1, batch_size: int = 100,
                            max_context_len: int = None, max_num_rows: int = None,
                            remove_self: bool = False, only_self: bool = False,
                            timeout: float = None, use_top1: str = None, op: str = 'max'):
    assert bywhich in {'context', 'table'}
    assert topk or botk, 'either topk or botk must be specified'
    idx2example: Dict[int, Dict] = {}
    with open(source_file, 'r') as fin:
        for idx, l in tqdm(enumerate(fin), desc='build index'):
            idx2example[idx] = json.loads(l)
    #pool = MyPool(processes=nthread)
    pool = multiprocessing.Pool(processes=nthread)
    start = time.time()
    batch_id = 0
    num_mentions: List[int] = []
    with open(retrieval_file, 'r') as fin, open(target_file, 'r') as tfin, open(output_file, 'w') as fout:
        example_lines = []
        ret_examples_li = []
        results = []
        tfin_idx = -1
        def get_next_target_until(idx):
            nonlocal tfin_idx
            l = None
            while tfin_idx < idx:
                l = tfin.readline()
                tfin_idx += 1
            return l
        for l in tqdm(fin, miniters=50):
            idx, bytext, bytable = l.rstrip('\n').split('\t')
            idx = int(idx)
            if only_self:
                byall = [idx]
            else:
                if topk:
                    bytext = [int(s.split(',')[0]) for s in bytext.split(' ') if len(s) > 0][:topk + 1]
                    bytable = [int(s.split(',')[0]) for s in bytable.split(' ') if len(s) > 0][:topk + 1]
                elif botk:
                    bytext = [int(s.split(',')[0]) for s in bytext.split(' ') if len(s) > 0][-botk - 1:]
                    bytable = [int(s.split(',')[0]) for s in bytable.split(' ') if len(s) > 0][-botk - 1:]
                if use_top1 == 'context':
                    byall = bytext[:1]
                    if remove_self and idx in byall:
                        byall = bytext[1:2]
                elif use_top1 == 'table':
                    byall = bytable[:1]
                    if remove_self and idx in byall:
                        byall = bytable[1:2]
                elif use_top1 is None:
                    if topk:
                        byall = list(set(bytext + bytable) - ({idx} if remove_self else set()))[:2 * topk]
                    elif botk:
                        byall = list(set(bytext + bytable) - ({idx} if remove_self else set()))[-2 * botk:]
                else:
                    raise NotImplementedError
            ret_examples = [idx2example[_idx] for _idx in byall]
            example_line = get_next_target_until(idx)
            ret_examples_li.append(ret_examples)
            example_lines.append(example_line)
            if len(example_lines) >= batch_size:
                r = pool.apply_async(
                    functools.partial(_generate_retrieval_data_single,
                                      bywhich=bywhich, max_context_len=max_context_len, max_num_rows=max_num_rows, batch_id=batch_id, op=op),
                    (example_lines, ret_examples_li))
                results.append(r)
                example_lines = []
                ret_examples_li = []
                if len(results) == nthread:
                    output_example(results, fout, batch_id, timeout, num_mentions)
                    results = []
                    batch_id += 1
        if len(example_lines) >= 0:
            r = pool.apply_async(
                functools.partial(_generate_retrieval_data_single,
                                  bywhich=bywhich, max_context_len=max_context_len, max_num_rows=max_num_rows, batch_id=batch_id, op=op),
                (example_lines, ret_examples_li))
            results.append(r)
        if len(results) > 0:
            output_example(results, fout, batch_id, timeout, num_mentions)
    end = time.time()
    print(f'total time {end - start}, with avg #mentions {np.mean(num_mentions)}')


def generate_random_neg(input_file: str, output_file: str, num_neg: int = 1):
    examples = []
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for l in fin:
            e = json.loads(l)
            examples.append(e)
        inds = list(range(len(examples)))
        for i, e in enumerate(examples):
            for j in list(set(random.sample(inds, num_neg + 1)) - {i})[:num_neg]:
                ne = copy.deepcopy(e)
                ne['table'] = examples[j]['table']
                ne['is_positive'] = False
                fout.write(json.dumps(ne) + '\n')


def compute_ret_mrr(filename: str):
    text_mrrs = []
    table_mrrs = []
    with open(filename, 'r') as fin:
        for l in fin:
            idx, bytext, bytable = l.rstrip('\n').split('\t')
            for by, mrrs in [(bytext, text_mrrs), (bytable, table_mrrs)]:
                mrr = 0
                for i, t in enumerate(by.split(' ')):
                    if t.split(',')[0] == idx:
                        mrr = 1 / (i + 1)
                        break
                mrrs.append(mrr)
        print(f'text mrr {np.mean(text_mrrs)}, table mrr {np.mean(table_mrrs)}')


def filter_mention(input_file: str, out_file: str, topk: int = 0, sort: bool = False):
    examples = []
    nms = []
    with open(input_file, 'r') as fin, open(out_file, 'w') as fout:
        for i, l in enumerate(fin):
            examples.append(l)
            nms.append(len(json.loads(l)['context_before_mentions'][0]))
        if sort:
            print('sort')
            rank = np.argsort(-np.array(nms))
        else:
            rank = list(range(len(examples)))
        if topk:
            rank = rank[:topk]
        for i in rank:
            fout.write(examples[i])
        print(f'mean # {np.mean([nms[i] for i in rank])}')


def whole_faiss(repr_file: str, output_file: str, index_emb_size: int, topk: int = 10, use_span: bool = False):
    repr_files: List[str] = WholeFaiss.find_files(repr_file)
    print(f'load embeddings from {repr_files}')

    if use_span:
        faiss_wrap = SpanFaiss(index_emb_size=index_emb_size, cuda=False)
        faiss_wrap.load_span_faiss(repr_files, index_name='table', query_name='context', reindex_shards=None)
    else:
        faiss_wrap = WholeFaiss(repr_files, index_emb_size=index_emb_size)
    ret_results = []
    for index_name, query_name in [('table', 'context'), ('context', 'table')]:  # add 1 for self retrieval
        if use_span:
            # TODO: this is only for tapas data0
            total_num_doc = 904370
            score_matrix, ind_matrix = faiss_wrap.interact(topk + 1, reverse=index_name == 'context', after_agg_size=total_num_doc)
        else:
            score_matrix, ind_matrix = faiss_wrap.interact(index_name, query_name, topk + 1)
        ret_results.append((ind_matrix, score_matrix))
    format_list = lambda inds, scores: ' '.join(['{},{}'.format(i, s) for i, s in zip(inds, scores)])
    with open(output_file, 'w') as fout:
        for idx, (bycontext_inds, bycontext_scores, bytable_inds, bytable_scores) in enumerate(
          zip(*(ret_results[0] + ret_results[1]))):
            fout.write('{}\t{}\t{}\n'.format(
                idx, format_list(bycontext_inds, bycontext_scores), format_list(bytable_inds, bytable_scores)))


def span_faiss(repr_file: str, ouput_file: str, topk: int, index_subsample: int = None, index_emb_size: int = 512):
    print('loading ...')
    findex = SpanFaiss(index_emb_size=index_emb_size, cuda=True)
    findex.load_span_faiss(repr_file, index_name='table', query_name='context', reindex_shards=10)
    findex.build_small_index(index_subsample)
    queryidx2retscores: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    # query
    print(f'retrieving ...')
    '''
    for i in range(query_emb.shape[0]):
        score_matrix, ind_matrix = index.search(query_emb[i:i+1], topk + 1)
        print('--->', query_text[i])
        for j in I[0]:
            print(index_text[j])
        input()
    '''
    score_matrix, ind_matrix = findex.small_index.search(findex.query_emb, topk + 1)  # add 1 for self retrieval
    print('retrieving done')

    print('aggregate scores')
    for i in range(findex.query_emb.shape[0]):
        qid = findex.query_index[i]
        for id, score in zip(ind_matrix[i], score_matrix[i]):
            rid = findex.small_index_index[id]
            queryidx2retscores[qid][rid].append(score)
    print(f'max query id is {np.max(list(queryidx2retscores.keys()))}')

    print('output')
    format_list = lambda indscores: ' '.join(['{},{}'.format(i, s) for i, s in indscores])
    with open(ouput_file, 'w') as fout:
        for qid in sorted(queryidx2retscores.keys()):
            rid2scores = queryidx2retscores[qid]
            rid2sumscore = sorted([(rid, np.sum(scores)) for rid, scores in rid2scores.items()], key=lambda x: -x[1])[:topk]
            fout.write('{}\t{}\t\n'.format(qid, format_list(rid2sumscore)))


def ret_filter_by_faiss(repr_file, ret_file: str, target_file: str, source_file: str, output_file: str,
                        topk: int, index_emb_size: int, agg: str = 'max-sum', batch_size: int = 1,
                        use_str_match: bool = False, separate: bool = False, skip_noret: bool = False):
    #findex = FaissUtils(index_emb_size=index_emb_size, cuda=True)
    #findex.load_span_faiss(repr_file, index_name='table', query_name='context', reindex_shards=10)
    agg = tuple(agg.split('-'))
    assert agg in set(itertools.product(['max', 'sum'], ['sum', 'count', 'avg_count']))
    findex = SpanFaissMulti(index_emb_size=index_emb_size, cuda=True)
    findex.load_span_faiss(str(repr_file), index_name='table', query_name='context', reindex_shards=None, merge=True)
    idx2example: Dict[int, Dict] = {}
    with open(source_file, 'r') as fin:
        for idx, l in tqdm(enumerate(fin), desc='build map'):
            idx2example[idx] = json.loads(l)

    def match_mention(t1, t2):
        return ''.join(t1.lower().split()) == ''.join(t2.lower().split())

    with open(ret_file, 'r') as rfin, open(target_file, 'r') as tfin, open(output_file, 'w') as fout:
        tfin_idx = -1
        def get_next_target_until(idx):
            nonlocal tfin_idx
            l = None
            while tfin_idx < idx:
                l = tfin.readline()
                tfin_idx += 1
            return l
        idx2byall: Dict[int, Set[int]] = {}
        idx_li: List[int] = []
        byall_li: List[int] = []
        num_mention_li: List[int] = []
        num_mention_li_before: List[int] = []
        with tqdm(miniters=50) as pbar:
            for l in rfin:
                pbar.update(1)
                pbar.set_postfix_str(f'{np.mean(num_mention_li)} before {np.mean(num_mention_li_before)}')
                idx, bytext, bytable = l.rstrip('\n').split('\t')
                idx = int(idx)
                bytext = [int(s.split(',')[0]) for s in bytext.split(' ') if len(s) > 0]
                bytable = [int(s.split(',')[0]) for s in bytable.split(' ') if len(s) > 0]
                byall = list(set(bytext + bytable) - {idx})  # always remove self
                idx_li.append(idx)
                byall_li.extend(byall)
                idx2byall[idx] = set(byall)
                if len(idx_li) >= batch_size:
                    query_dict = findex.get_subset_query_emb(idx_li)
                    li_retid2textscore = findex.query_from_subset(query_dict['emb'], byall_li, topk=topk, use_faiss=False)
                    # visualize
                    #for qtext, r2ts in zip(query_dict['text'], li_retid2textscore):
                    #    print(qtext)
                    #    print(set(t for r, tss in r2ts.items() for t, s in tss))
                    #input()
                    if separate:
                        qid2qtext_rid2score: Dict[int, List[Tuple[str, Dict[int, float]]]] = defaultdict(list)
                        for qid, qtext, r2ts in zip(query_dict['index'], query_dict['text'], li_retid2textscore):
                            qid2qtext_rid2score[qid].append((qtext, {}))
                            for rid, tss in r2ts.items():
                                qid2qtext_rid2score[qid][-1][1][rid] = np.sum([s for t, s in tss])
                        for qid in sorted(qid2qtext_rid2score.keys()):
                            raw_example = json.loads(get_next_target_until(qid))
                            for qtext, rid2score in qid2qtext_rid2score[qid]:
                                rid = sorted(rid2score.items(), key=lambda x: -x[1])[0][0]
                                ret_example = idx2example[rid]
                                example = copy.deepcopy(raw_example)
                                context = example['context_before'][0]
                                example['table'] = ret_example['table']
                                example['context_before_mentions'] = [ml for ml in raw_example['context_before_mentions'][0] if match_mention(qtext, context[ml[0]:ml[1]])]
                                # only for visualizing
                                locations, _ = BasicDataset.get_mention_locations(example['context_before'][0], example['table']['data'])
                                num_mention_li.append(len(locations))
                                fout.write(json.dumps(example) + '\n')
                    else:
                        # find the one with the most scores
                        qid2rid2scores: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
                        qid2rid2text2score: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(
                            lambda: defaultdict(lambda: defaultdict(lambda: 0)))
                        for qid, r2ts in zip(query_dict['index'], li_retid2textscore):
                            for rid, tss in r2ts.items():
                                if rid not in idx2byall[qid]:  # use retrieval results to filter
                                    continue
                                if agg[0] == 'sum':  # sum over cells within a table
                                    qid2rid2scores[qid][rid].append(np.sum([s for t, s in tss]))
                                elif agg[0] == 'max':  # avg over cells within a table
                                    qid2rid2scores[qid][rid].append(np.max([s for t, s in tss]))
                                else:
                                    raise NotImplementedError
                                for t, s in tss:
                                    qid2rid2text2score[qid][rid][t] += s
                        for qid in idx_li:
                            example = json.loads(get_next_target_until(qid))
                            if qid not in qid2rid2scores or len(qid2rid2scores[qid]) <= 0:
                                if skip_noret:
                                    continue
                                logging.warning(f'query {qid} has no retrieval results')
                                example['text2score'] = []
                                fout.write(json.dumps(example) + '\n')
                            else:
                                rid2scores = qid2rid2scores[qid]
                                if agg[1] == 'sum':
                                    rid = sorted(rid2scores.items(), key=lambda x: -np.sum(x[1]))[0][0]
                                elif agg[1] == 'count':
                                    rid = sorted(rid2scores.items(), key=lambda x: (-len(x[1]), -np.sum(x[1])))[0][0]
                                elif agg[1] == 'avg_count':
                                    rid = sorted(rid2scores.items(), key=lambda x: -np.mean(x[1]) * np.exp(len(x[1]) - 1))[0][0]
                                else:
                                    raise NotImplementedError
                                ret_example = idx2example[rid]
                                example['table'] = ret_example['table']
                                example['text2score'] = sorted(qid2rid2text2score[qid][rid].items(), key=lambda x: -x[1])
                                num_mention_li_before.append(len(example['context_before_mentions'][0]))
                                if use_str_match:
                                    locations, _ = BasicDataset.get_mention_locations(example['context_before'][0], ret_example['table']['data'])
                                    example['context_before_mentions'] = [locations]
                                num_mention_li.append(len(example['context_before_mentions'][0]))
                                fout.write(json.dumps(example) + '\n')
                    idx2byall = {}
                    idx_li = []
                    byall_li = []


def ner_example(batch_id, examples, nlp, skip_ner_types: Set[str] = None):
    skip_ner_types = set() if skip_ner_types is None else skip_ner_types
    contexts = [e['context_before'][0] for e in examples]
    docs = list(nlp.pipe(contexts, disable=['parser']))
    for doc, example in zip(docs, examples):
        ents = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ not in skip_ner_types]
        example['context_before_mentions'] = [sorted([(e[1], e[2]) for e in ents])]
    print(f'{batch_id} completed')
    return examples

def ner_file(prep_file: str, out_file: str, batch_size: int, nthread: int):
    nlp = spacy.load('en_core_web_sm')
    unique_tables: Set[str] = set()
    num_mentions_li: List[int] = []
    total_count = 0
    examples = []
    pool = multiprocessing.Pool(processes=nthread)
    results = []
    batch_id = 0
    with open(prep_file, 'r') as fin, open(out_file, 'w') as fout:
        for l in tqdm(fin):
            total_count += 1
            example = json.loads(l)
            examples.append(example)
            table = ''.join([c for r in example['table']['data'] for c in r])
            unique_tables.add(table)
            num_mentions_li.append(len(example['context_before_mentions'][0]))
            if len(examples) >= batch_size:
                r = pool.apply_async(functools.partial(ner_example, nlp=nlp, skip_ner_types=None), (batch_id, examples))
                results.append(r)
                examples = []
                if len(results) >= nthread:
                    for r in results:
                        for e in r.get():
                            fout.write(json.dumps(e) + '\n')
                    batch_id += 1
                    results = []
        if len(examples) > 0:
            r = pool.apply_async(functools.partial(ner_example, nlp=nlp, skip_ner_types=None), (batch_id, examples))
            results.append(r)
        if len(results) > 0:
            for r in results:
                for e in r.get():
                    fout.write(json.dumps(e) + '\n')
    print(f'total table {total_count}, #unique {len(unique_tables)}, avg #mentions {np.mean(num_mentions_li)}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=[
        'totto', 'wikisql', 'tablefact', 'wikitq', 'wikitq-pageid', 'turl', 'tapas', 'tapex',
        'overlap', 'fakepair', 'match_context_table', 'tableshuffle',
        'whole_faiss', 'span_faiss', 'span_as_whole_faiss', 'ret_filter_by_faiss', 'random_neg',
        'mrr', 'filter_mention', 'ner'])
    parser.add_argument('--path', type=Path, required=True, nargs='+')
    parser.add_argument('--output_dir', type=Path, required=False)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--model_type', type=str)
    args = parser.parse_args()

    random.seed(2021)
    np.random.seed(2021)

    # dataset-specific prep
    if args.data == 'totto':
        totto = Totto(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        totto.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_mention_cell.jsonl')
    elif args.data == 'wikisql':
        wsql = WikiSQL(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        wsql.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_mention_cell_with_sql.jsonl', add_sql=True)
        WikiSQL.add_answer(args.output_dir / args.split / 'preprocessed_mention_cell_with_sql.jsonl',
                           args.output_dir / 'converted' / f'{args.split}.tsv',  # generated by TAPAS
                           args.output_dir / args.split / 'preprocessed_mention_cell_with_sql_ans.jsonl')
    elif args.data == 'tablefact':
        tf = TableFact(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        tf.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_mention_cell.jsonl')
    elif args.data == 'wikitq':
        wtq = WikiTQ(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        wtq.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed.jsonl')
        split2file = {
            'train': 'random-split-5-train.tsv',
            'dev': 'random-split-5-dev.tsv',
            'test': 'test.tsv'
        }
        WikiTQ.add_answer(args.output_dir / args.split / 'preprocessed.jsonl',
                          args.output_dir / 'converted' / split2file[args.split],
                          args.output_dir / args.split / 'preprocessed_with_ans.jsonl',
                          string_match=True)
    elif args.data == 'wikitq-pageid':  # to get page id of the table
        split = args.split
        wikitq_root = args.path[0]
        output_file = args.output_dir
        wtq = WikiTQ(wikitq_root)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        wtq.convert_to_tabert_format(split, output_file)
    elif args.data == 'turl':
        avoid_titles = set()
        with open(str(args.path[0] / 'titles_in_3merge.txt'), 'r') as fin:
            for l in fin:
                avoid_titles.add(l.strip())
        turl = TurlData(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        # only use avoid_titles for the test split
        turl.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_cf_avoid3merge.jsonl',
                                      task='cell_filling', avoid_titles=avoid_titles)
        turl.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed_sa_avoid3merge.jsonl',
                                      task='schema_augmentation', avoid_titles=avoid_titles)
    elif args.data == 'tapas':
        tt = TapasTables(args.path[0])
        os.makedirs(args.output_dir / args.split, exist_ok=True)
        tt.convert_to_tabert_format(args.split, args.output_dir / args.split / 'preprocessed.jsonl')
    elif args.data == 'tapex':
        source_file = str(args.path[0])
        output_file = str(args.output_dir)
        tapex = Tapex(source_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        tapex.convert_to_tabert_format(output_file)

    # others
    elif args.data == 'fakepair':
        find_other_table(args.path[0], args.output_dir, max_count=3)
    elif args.data == 'match_context_table':
        only_self = False
        remove_self = True
        use_top1 = None
        is_tapas = True
        op = 'max'
        batch_size = 5000 if only_self else 1000
        timeout = batch_size * 0.5  # 0.5s per example
        nthread = 40
        topk = 10
        retrieval_file, target_file, source_file = args.path
        if is_tapas:
          max_context_len = 512
          max_num_rows = 100
        else:
          max_context_len = max_num_rows = None
        generate_retrieval_data(retrieval_file, target_file, source_file, args.output_dir,
                                bywhich=args.split, topk=topk, nthread=nthread, batch_size=batch_size,
                                max_context_len=max_context_len, max_num_rows=max_num_rows,  # used for tapas setting
                                remove_self=remove_self, only_self=only_self, timeout=timeout, use_top1=use_top1, op=op)
    elif args.data == 'random_neg':
        generate_random_neg(args.path[0], args.output_dir)
    elif args.data == 'tableshuffle':
        tableshuffle(args.path[0], args.output_dir)
    elif args.data == 'whole_faiss':
        index_emb_size = 256
        repr_file = args.path[0]
        output_file = args.output_dir
        whole_faiss(repr_file, output_file, index_emb_size=index_emb_size)
    elif args.data == 'span_faiss':
        repr_file = args.path[0]
        topk = 10
        index_emb_size = 256
        subsample = 1000000
        span_faiss(repr_file, args.output_dir, topk=topk, index_subsample=subsample, index_emb_size=index_emb_size)
    elif args.data == 'span_as_whole_faiss':
        index_emb_size = 256
        repr_file = args.path[0]
        output_file = args.output_dir
        whole_faiss(repr_file, output_file, index_emb_size=index_emb_size, use_span=True)
    elif args.data == 'ret_filter_by_faiss':
        repr_file, ret_file, prep_file = args.path
        topk = None
        index_emb_size = 256
        agg = 'max-sum'
        batch_size = 64
        use_str_match = True
        ret_filter_by_faiss(repr_file, ret_file, prep_file, prep_file, args.output_dir,
                            topk=topk, index_emb_size=index_emb_size, agg=agg, batch_size=batch_size,
                            use_str_match=use_str_match, separate=False, skip_noret=True)
    elif args.data == 'mrr':
        compute_ret_mrr(args.path[0])
    elif args.data == 'filter_mention':
        filter_mention(args.path[0], args.output_dir, topk=0, sort=True)
    elif args.data == 'ner':
        ner_file(args.path[0], args.output_dir, batch_size=10192, nthread=30)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
