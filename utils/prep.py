from typing import List, Dict, Set, Tuple, Union
import argparse
import random
import json
import re
import os
from shutil import copyfile
from pathlib import Path
from tqdm import tqdm
from operator import itemgetter
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats
import functools
from multiprocessing import Queue
from transformers import BartForConditionalGeneration
from omnitab.dataset_utils import BasicDataset
from omnitab.utils import get_url, MultiprocessWrapper
from omnitab.squall import Squall
from omnitab.wikitablequestions import WikiTQ
from omnitab.config import TableBertConfig
from .eval import load_tagged_file


def self_in_dense(ret_file: str):
  mrrs: List[float] = []
  with open(ret_file, 'r') as fin:
    for l in fin:
      idx, bytext, bytable = l.rstrip('\n').split('\t')
      idx = int(idx)
      bytext = [int(s.split(',')[0]) for s in bytext.split(' ') if len(s) > 0]
      bytable = [int(s.split(',')[0]) for s in bytable.split(' ') if len(s) > 0]
      mrr = 0
      for i in range(max(len(bytext), len(bytable))):
        if (i < len(bytext) and bytext[i] == idx) or (i < len(bytable) and bytable[i] == idx):
          mrr = 1 / (i + 1)
          break
      mrrs.append(mrr)
  print(f'avg MRR {np.mean(mrrs)}')


def count_mentions(prep_file: str, max_num_examples: Union[int, None] = 50000, out_file: Path = None):
  num_mentions: List[int] = []
  context_lens: List[int] = []
  if out_file:  # filter sentence with mentions
    fout = open(out_file, 'w')
  with open(prep_file, 'r') as fin:
    for i, l in tqdm(enumerate(fin)):
      if max_num_examples and i >= max_num_examples:
        break
      d = json.loads(l)
      nm = len(d['context_before_mentions'][0])
      cl = len(d['context_before'][0])
      num_mentions.append(nm)
      context_lens.append(cl)
      if out_file and nm > 0:
        fout.write(l)
  print(f'avg #mention {np.mean(num_mentions)}, avg context len {np.mean(context_lens)}')
  plt.hist(num_mentions, bins=100, weights=np.ones(len(num_mentions)) / len(num_mentions))
  plt.savefig('test_num_mentions.png')
  plt.hist(context_lens, bins=100, weights=np.ones(len(context_lens)) / len(context_lens))
  plt.savefig('test_context_lens.png')


def tapex_ans_in_source(pred_file: str):
  ins = []
  with open(pred_file, 'r') as fin:
    for l in fin:
      l = json.loads(l)
      source = [h['name'] for h in l['table']['header']]
      source += [c for r in l['table']['data'] for c in r]
      source = ' '.join(source).lower()
      anss = l['answers']
      _in = True
      for ans in anss:
        if ans.lower() not in source:
          _in = False
          break
      ins.append(_in)
  print(f'answer in source: {np.mean(ins)}')


def get_shard_num(dir: Path, epoch: int) -> int:
  epoch_prefix = dir / f'epoch_{epoch}'
  shard_files = list(epoch_prefix.parent.glob(epoch_prefix.name + '.shard*.h5'))
  shard_ids = [int(re.search(r'shard(\d+)', str(f)).group(1)) for f in shard_files]
  shard_num = max(shard_ids) + 1
  return shard_num


def merge_shards(dirs: List[Path],
                 out_dir: Path,
                 epochs: int = 10,
                 keep_shards: List[int] = [-1, -1],
                 skip_first: bool = False,
                 use_softlink: bool = False,
                 alway_use_epochs: List[int] = [None, None]):
  assert len(dirs) == len(keep_shards) == len(alway_use_epochs)
  os.makedirs(str(out_dir), exist_ok=True)
  for e in range(epochs):
    sns: List[int] = [get_shard_num(dir, aue if aue is not None else e) for aue, dir in zip(alway_use_epochs, dirs)]
    keep_sns: List[int] = [sn if keep_shards[i] == -1 else min(sn, keep_shards[i]) for i, sn in enumerate(sns)]
    new_shard_id = 0
    for i, (dir, ksn, aue) in enumerate(zip(dirs, keep_sns, alway_use_epochs)):
      for s in range(ksn):
        from_file = dir / (f'epoch_{aue}.shard{s}.h5' if aue is not None else f'epoch_{e}.shard{s}.h5')
        to_file = out_dir / f'epoch_{e}.shard{new_shard_id}.h5'
        new_shard_id += 1
        if not skip_first or i != 0:
          print(f'{from_file} -> {to_file}')
          if use_softlink:
            from_file_rel = os.path.relpath(str(from_file), str(out_dir))
            os.symlink(str(from_file_rel), str(to_file))
          else:
            copyfile(str(from_file), str(to_file))


def get_table_style() -> str:
  return """
    <head>
      <style>
      table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
      }
      </style>
    </head>
  """


def get_table_html(table: Dict):
  headers = table['header']
  highlight_stype = 'style="color:red;"'
  headers = '<tr>' + ' '.join(
    [f'<th {highlight_stype if h["used"] else ""}>{h["name"]}</th>' for h in headers]) + '</tr>'
  data = table['data']
  data_used = set(map(tuple, table['data_used']))
  data = '<tr>' + '</tr><tr>'.join([' '.join(
    [f'<th {highlight_stype if (row_idx, col_idx) in data_used else ""}>{c}</th>' for col_idx, c in enumerate(row)])
    for row_idx, row in enumerate(data)]) + '</tr>'
  table_str = f"""
    <table>
      <thead>
        {headers}
      </thead>
      <tbody>
        {data}
      </tbody>
    </table>
  """
  return table_str


def get_context_html(context: str, highlight_spans: List[Tuple[int, int]]):
  prev = 0
  context_: List[str] = []
  for s, e in highlight_spans:
    context_.append(context[prev:s])
    context_.append(f'<span style="color:red;">{context[s:e]}</span>')
    prev = e
  context_.append(context[prev:])
  return ''.join(context_)


def ret_compare(ret_files: List[str],
                skip_first: List[bool],
                is_ret: List[bool],
                ret_query_file: str,
                ret_doc_file: str,
                output_file: str,
                sample_ratio: float = 1.0,
                topk: int = 1):
  is_same = ret_query_file == ret_doc_file

  print('read retrieval file ...')
  qid2docids: Dict[int, List[List[int]]] = defaultdict(list)
  all_qids: Set[int] = set()
  all_docids: Set[int] = set()
  ret_fins = [open(rf, 'r') for ir, rf in zip(is_ret, ret_files) if ir]
  try:
    while True:
      try:
        lines: List[str] = [rf.readline() for rf in ret_fins]
        if lines[0] == '':
          break
        if random.random() > sample_ratio:
          continue
        for sf, line in zip(skip_first, lines):
          qid, docs = line.rstrip('\n').split('\t')[:2]
          qid = int(qid)
          docids = [int(d.split(',')[0]) for d in docs.split(' ')]
          if is_same and sf:  # skip the first retrieved doc which is always self
            docids = docids[1:]
          docids = docids[:topk]
          qid2docids[qid].append(docids)
          all_qids.add(qid)
          all_docids.update(docids)
      except StopIteration:
        break
  finally:
    for rf in ret_fins:
      if rf:  rf.close()

  print('read prep file ...')
  context2table: List[List[Tuple[str, Dict]]] = []
  prep_fins = [open(rf, 'r') for ir, rf in zip(is_ret, ret_files) if not ir]
  if len(prep_fins) > 0:
    try:
      while True:
        try:
          lines: List[str] = [pf.readline() for pf in prep_fins]
          if lines[0] == '':
            break
          if random.random() > sample_ratio:
            continue
          context2table.append([])
          for line in lines:
            line = json.loads(line)
            context2table[-1].append((line['context_before'][0], line['table']))
        except StopIteration:
          break
    finally:
      for pf in prep_fins:
        if pf:  pf.close()

  print('read query/doc files ...')
  id2query: Dict[str, Dict] = {}
  with open(ret_query_file, 'r') as fin:
    for i, l in tqdm(enumerate(fin)):
      if i not in all_qids and i not in all_docids:
        continue
      id2query[i] = json.loads(l)
  if not is_same:
    id2doc: Dict[str, Dict] = {}
    with open(ret_doc_file, 'r') as fin:
      for i, l in enumerate(fin):
        if i not in all_docids:
          continue
        id2doc[i] = json.loads(l)
  else:
    id2doc = id2query

  print('output ...')
  with open(output_file, 'w') as fout:
    fout.write(get_table_style())
    fout.write('<body>\n')
    qid2docids: List[Tuple] = list(qid2docids.items())

    group2nm: Dict[int, List[int]] = defaultdict(list)
    # analyze retrieval files
    for i in tqdm(np.random.permutation(len(qid2docids))):
      qid, docids = qid2docids[i]
      context_url = get_url(id2query[qid]['uuid'])
      context = id2query[qid]['context_before'][0]
      fout.write(f'<div><a href="{context_url}">{context_url}</a><br><h3>{context}</h3></div>\n')
      for group, _docids in enumerate(docids):
        # find the one with highest overlap
        max_nm = 0
        max_docid = None
        for docid in _docids:
          num_mentions = len(BasicDataset.get_mention_locations(context, id2doc[docid]['table']['data'])[0])
          if num_mentions >= max_nm:
            max_nm = num_mentions
            max_docid = docid
        table_url = get_url(id2doc[max_docid]['uuid'])
        table = get_table_html(id2doc[max_docid]['table'])
        group2nm[group].append(max_nm)
        fout.write(f'<div><a href="{table_url}">{table_url}</a><br>{table}</div><br>\n')
      fout.write('<hr>\n')
    fout.write('</body>')

    # analyze prep files
    for _context2table in context2table:
      for group, (context, table) in enumerate(_context2table):
        nm = len(BasicDataset.get_mention_locations(context, table['data'])[0])
        group2nm[group + len(ret_fins)].append(nm)

    print(f'total count {len(qid2docids)}')
    group2nm = {k: (np.mean(v), len(v)) for k, v in group2nm.items()}
    print(f'group2nm: {group2nm}')


def visualize_prep_file(prep_file: str, output_file: str, sample_ratio: float = 1.0):
  with open(prep_file, 'r') as fin, open(output_file, 'w') as fout:
    fout.write(get_table_style())
    for i, l in tqdm(enumerate(fin)):
      if random.random() > sample_ratio:
        continue
      l = json.loads(l)
      url = get_url(l['uuid']) if 'metadata' not in l else l['metadata']['page_url']
      context = get_context_html(l['context_before'][0], l['context_before_mentions'][0])
      table = l['table']
      table_html = get_table_html(table)
      fout.write(f'<div><a href="{url}">{url}</a><br><h3>{context}</h3></div>\n')
      fout.write(f'{table_html}\n')
      fout.write('<hr>\n')


def replace_context(generation_file: Path,
                    prep_file: Path,
                    full_prep_file: Path,
                    output_file: Path,
                    num_files: int,
                    remove_dup: bool = False,
                    remove_empty: bool = False,
                    has_logprob: bool = False):
  idx2gens_logprobs: Dict[int, List[Tuple[str, float]]] = {}
  num_gens_before_dedup: List[int] = []
  num_gens_after_dedup: List[int] = []
  empty_count = 0
  for i in range(num_files):
    with open(f'{generation_file}.{i}', 'r') as fin:
      for l in tqdm(fin):
        l = l.rstrip('\n').split('\t')
        idx = int(l[-1])
        gens = l[:-3]
        if has_logprob:
          assert len(gens) % 2 == 0
          gens, logprobs = gens[:len(gens) // 2], gens[len(gens) // 2:]
          logprobs = list(map(float, logprobs))
        else:
          logprobs = [.0] * len(gens)
        idx2gens_logprobs[idx] = []
        used: Set[str] = set()
        for gen, logprob in zip(gens, logprobs):
          for rmt in ['<pad>', '<s>', '</s>']:
            gen = gen.replace(rmt, '')
          gen = gen.strip()
          if remove_empty and len(gen) <= 0:
            empty_count += 1
            continue
          if not remove_dup or gen not in used:
            used.add(gen)
            idx2gens_logprobs[idx].append((gen, logprob))
        num_gens_before_dedup.append(len(gens))
        num_gens_after_dedup.append(len(used))
  print(f'#sentences before/after dedup {np.mean(num_gens_before_dedup)}/{np.mean(num_gens_after_dedup)} '
        f'with {empty_count} empty generations')

  # load some other stuff from the full prep file
  wtqid2answer: Dict[str, str] = {}
  if full_prep_file:
    with open(full_prep_file, 'r') as fin:
      for idx, l in enumerate(tqdm(fin)):
        l = json.loads(l)
        wtqid2answer[l['uuid']] = l['answers']

  prev_len: List[int] = []
  new_len: List[int] = []
  ids: Set[str] = set()
  with open(prep_file, 'r') as fin, \
    open(output_file, 'w') as fout, \
    open(f'{output_file}.compare', 'w') as cfout:
    for idx, l in enumerate(tqdm(fin)):
      l = json.loads(l)
      if idx not in idx2gens_logprobs:
        continue
      if 'metadata' in l and 'wtqid' in l['metadata'] and l['metadata']['wtqid'] in wtqid2answer:  # add answer
        l['answers'] = wtqid2answer[l['metadata']['wtqid']]
      prev = l['context_before'][0]
      prev_len.append(len(prev))
      ids.add(l['uuid'])
      for gen, logprob in idx2gens_logprobs[idx]:
        l['context_before'] = [gen]
        if has_logprob:
          if 'metadata' not in l:  l['metadata'] = {}
          l['metadata']['logprob'] = logprob
        new_len.append(len(gen))
        fout.write(json.dumps(l) + '\n')
        cfout.write(f'{prev}\t{gen}\n')

  print(f'#len {np.mean(prev_len)} -> {np.mean(new_len)}')


def select_by_loss(loss_file: Path,
                   prep_file: Path,
                   output_file: Path,
                   num_files: int,
                   topks: List[int],
                   skips: Set[str] = None,
                   used_for_sql2nl: bool = False,
                   selection_method: str = 'max',
                   criteria: str = 'denotation'):
  assert selection_method in {'max', 'min'}
  assert criteria in {'denotation', 'generation'}

  idx2lp: Dict[int, float] = {}
  for i in range(num_files):
    with open(f'{loss_file}.{i}', 'r') as fin:
      for l in tqdm(fin):
        l = l.rstrip('\n').split('\t')
        idx, loss = int(l[1]), float(l[0])
        idx2lp[idx] = -loss

  # aggregate based on wtq id
  skip_count = 0
  wtqid2examples: Dict[str, List[Tuple[Dict, float, float]]] = defaultdict(list)
  with open(prep_file, 'r') as fin:
    for idx, l in enumerate(tqdm(fin)):
      l = json.loads(l)
      wtqid = l['metadata']['wtqid']
      logprob = l['metadata']['logprob'] if 'logprob' in l['metadata'] else 0.0
      if skips and wtqid in skips:  # skip examples with real NL
        skip_count += 1
        continue
      wtqid2examples[wtqid].append((l, idx2lp[idx], logprob))
  print(f'#skip {skip_count}')

  # compute the correlation between two log probs
  corrs = []
  for _, examples in wtqid2examples.items():
    corrs.append(scipy.stats.pearsonr(list(map(itemgetter(1), examples)), list(map(itemgetter(2), examples))))
  print(f'correlation between two log probs {np.mean(corrs)}')

  if criteria == 'denotation':
    criteria_ind = 1
  elif criteria == 'generation':
    criteria_ind = 2
  else:
    raise NotImplementedError

  # only use the sampled nl with max lp
  wtqid2example: Dict[str, Tuple[Dict, float, float]] = {}
  for wtqid, examples in wtqid2examples.items():
    if selection_method == 'max':
      ind = np.argmax(list(map(itemgetter(criteria_ind), examples)))
    elif selection_method == 'min':
      ind = np.argmin(list(map(itemgetter(criteria_ind), examples)))
    else:
      raise NotImplementedError
    example = examples[ind]
    if used_for_sql2nl:
      example[0]['metadata']['nl'] = example[0]['context_before'][0]
    wtqid2example[wtqid] = example

  # visualize
  lps = list(map(itemgetter(criteria_ind), wtqid2example.values()))
  print(f'avg lp {np.mean(lps)}')
  plt.hist(lps, bins=100, weights=np.ones(len(lps)) / len(lps))
  plt.savefig('test.png')

  for topk in topks:
    # get topk
    if selection_method == 'max':
      sort_func = lambda x: -x[1][criteria_ind]
    elif selection_method == 'min':
      sort_func = lambda x: x[1][criteria_ind]
    else:
      raise NotImplementedError
    _wtqid2example: List[Tuple[str, Tuple[Dict, float, float]]] = sorted(wtqid2example.items(), key=sort_func)[:topk]
    lens = []
    with open(f'{output_file}.top{topk}', 'w') as fout:
      for wtqid, (example, lp, gen_lp) in _wtqid2example:
        fout.write(json.dumps(example) + '\n')
        lens.append(len(example['context_before'][0]))
    print(f'topk: {topk}: avg lp {np.mean(list(map(lambda x: x[1][criteria_ind], _wtqid2example)))}, avg len {np.mean(lens)}')


def process_bidirection(bi_file: str, prep_file: str, output_file: str, num_files: int):
  idx2lp: Dict[int, float] = {}
  for i in range(num_files):
    with open(f'{bi_file}.{i}', 'r') as fin:
      for l in tqdm(fin):
        loss, idx = l.strip().split('\t')
        idx, lp = int(idx), -float(loss)
        if idx >= 15539230:  # TODO: some unknow preprocessing errors
          continue
        idx2lp[idx] = lp

  print(f'#idx in bidirection file {len(idx2lp)} with the max idx {np.max(list(idx2lp.keys()))}')

  prev_uuid = None
  example = None
  context_with_score: List[Tuple[str, Tuple[float, float]]] = []
  num_contexts_to_choose_from: List[int] = []

  def choose(context_with_score, example, fout):
    best_context = sorted(context_with_score, key=lambda x: -sum(x[1]))[0][0]
    example['context_before'] = [best_context]
    fout.write(json.dumps(example) + '\n')

  with open(prep_file, 'r') as fin, open(output_file, 'w') as fout:
    for _idx, l in enumerate(tqdm(fin)):
      l = json.loads(l)
      uuid = l['uuid']
      context = l['context_before'][0]
      taidx = _idx * 2
      teidx = _idx * 2 + 1
      if taidx not in idx2lp or teidx not in idx2lp:
        continue
      table2text_lp = idx2lp[taidx]
      text2table_lp = idx2lp[teidx]
      if prev_uuid is not None and uuid != prev_uuid:  # choose the best context
        num_contexts_to_choose_from.append(len(context_with_score))
        choose(context_with_score, example, fout)
        context_with_score = []
      context_with_score.append((context, (table2text_lp, text2table_lp)))
      example = l
      prev_uuid = uuid
    if len(context_with_score) > 0:
      num_contexts_to_choose_from.append(len(context_with_score))
      choose(context_with_score, example, fout)
      context_with_score = []

  print(f'avg #context {np.mean(num_contexts_to_choose_from)}')


def compare_two_files(prep_file1: str, prep_file2: str):
  '''
  prep_file2 might include only a subset from prep_file1
  '''
  lens1: List[int] = []
  lens2: List[int] = []
  with open(prep_file1, 'r') as fin1, open(prep_file2, 'r') as fin2:
    for l2 in tqdm(fin2):
      l1 = fin1.readline()
      if l1 == '': break
      e1 = json.loads(l1)
      e2 = json.loads(l2)
      while e1['uuid'] != e2['uuid']:
        l1 = fin1.readline()
        if l1 == '': break
        e1 = json.loads(l1)
      assert e1['uuid'] == e2['uuid']
      c1 = e1['context_before'][0]
      c2 = e2['context_before'][0]
      #print(c1, '\n', c2)
      lens1.append(len(c1))
      lens2.append(len(c2))
  print(f'compare count {len(lens1)}, avg len {np.mean(lens1)}, {np.mean(lens2)}')


def tapex_which_table(wtq_prep_files: List[str], tapex_prep_file: str, out_file: str, threads: int = 1):
  def get_table_cells(table: Dict):
    cells: List[str] = []
    columns = [h['name'] for h in table['header']]
    rows = table['data']
    for r in [columns] + rows:
      for c in r:
        c = c.lower()
        c = '' if c == 'none' else c
        c = re.sub(r'[^a-zA-Z0-9_ ]+', '', c).strip()
        if len(c):
          cells.append(c)
    return cells

  def get_table_signature(table: Dict):
    columns = [h['name'] for h in table['header']]
    rows = table['data']
    # it seems that the TAPEX pretrain data use none for empty cell
    # use ten characters at most
    sig = ' '.join([' '.join(map(lambda x: '' if x == 'none' else x[:20], r)) for r in ([columns] + rows)]).lower()
    sig_normalized = re.sub(r'[^a-zA-Z0-9_ ]+', '', sig)
    return sig, sig_normalized

  def worker(input_queue: Queue, output_queue: Queue, sig2id: Dict[str, str], cell2id: Dict[str, Set[str]]):
    while True:
      examples = input_queue.get()
      if type(examples) is str and examples == 'DONE':
        break
      for example in examples:
        jd: Dict = json.loads(example['json_str'])
        sig = get_table_signature(jd['table'])[-1]
        found = sig in sig2id

        id2count: Dict[str, int] = defaultdict(lambda: 0)
        cells = get_table_cells(jd['table'])
        used_cells = 0
        for cell in cells:
          if len(cell2id[cell]) >= 100:  # skip common cells
            continue
          used_cells += 1
          for id in cell2id[cell]:
            id2count[id] += 1
        id2count: List[Tuple[str, int]] = sorted(id2count.items(), key=lambda x: -x[1])
        found_by_cell = len(id2count) > 0 and (id2count[0][1] / (used_cells or 1)) >= 0.5

        found_by_cell_id = None
        if found_by_cell:
          found_by_cell_id = id2count[0][0]
        jd['wtq_id'] = found_by_cell_id

        output_queue.put(jd)

  def writer(output_file: str, output_queue: Queue, monitored_ids: Set[str]):
    founds: List[bool] = []
    found_in_monitored = 0
    count = 0

    with open(output_file, 'w') as fout, tqdm(disable=False) as pbar:
      while True:
        example = output_queue.get()
        if type(example) is str and example == 'DONE':
          break
        pbar.update(1)
        count += 1
        found = example['wtq_id'] is not None
        founds.append(found)
        found_in_monitored += int(example['wtq_id'] in monitored_ids)
        if found:
          fout.write(json.dumps(example) + '\n')
        if count % 10000 == 0:
          print(f'found {np.mean(founds)}, '
          f'found in test set {found_in_monitored}')

    print(f'found {np.mean(founds)}, '
          f'found in test set {found_in_monitored}')

  _sig2id: Dict[str, str] = {}
  sig2id: Dict[str, str] = {}
  cell2id: Dict[str, Set[str]] = defaultdict(set)
  monitored_ids: Set[str] = set()
  for wtq_file_id, wtq_prep_file in enumerate(wtq_prep_files):
    with open(wtq_prep_file, 'r') as fin:
      for l in fin:
        l = json.loads(l)
        id = l['uuid']
        table, table_normalized = get_table_signature(l['table'])
        if table not in _sig2id and table_normalized in sig2id:
          raise Exception(f'collision after normalizing for {id}')
        _sig2id[table] = id
        sig2id[table_normalized] = id
        for cell in get_table_cells(l['table']):
          cell2id[cell].add(id)
        if wtq_file_id == len(wtq_prep_files) - 1:
          monitored_ids.add(id)
  print(f'#tables in WTQ {len(sig2id)}')

  mpw = MultiprocessWrapper(
    num_threads=threads,
    worker=functools.partial(worker, sig2id=sig2id, cell2id=cell2id),
    writer=functools.partial(writer, monitored_ids=monitored_ids),
    output_file=out_file,
    batch_size=128)

  with open(tapex_prep_file, 'r') as fin:
    for l in fin:
      mpw.add_example({'json_str': l})
  mpw.finish()


def random_pair_context_with_table(prep_file: str, out_file: str):
  context_related_keys = ['context_before', 'context_before_mentions',
                          'context_after', 'context_after_mentions']
  tables: List[Dict] = []
  contexts: List[Dict] = []
  with open(prep_file, 'r') as fin:
    for i, l in enumerate(fin):
      l = json.loads(l)
      tables.append(l)
      contexts.append({k: l[k] for k in context_related_keys if k in l})
  perm = np.random.permutation(len(contexts))
  with open(out_file, 'w') as fout:
    for i in range(len(tables)):
      example = tables[i]
      for k in context_related_keys:
        if k not in example:
          continue
        example[k] = contexts[perm[i]][k]
      fout.write(json.dumps(example) + '\n')


def count_cells(prep_file: str):
  nums: List[int] = []
  with open(prep_file, 'r') as fin:
    for l in tqdm(fin):
      table = json.loads(l)['table']
      nh = len(table['header'])
      nc = np.sum([len(r) for r in table['data']])
      nums.append(nh + nc)
  print(f'mean {np.mean(nums)}, max {np.max(nums)}, median {np.median(nums)}')


def fewshot_pcyin_nsm(fewshot_prep_file: Path, nsm_dir: Path, out_dir: Path, from_shard: int = 0, to_shard: int = 90):
  wtqids: Set[str] = set()
  with open(fewshot_prep_file, 'r') as fin:
    for l in fin:
      wtqid = json.loads(l)['uuid']
      wtqids.add(wtqid)
  print(f'#ids in our file {len(wtqids)}')

  os.makedirs(out_dir, exist_ok=True)
  final_count = 0
  for s in range(from_shard, to_shard):
    in_file = nsm_dir / f'train_split_shard_{to_shard}-{s}.jsonl'
    to_file = out_dir / f'train_split_shard_{to_shard}-{s}.jsonl'
    with open(in_file, 'r') as fin, open(to_file, 'w') as fout:
      for l in fin:
        if json.loads(l)['id'] in wtqids:
          assert l.endswith('\n')
          fout.write(l)
          final_count += 1
  print(f'#ids in output file {final_count}')


def lenof(cases: Dict[str, Tuple[Dict, Dict]], squall: Squall, wtq: WikiTQ, field: str = 'sql'):
  len_list = []
  for key, (better, worse) in cases.items():
    l = len(squall.wtqid2example[better['id']][field])
    len_list.append(l)
  return np.mean(len_list)


def distributionof(cases: Dict[str, Tuple[Dict, Dict]], squall: Squall, wtq: WikiTQ, field: str = 'sql', dedup: bool = False):
  tok2count = defaultdict(lambda: 0)
  for key, (better, worse) in cases.items():
    toks = squall.wtqid2example[better['id']][field]
    if dedup:  toks = set(toks)
    for tok in toks:
      tok2count[tok] += 1
  for tok in tok2count:
    tok2count[tok] /= len(cases)
  return tok2count


def sumof(cases: Dict[str, Tuple[Dict, Dict]], squall: Squall, wtq: WikiTQ, field: str = 'sql'):
  len_list = []
  for key, (better, worse) in cases.items():
    l = sum(squall.wtqid2example[better['id']]['raw'][field])
    len_list.append(l)
  return np.mean(len_list)


def tablecell(cases: Dict[str, Tuple[Dict, Dict]], squall: Squall, wtq: WikiTQ):
  header_lens = []
  data_lens = []
  for key, (better, worse) in cases.items():
    header, _, data = wtq.get_table(wtq.wtqid2tableid[better['id']])
    header_lens.append(len(header))
    data_lens.append(len(data) * len(data[0]))
  return np.mean(header_lens), np.mean(data_lens), np.mean(header_lens) + np.mean(data_lens)


def compare_nat_syn(nat_file: str,  syn_file: str, multi_file: str, squall: Squall, wtq: WikiTQ):
  def compare_distribution(tok2count1: Dict[str, int], tok2count2: Dict[str, int]):
    keys1 = [x[0] for x in sorted(tok2count1.items(), key=lambda x: -x[1])]
    keys2 = [x[0] for x in sorted(tok2count2.items(), key=lambda x: -x[1])]
    all_keys = keys1 + [k for k in keys2 if k not in keys1]
    print('\t'.join(all_keys))
    print('\t'.join(['{:.3f}'.format(tok2count1[k]) for k in all_keys]))
    print('\t'.join(['{:.3f}'.format(tok2count2[k]) for k in all_keys]))

  key2datas = []
  for file in [nat_file,  syn_file, multi_file]:
    key2data = {}
    with open(file, 'r') as fin:
      for l in fin:
        l = json.loads(l)
        key2data[l['id']] = l
    key2datas.append(key2data)
  nat_data, syn_data, multi_data = key2datas
  print(set(nat_data.keys()) == set(syn_data.keys()) == set(multi_data.keys()))

  keys = [k for k in squall.wtqid2example if k in nat_data]
  nat_cases: Dict[str, Tuple[Dict, Dict]] = {}
  syn_cases: Dict[str, Tuple[Dict, Dict]] = {}
  print(f'#examples {len(keys)}')
  for key in keys:
    em = [nat_data[key]['em'], syn_data[key]['em'], multi_data[key]['em']]
    if nat_data[key]['em'] and not syn_data[key]['em']:  # nat good
      nat_cases[key] = (nat_data[key], syn_data[key])

    if not nat_data[key]['em'] and syn_data[key]['em']:  # syn good
      syn_cases[key] = (syn_data[key], nat_data[key])

  print(f'nat better {len(nat_cases)}, syn better {len(syn_cases)}')
  print('sql-char', lenof(nat_cases, squall, wtq, field='sql'), lenof(syn_cases, squall, wtq, field='sql'))
  print('nl-char', lenof(nat_cases, squall, wtq, field='nl'), lenof(syn_cases, squall, wtq, field='nl'))
  print('sql-kw', lenof(nat_cases, squall, wtq, field='sql_kw'), lenof(syn_cases, squall, wtq, field='sql_kw'))
  print('sql-token', lenof(nat_cases, squall, wtq, field='sql_raw'), lenof(syn_cases, squall, wtq, field='sql_raw'))
  print('nl-token', lenof(nat_cases, squall, wtq, field='nl_raw'), lenof(syn_cases, squall, wtq, field='nl_raw'))

  print('nl_incolumns', sumof(nat_cases, squall, wtq, field='nl_incolumns'), sumof(syn_cases, squall, wtq, field='nl_incolumns'))
  print('nl_incells', sumof(nat_cases, squall, wtq, field='nl_incells'), sumof(syn_cases, squall, wtq, field='nl_incolumns'))
  print('columns_innl', sumof(nat_cases, squall, wtq, field='columns_innl'), sumof(syn_cases, squall, wtq, field='columns_innl'))

  print('tablecell', tablecell(nat_cases, squall, wtq), tablecell(syn_cases, squall, wtq))

  print('sql-kw')
  compare_distribution(
    distributionof(nat_cases, squall, wtq, field='sql_kw', dedup=True),
    distributionof(syn_cases, squall, wtq, field='sql_kw', dedup=True))



def fewshot_tapas(fewshot_prep_file: Path, tsv_file: Path, out_file: Path):
  wtqids: Set[str] = set()
  with open(fewshot_prep_file, 'r') as fin:
    for l in fin:
      wtqid = json.loads(l)['uuid']
      wtqids.add(wtqid)
  print(f'#ids in our file {len(wtqids)}')

  final_count = 0
  with open(tsv_file, 'r') as fin, open(out_file, 'w') as fout:
    for i, l in enumerate(fin):
      if i == 0 or l.strip().split()[0] in wtqids:
        assert l.endswith('\n')
        fout.write(l)
        final_count += 1
  print(f'#ids in output file {final_count}')


def convert_to_official_eval(pred_file: Path, prep_file: Path, out_file: Path):
  ind2tagged = load_tagged_file('/root/exp/WikiTableQuestions/tagged/data/pristine-unseen-tables.tagged')
  ind2tagged: Dict[int, List[List[str]]] = dict(zip(range(len(ind2tagged)), ind2tagged))

  cls_token, sep_token, pad_token = TableBertConfig.get_special_tokens('facebook/bart-base')
  with open(pred_file, 'r') as pfin, open(prep_file, 'r') as sfin, open(out_file, 'w') as fout:
    for i, p in enumerate(pfin):
      pred = p.rstrip('\n').split('\t')[0]
      for rmt in [pad_token, cls_token, sep_token]:
        pred = pred.replace(rmt, '')
      if len(ind2tagged[i]) == 1:
        pass
      else:
        pred = '\t'.join(pred.split(', '))
      id = json.loads(sfin.readline())['uuid']
      fout.write(f'{id}\t{pred}\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, required=True, choices=[
    'self_in_dense', 'count_mentions', 'tapex_ans_in_source', 'merge_shards',
    'ret_compare', 'vis_prep', 'replace_context', 'process_bidirection', 'compare_two_files',
    'dump_correct_bart', 'tapex_which_table', 'random_pair_context_with_table',
    'select_by_loss', 'count_cells', 'fewshot_pcyin_nsm', 'fewshot_tapas',
    'compare_nat_syn', 'convert_to_official_eval'])
  parser.add_argument('--inp', type=Path, required=False, nargs='+')
  parser.add_argument('--other', type=str, nargs='+')
  parser.add_argument('--out', type=Path, required=False, default=None)
  args = parser.parse_args()

  SEED = 2021
  random.seed(SEED)
  np.random.seed(SEED)

  if args.task == 'self_in_dense':
    self_in_dense(args.inp[0])

  elif args.task == 'count_mentions':
    out_file = args.out
    count_mentions(args.inp[0], max_num_examples=None, out_file=out_file)

  elif args.task == 'tapex_ans_in_source':
    tapex_ans_in_source(args.inp[0])

  elif args.task == 'merge_shards':
    merge_shards(args.inp[:2], args.out, epochs=1, keep_shards=[-1, -1],
                 skip_first=True, use_softlink=False, alway_use_epochs=[None, None])

  elif args.task == 'ret_compare':
    ret_files = args.inp[:-2]
    skip_first = [False, False, False, True, False]
    is_ret_file = [True, True, True, True, False]
    assert len(ret_files) == len(skip_first) == len(is_ret_file)
    ret_query_file, ret_doc_file = args.inp[-2:]
    output_file = args.out
    ret_compare(ret_files, skip_first, is_ret_file, ret_query_file, ret_doc_file, output_file, sample_ratio=0.001, topk=5)

  elif args.task == 'vis_prep':
    prep_file = args.inp[0]
    output_file = args.out
    visualize_prep_file(prep_file, output_file, sample_ratio=0.01)

  elif args.task == 'replace_context':
    generation_file, prep_file = args.inp[:2]
    full_prep_file = args.inp[2] if len(args.inp) > 2 else None
    output_file = args.out
    num_gpu, remove_dup, remove_empty = int(args.other[0]), eval(args.other[1]), eval(args.other[2])
    has_logprob = False
    replace_context(generation_file, prep_file, full_prep_file=full_prep_file, output_file=output_file,
                    num_files=num_gpu, remove_dup=remove_dup, remove_empty=remove_empty, has_logprob=has_logprob)

  elif args.task == 'process_bidirection':
    bi_file, prep_file = args.inp
    output_file = args.out
    num_gpu = 64
    process_bidirection(bi_file, prep_file, output_file, num_files=num_gpu)

  elif args.task == 'compare_two_files':
    prep_file1, prep_file2 = args.inp
    compare_two_files(prep_file1, prep_file2)

  elif args.task == 'dump_correct_bart':
    # the mask embedding in old verions of BART-base is incorrect
    # use the latest version of transformers when running this function
    model_name = 'facebook/bart-large'
    dump_dir = '/mnt/root/TaBERT/data/runs/bart_large'
    model = BartForConditionalGeneration.from_pretrained(model_name).eval()
    save_function = lambda obj, f: torch.save(obj, f, _use_new_zipfile_serialization=False)
    model.save_pretrained(dump_dir, save_function=save_function)

  elif args.task == 'tapex_which_table':
    wtq_prep_files = args.inp[:-1]
    tapex_prep_file = args.inp[-1]
    out_file = args.out
    tapex_which_table(wtq_prep_files, tapex_prep_file, out_file, threads=8)

  elif args.task == 'random_pair_context_with_table':
    prep_file = args.inp[0]
    out_file = args.out
    random_pair_context_with_table(prep_file, out_file)

  elif args.task == 'select_by_loss':
    loss_file, prep_file, to_skip_file = args.inp
    output_file = args.out
    num_gpu = int(args.other[0])
    selection_method = 'max'
    criteria = 'denotation'
    topks = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    skips = set(json.loads(l)['uuid'] for l in open(to_skip_file, 'r').readlines())
    select_by_loss(loss_file, prep_file, output_file,
                   num_files=num_gpu, topks=topks, skips=skips, used_for_sql2nl=True,
                   selection_method=selection_method, criteria=criteria)

  elif args.task == 'count_cells':
    prep_file = args.inp[0]
    count_cells(prep_file)

  elif args.task == 'fewshot_pcyin_nsm':
    fewshot_prep_file, nsm_dir = args.inp
    out_dir = args.out
    fewshot_pcyin_nsm(fewshot_prep_file, nsm_dir, out_dir)

  elif args.task == 'fewshot_tapas':
    fewshot_prep_file, wtq_dir = args.inp
    out_file = args.out
    tsv_file = wtq_dir / 'data' / 'random-split-1-train.tsv'
    fewshot_tapas(fewshot_prep_file, tsv_file, out_file)

  elif args.task == 'compare_nat_syn':
    #nat_file,  syn_file, multi_file = args.inp
    nat_file, syn_file, multi_file = 'analysis/natural.jsonl', 'analysis/synthetic.jsonl', 'analysis/multitask.jsonl'
    squall = Squall(Path('data/squall/data/squall.json'), wikitq=None)
    wtq = WikiTQ(Path('data/wikitablequestions/WikiTableQuestions'))
    compare_nat_syn(nat_file, syn_file, multi_file, squall, wtq)

  elif args.task == 'convert_to_official_eval':
    pred_file, prep_file = args.inp
    out_file = args.out
    convert_to_official_eval(pred_file, prep_file, out_file)
