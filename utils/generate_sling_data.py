from typing import List, Dict, Tuple, Set, Callable, Any
import argparse
import uuid
from pathlib import Path
import json
import random
import glob
import os
from tqdm import tqdm
from collections import defaultdict
import functools
from operator import itemgetter
import time
import sling
import re
import numpy as np
import spacy
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import torch
from torch_scatter import scatter_max, scatter_mean, scatter_add
from omnitab.utils import get_url, MultiprocessWrapper
from omnitab.dataset_utils import BasicDataset
from omnitab.dataset import Example
from omnitab.wikidata import topic2categories, get_cateid2name, WikipediaCategory
from omnitab.wikitablequestions import WikiTQ
from omnitab.retrieval import ESWrapper
from omnitab.squall import Squall
from omnitab.bm25 import BM25Wrapper
from omnitab.config import TableBertConfig
from omnitab.vanilla_table_bert import VanillaTableBert


def get_commons_docschema():
  commons = sling.Store()
  docschema = sling.DocumentSchema(commons)
  commons.freeze()
  return commons, docschema
self_commons, self_docschema = get_commons_docschema()


only_alphanumeric = re.compile('[\W_]+')


def load_url2tables(prep_file: Path) -> Tuple[Dict[str, List], List[Dict]]:
  table_count = image_url_count = 0
  url2tables: Dict[str, List] = defaultdict(list)
  raw_examples: List[Dict] = []
  with open(str(prep_file), 'r') as fin:
    for l in tqdm(fin):
      l = json.loads(l)
      table = l['table']
      url = get_url(l['uuid'])
      if url.endswith(('.jpg', '.jpeg', '.svg', '.gif', '.png', '.eps', '.bmp', '.tif', '.tiff')):
        image_url_count += 1
        continue
      url2tables[url].append(table)
      raw_examples.append(l)
      table_count += 1
  url_count = len(url2tables)
  print(f'#url {url_count}, #table {table_count}, #image url {image_url_count}')
  return url2tables, raw_examples


def load_pageid2tables(prep_file: Path) -> Dict[str, List]:
  table_count = 0
  pageid2tables: Dict[str, List] = defaultdict(list)
  with open(str(prep_file), 'r') as fin:
    for l in tqdm(fin):
      l = json.loads(l)
      table = l['table']
      pageid = l['pageid']
      pageid2tables[pageid].append(table)
      table_count += 1
  pageid_count = len(pageid2tables)
  print(f'#pageid {pageid_count}, #table {table_count}')
  return pageid2tables


def match_raw_worker(input_queue, output_queue):
  while True:
    args = input_queue.get()
    if type(args) is str and args == 'DONE':
      break
    examples = args['examples']
    for example in examples:
      context = example['context_before'][0]
      table = example['table']
      locations, location2cells = BasicDataset.get_mention_locations(context, table['data'])
      mention_cells: List[List[Tuple[int, int]]] = [location2cells[ml] for ml in locations]
      data_used = sorted(list(set(mc for mcs in mention_cells for mc in mcs)))

      table['data_used'] = data_used
      example = {
        'uuid': example['uuid'],
        'table': table,
        'context_before': [context],
        'context_before_mentions': [locations],
        'context_before_mentions_cells': [mention_cells],
        'context_after': []
      }
      output_queue.put(example)


def match_raw(examples: List[Dict],
              output_file: str,
              batch_size: int = 16,
              num_threads: int = 1,
              log_interval: int = 100):
  input_queue = Queue()
  output_queue = Queue()
  processes = []

  # start processes
  for _ in range(num_threads):
    p = Process(target=match_raw_worker,
                args=(input_queue, output_queue))
    p.daemon = True
    p.start()
    processes.append(p)
  write_p = Process(target=functools.partial(SlingExtractor.mention_write_worker, log_interval=log_interval),
                    args=(output_file, output_queue))
  write_p.start()

  for i in range(0, len(examples), batch_size):
    batch = examples[i:i + batch_size]
    input_queue.put({'examples': batch})

  for _ in processes:
    input_queue.put('DONE')
  for p in processes:
    p.join()
  output_queue.put('DONE')
  write_p.join()


class EnoughTokenException(Exception):
  pass


class SlingExtractor(object):
  def __init__(self, root_dir: Path, num_splits: int = 10):
    self.root_dir = root_dir
    self.num_splits = num_splits
    self.lang = 'en'

  @staticmethod
  def get_metadata(frame: sling.Frame) -> Dict:
    pageid = str(frame.get('/wp/page/pageid'))  # wikiPEDIA page ID
    title = frame.get('/wp/page/title')  # article title
    item = frame.get('/wp/page/item')  # wikidata ID associated to the article
    url = frame.get('url')
    return {
      'pageid': pageid,
      'title': title,
      'item': item,
      'url': url,
    }

  @staticmethod
  def get_categories(frame: sling.Frame) -> List[str]:
    return [v.id for k, v in frame if k.id == '/wp/page/category']

  @staticmethod
  def get_linked_entity(mention):
    if 'evokes' not in mention.frame:
      return None
    if type(mention.frame['evokes']) != sling.Frame:
      return None
    if 'is' in mention.frame['evokes']:
      if type(mention.frame['evokes']['is']) != sling.Frame:
        if ('isa' in mention.frame['evokes'] and
          mention.frame['evokes']['isa'].id == '/w/time' and
          type(mention.frame['evokes']['is']) == int):
          return mention.frame['evokes']['is']
        else:
          return None
      else:
        return mention.frame['evokes']['is'].id
    return mention.frame['evokes'].id

  @staticmethod
  def split_document_by_sentence(tokens) -> Tuple[Dict[int, int], Dict[int, Tuple[int, int]]]:
    # build token maps
    tok_to_sent_id, tok_to_para_id, sent_to_span, para_to_span = {}, {}, {}, {}
    i = sent_begin = para_begin = 0
    sent_id = para_id = 0
    for i, token in enumerate(tokens):
      if i > 0 and token.brk == 4:  # new para
        para_to_span[para_id] = (para_begin, i)
        sent_to_span[sent_id] = (sent_begin, i)
        para_id += 1
        sent_id += 1
        sent_begin = para_begin = i
      elif i > 0 and token.brk == 3:  # new sentence
        sent_to_span[sent_id] = (sent_begin, i)
        sent_id += 1
        sent_begin = i
      tok_to_sent_id[i] = sent_id
    para_to_span[para_id] = (para_begin, i + 1)
    sent_to_span[sent_id] = (sent_begin, i + 1)
    return tok_to_sent_id, sent_to_span

  @staticmethod
  def match_sentence_with_table_worker(input_queue: Queue,
                                       output_queue: Queue,
                                       use_all_more_than_num_mentions: int,
                                       topk: int,
                                       filter_function: Callable = None):
    while True:
      input_examples = input_queue.get()
      if type(input_examples) is str and input_examples == 'DONE':
        break

      examples: List[Dict] = []
      for inp_example in input_examples:
        key, document, tables = inp_example['key'], inp_example['document'], inp_example['tables']

        # get sentences
        sentences = SlingExtractor.get_sentences(document, skip_numeric_sent=True)

        for index, table in enumerate(tables):
          kept_sent_with_mentions: List[Tuple[str, List[Tuple], List[List[Tuple]]]] = []
          # find the topk best-matching sentence
          local_num_context_count = 0
          for sent in sentences:
            locations, location2cells = BasicDataset.get_mention_locations(sent, table['data'])
            mention_cells: List[List[Tuple[int, int]]] = [location2cells[ml] for ml in locations]
            data_used = sorted(list(set(mc for mcs in mention_cells for mc in mcs)))
            cover_ratio = np.sum([e - s for s, e in locations]) / (len(sent) or 1)
            if cover_ratio > 0.8:
              # rule: skip text that has over 80% of overlap with the table, which is likely to be table snippets
              continue
            if not use_all_more_than_num_mentions:
              kept_sent_with_mentions.append((sent, locations, mention_cells))
            elif use_all_more_than_num_mentions and len(locations) >= use_all_more_than_num_mentions:
              table['data_used'] = data_used
              examples.append({
                'uuid': f'sling_{key}_{index}_{local_num_context_count}',
                'table': table,
                'context_before': [sent],
                'context_before_mentions': [locations],
                'context_before_mentions_cells': [mention_cells],
                'context_after': []
              })
              local_num_context_count += 1

          if len(kept_sent_with_mentions) <= 0:
            continue

          if not use_all_more_than_num_mentions:  # output topk
            if filter_function is not None:  # use filter function
              kept_sent_with_mentions = filter_function(kept_sent_with_mentions)
            else:  # use topk
              kept_sent_with_mentions = sorted(kept_sent_with_mentions, key=lambda x: -len(x[1]))[:topk]

            if len(kept_sent_with_mentions) <= 0:
              continue

            # merge multiple sentences
            merge_sent: str = ''
            merge_locations: List[Tuple] = []
            merge_mention_cells: List[List[Tuple]] = []
            for sent, locations, mention_cells in kept_sent_with_mentions:
              offset = len(merge_sent) + int(len(merge_sent) > 0)
              merge_sent = (merge_sent + ' ' + sent) if len(merge_sent) > 0 else sent
              merge_locations.extend([(s + offset, e + offset) for s, e in locations])
              merge_mention_cells.extend(mention_cells)
            data_used = sorted(list(set(mc for mcs in merge_mention_cells for mc in mcs)))
            table['data_used'] = data_used
            examples.append({
              'uuid': f'sling_{key}_{index}',
              'table': table,
              'context_before': [merge_sent],
              'context_before_mentions': [merge_locations],
              'context_before_mentions_cells': [merge_mention_cells],
              'context_after': []
            })
      for example in examples:
        output_queue.put(example)

  @staticmethod
  def bm25_sentence_with_table_worker_sklearn(input_queue: Queue,
                                              output_queue: Queue,
                                              topk: int):
    while True:
      examples = input_queue.get()
      if type(examples) is str and examples == 'DONE':
        break
      for example in examples:
        key, document, tables = example['key'], example['document'], example['tables']
        sentences = SlingExtractor.get_sentences(document)

        # build index
        try:
          bm25 = BM25Wrapper(sentences)
        except ValueError:
          continue  # empty vocab

        # search
        for index, table in enumerate(tables):
          table_query: str = ESWrapper.format_table(table, full_table=True)
          ret_sents = bm25.query(table_query, topk=topk)
          context = ' '.join(ret_sents).strip()
          if len(context) <= 0:  # this shouldn't happens a lot
            continue

          # find mentions
          locations, location2cells = BasicDataset.get_mention_locations(context, table['data'])
          mention_cells: List[List[Tuple[int, int]]] = [location2cells[ml] for ml in locations]
          data_used = sorted(list(set(mc for mcs in mention_cells for mc in mcs)))
          if len(locations) <= 0:
            continue

          # output
          table['data_used'] = data_used
          out = {
            'uuid': f'bm25_{key}_{index}',
            'table': table,
            'context_before': [context],
            'context_before_mentions': [locations],
            'context_before_mentions_cells': [mention_cells],
            'context_after': []
          }
          output_queue.put(out)

  @staticmethod
  def bm25_sentence_with_table_worker(input_queue: Queue,
                                      output_queue: Queue,
                                      topk: int):
    es = ESWrapper()
    while True:
      examples = input_queue.get()
      if type(examples) is str and examples == 'DONE':
        break
      for example in examples:
        key, document, tables = example['key'], example['document'], example['tables']
        index_name = uuid.uuid4().hex  # url is not applicable as index name
        sentences = SlingExtractor.get_sentences(document)

        # build index
        def index_doc_iter(sents: List[str]):
          for sent in sents:
            yield {
              '_index': index_name,
              '_source': {'sent': sent},
            }
        es.set_index_name(index_name)
        es.build_index(index_doc_iter(sentences), refresh=True)

        # search
        for index, table in enumerate(tables):
          table_query: str = ESWrapper.format_table(table, full_table=True)
          ret_sents = [doc['sent'] for doc, score in es.get_topk(table_query, field='sent', topk=topk)]
          context = ' '.join(ret_sents).strip()
          if len(context) <= 0:  # this shouldn't happens a lot
            continue

          # find mentions
          locations, location2cells = BasicDataset.get_mention_locations(context, table['data'])
          mention_cells: List[List[Tuple[int, int]]] = [location2cells[ml] for ml in locations]
          data_used = sorted(list(set(mc for mcs in mention_cells for mc in mcs)))
          if len(locations) <= 0:
            continue

          # output
          table['data_used'] = data_used
          out = {
            'uuid': f'bm25_{key}_{index}',
            'table': table,
            'context_before': [context],
            'context_before_mentions': [locations],
            'context_before_mentions_cells': [mention_cells],
            'context_after': []
          }
          output_queue.put(out)

        # delete index
        es.delete_index()

  @staticmethod
  def random_sentence_with_table_worker(input_queue: Queue,
                                        output_queue: Queue,
                                        num_entities: int = 2):
    nlp = spacy.load('en_core_web_sm')
    while True:
      inp_examples = input_queue.get()
      if type(inp_examples) is str and inp_examples == 'DONE':
        break
      for inp_example in inp_examples:
        key, document, tables = inp_example['key'], inp_example['document'], inp_example['tables']
        # get sentences
        sentences = SlingExtractor.get_sentences(document)
        if len(sentences) <= 0:
          continue
        for index, table in enumerate(tables):
          sent = sentences[np.random.choice(len(sentences), 1)[0]]
          doc = nlp(sent)
          ents = [(ent.start_char, ent.end_char) for ent in doc.ents]
          if len(ents) <= 0:
            continue
          ents = sorted([ents[i] for i in np.random.choice(len(ents), min(num_entities, len(ents)), replace=False)])
          example = {
            'uuid': f'random_{index}',
            'table': table,
            'context_before': [sent],
            'context_before_mentions': [ents],
            'context_after': []
          }
          output_queue.put(example)

  @staticmethod
  def get_sentences(doc: bytes, skip_numeric_sent: bool = False) -> List[str]:
    store = sling.Store(self_commons)
    doc_frame = store.parse(doc)
    document = sling.Document(doc_frame, store, self_docschema)
    tokens = [t.word for t in document.tokens]
    token_brks = [t.brk for t in document.tokens]
    tokens = [t if tb == 0 else (' ' + t) for t, tb in zip(tokens, token_brks)]  # add space if needed

    # split by sentence
    tok_to_sent_id, sent_to_span = SlingExtractor.split_document_by_sentence(document.tokens)

    # filter sentence
    sents: List[str] = []
    for sent_id, (sent_start, sent_end) in sent_to_span.items():
      if sent_end - sent_start > 512:  # skip long sentence
        continue
      sent = ''.join(tokens[sent_start:sent_end])
      if not sent.endswith(('!', '"', "'", '.', ':', ';', '?')):
        # skip text without punctuations at the end (e.g., table snippets)
        continue
      if skip_numeric_sent:  # skip sentences where most characters are numeric characters
        sent_only_alphanumeric = only_alphanumeric.sub('', sent)
        sent_only_numberic = re.sub('[^0-9]', '', sent_only_alphanumeric)
        number_ratio = len(sent_only_numberic) / (len(sent_only_alphanumeric) or 1)
        if number_ratio > 0.8:
          continue
      sents.append(sent)
    return sents

  @staticmethod
  def match_sentence_with_sql_worker(input_queue: Queue,
                                     output_queue: Queue,
                                     topk: int,
                                     remove_keyword: bool = False):
    while True:
      examples = input_queue.get()
      if type(examples) is str and examples == 'DONE':
        break
      for example in examples:
        index_name = example['pageid']
        sentences = SlingExtractor.get_sentences(example['doc'])

        # build index
        es = ESWrapper(index_name)
        def index_doc_iter(sents: List[str]):
          for sent in sents:
            yield {
              '_index': index_name,
              'sent': sent
            }
        es.build_index(index_doc_iter(sentences))
        time.sleep(1)  # flush the index to avoid bug

        # search
        for id, sql, nl, raw_example in example['sqlnls']:
          sql_query = sql
          if remove_keyword:
            sql_wo_kw: List[str] = sql.strip().lower().split()  # split by whitespace
            sql_wo_kw = [w for w in sql_wo_kw if w not in Squall.KEYWORDS]
            sql_wo_kw: str = ' '.join(sql_wo_kw)
            sql_query = sql_wo_kw
          ret_sents = [doc['sent'] for doc, score in es.get_topk(sql_query, field='sent', topk=topk)]
          sqlbm25 = '{} {}'.format(' '.join(ret_sents), sql)
          raw_example['uuid'] = f'sqlbm25_{index_name}_{id}'
          raw_example['metadata'] = {'sql': sqlbm25, 'nl': nl}
          raw_example['context_before'] = [sqlbm25]
          output_queue.put(raw_example)

        # delete index
        es.delete_index()

  @staticmethod
  def emb_write_worker(output_file: str, output_queue: Queue):
    with tqdm() as pbar:
      while True:
        example = output_queue.get()
        if type(example) is str and example == 'DONE':
          break
        pbar.update(1)

  @staticmethod
  def write_worker(output_file: str, output_queue: Queue):
    with open(output_file, 'w') as fout, tqdm() as pbar:
      while True:
        example = output_queue.get()
        if type(example) is str and example == 'DONE':
          break
        pbar.update(1)
        fout.write(json.dumps(example) + '\n')

  @staticmethod
  def mention_write_worker(output_file: str, output_queue: Queue, log_interval):
    sent_len_li: List[int] = []
    num_mentions_li: List[int] = []
    empty_count = non_empty_count = i = 0
    with open(output_file, 'w') as fout, tqdm() as pbar:
      while True:
        example = output_queue.get()
        i += 1
        if type(example) is str and example == 'DONE':
          break
        pbar.update(1)
        nm = len(example['context_before_mentions'][0])
        empty_count += int(nm <= 0)
        non_empty_count += int(nm > 0)
        sent_len_li.append(len(example['context_before'][0]))
        num_mentions_li.append(nm)
        fout.write(json.dumps(example) + '\n')
        if i > 0 and i % log_interval == 0:
          print(f'avg sent len {np.mean(sent_len_li)}, '
                f'avg #mentions {np.mean(num_mentions_li)} '
                f'empty/non-empty {empty_count}/{non_empty_count}')

  def get_sentence_with_table(self,
                              key2tables: Dict[str, List],
                              key_feild: str,
                              from_split: int,  # inclusive
                              to_split: int,  # exclusive
                              output_file: str,
                              use_all_more_than_num_mentions: int = 0,
                              topk: int = 1,
                              filter_function: Callable = None,
                              get_method: str = 'match',
                              batch_size: int = 16,
                              num_threads: int = 1,
                              log_interval: int = 5000):

    if get_method == 'random':
      worker = self.random_sentence_with_table_worker
    elif get_method == 'match':
      worker = functools.partial(self.match_sentence_with_table_worker,
                                 use_all_more_than_num_mentions=use_all_more_than_num_mentions,
                                 topk=topk,
                                 filter_function=filter_function)
    elif get_method == 'bm25':
      worker = functools.partial(self.bm25_sentence_with_table_worker_sklearn, topk=topk)
    else:
      raise NotImplementedError

    mpw = MultiprocessWrapper(
      num_threads=num_threads,
      worker=worker,
      writer=functools.partial(self.mention_write_worker, log_interval=log_interval),
      output_file=output_file,
      batch_size=batch_size)

    # read data
    coverted_keys: Set[str] = set()
    for sp in range(from_split, to_split):
      corpus = sling.Corpus(str(self.root_dir / self.lang / f'documents-0000{sp}-of-00010.rec'))
      for n, (doc_id, doc_raw) in tqdm(enumerate(corpus.input), disable=True):
        store = sling.Store(self_commons)
        doc_frame = store.parse(doc_raw)
        metadata = self.get_metadata(doc_frame)
        key = metadata[key_feild]
        if key not in key2tables:
          continue
        if key in coverted_keys:
          print(f'{key} shows multiple times')
        coverted_keys.add(key)
        mpw.add_example({'key': key, 'document': doc_raw, 'tables': key2tables[key]})
    mpw.finish()

    print(f'#coverted keys {len(coverted_keys)}, #total keys {len(key2tables)}')
    print(f'uncovered keys {list(set(key2tables.keys()) - coverted_keys)[:10]}')

  def match_sentence_with_sql(self,
                              pageid2sqlnls: Dict[str, List[Tuple[str, str, str, Dict]]],
                              from_split: int,  # inclusive
                              to_split: int,  # exclusive
                              output_file: str,
                              topk: int = 1,
                              remove_keyword: bool = False,
                              batch_size: int = 16,
                              num_threads: int = 1):

    mpw = MultiprocessWrapper(
      num_threads=num_threads,
      worker=functools.partial(self.match_sentence_with_sql_worker, topk=topk, remove_keyword=remove_keyword),
      writer=self.write_worker,
      output_file=output_file,
      batch_size=batch_size)

    found = set()
    for sp in range(from_split, to_split):
      corpus = sling.Corpus(str(self.root_dir / self.lang / f'documents-0000{sp}-of-00010.rec'))
      for n, (doc_id, doc_raw) in tqdm(enumerate(corpus.input), disable=True):
        store = sling.Store(self_commons)
        doc_frame = store.parse(doc_raw)
        metadata = self.get_metadata(doc_frame)
        pageid = metadata['pageid']
        if pageid not in pageid2sqlnls:
          continue
        found.add(pageid)
        mpw.add_example({'pageid': pageid, 'doc': doc_raw, 'sqlnls': pageid2sqlnls[pageid]})

    # output examples where pageids are not found
    unfound = 0
    for pageid, sqlnls in pageid2sqlnls.items():
      if pageid in found:
        continue
      unfound += 1
      for id, sql, nl, raw_example in sqlnls:
        raw_example['uuid'] = f'sqlbm25_{pageid}_{id}'
        raw_example['metadata'] = {'sql': sql, 'nl': nl}
        raw_example['context_before'] = [sql]
        mpw.output_queue.put(raw_example)
    print(f'unfound #pageids {unfound}')
    mpw.finish()

  @staticmethod
  def prepare_sentences(sents: List[str], ner, tokenizer_fast, lower: bool, max_length: int):
    assert max_length > 2
    sents_prep: List[Dict] = []
    for sent in sents:
      ents: List[Tuple[int, int, str]] = [(ent.start_char, ent.end_char, ent.label_) for ent in ner(sent).ents]
      if lower:  sent = sent.lower()
      # remove overlap
      ents = sorted(ents, key=lambda x: (x[0], x[1]))
      ents = [ent for i, ent in enumerate(ents) if (i <= 0 or ent[0] >= ents[i - 1][1])]
      mentions: List[Tuple[Tuple[int, int], List]] = [((ent[0], ent[1]), []) for ent in ents]
      # tokenize
      sent_tokenized = tokenizer_fast(
        sent, max_length=max_length, truncation=True, add_special_tokens=True, return_offsets_mapping=True)
      offsets = sent_tokenized['offset_mapping']
      # convert char-based index to token-based index
      added_prefix_space = True if ('added_prefix_space' in sent_tokenized and sent_tokenized['added_prefix_space']) else False
      mentions_tok_based: List[Tuple[Tuple[int, int], List]] = Example.char_index2token_index(
        offsets, mentions, added_prefix_space=added_prefix_space)
      assert len(mentions_tok_based) == len(mentions), 'the char-to-token index conversion is not revertible'
      ents_tok_based: List[Tuple[int, int, str]] = [m[0] + (ent[-1],) for m, ent in zip(mentions_tok_based, ents)]
      ents_char_based: List[Tuple[int, int, str]] = [m[0] + (ent[-1],) for m, ent in zip(mentions, ents)]
      # output
      assert len(ents_tok_based) == len(ents_char_based)
      if len(ents_tok_based) <= 0:
        continue
      sent_prep = {
        'ids': sent_tokenized['input_ids'],
        'spans': ents_tok_based,
        'text': sent,
        'spans_char': ents_char_based,
      }
      sents_prep.append(sent_prep)
    return sents_prep

  @staticmethod
  def prepare_tables(tables: List[Dict], tokenizer_fast, lower: bool, max_length: int, linearization: str = 'tapex'):
    col_indicator = 'col :'
    sep = '|'
    assert max_length > 2
    assert linearization in {'tapex'}
    tokenize = lambda x: tokenizer_fast(x, add_special_tokens=False)['input_ids']
    sep_id = tokenize(sep)[0]
    tables_prep: List[Dict] = []
    for table in tables:
      header, data = [h['name'] for h in table['header']], table['data']
      tokens: List[int] = [tokenizer_fast.cls_token_id]  # add start special token
      text: List[str] = []
      text_off: int = 0
      cells_tok: List[Tuple[int, int, str]] = []
      cells_char: List[Tuple[int, int, str]] = []
      try:
        toadd = tokenize(col_indicator)
        if len(tokens) + len(toadd) > max_length - 2:  raise EnoughTokenException()
        tokens.extend(toadd)
        text.append(col_indicator)
        text_off += len(col_indicator)
        for i, h in enumerate(header):
          if lower:  h = h.lower()
          start = len(tokens)
          toadd = tokenize(h)
          if len(tokens) + len(toadd) > max_length - 2:  raise EnoughTokenException()
          tokens.extend(toadd)
          text.append(h)
          cells_tok.append((start, len(tokens), 'header'))
          cells_char.append((text_off + 1, text_off + 1 + len(h), 'header'))
          text_off += 1 + len(h)
          if i < len(header) - 1:
            tokens.append(sep_id)
            text.append(sep)
            text_off += 1 + len(sep)
        for row_idx, row in enumerate(data):
          toadd = tokenize(f'row {row_idx + 1} :')
          if len(tokens) + len(toadd) > max_length - 2:  raise EnoughTokenException()
          tokens.extend(toadd)
          text.append(f'row {row_idx + 1} :')
          text_off += 1 + len(f'row {row_idx + 1} :')
          for i, cell in enumerate(row):
            if lower:  cell = cell.lower()
            start = len(tokens)
            toadd = tokenize(cell)
            if len(tokens) + len(toadd) > max_length - 2:  raise EnoughTokenException()
            tokens.extend(toadd)
            text.append(cell)
            cells_tok.append((start, len(tokens), 'data'))
            cells_char.append((text_off + 1, text_off + 1 + len(cell), 'data'))
            text_off += 1 + len(cell)
            if i < len(row) - 1:
              tokens.append(sep_id)
              text.append(sep)
              text_off += 1 + len(sep)
      except EnoughTokenException:
        pass
      tokens.append(tokenizer_fast.sep_token_id)  # add end special token
      text: str = ' '.join(text)
      assert len(tokens) <= max_length
      assert len(cells_tok) == len(cells_char)
      if len(cells_tok) <= 0:
        continue
      table_prep = {
        'ids': tokens,
        'spans': cells_tok,
        'text': text,
        'spans_char': cells_char,
        'raw_table': table
      }
      tables_prep.append(table_prep)
    return tables_prep

  @staticmethod
  def gen_sentence_emb_worker(input_queue: Queue,
                              output_queue: Queue,
                              device_id: int,
                              config_path: str,
                              extra_config: Dict[str, Any],
                              gpu_batch_size: int,
                              topk_match: int,
                              topk_sent: int,
                              lower: bool,
                              score_function: str,
                              lowest_score: float = 0.0,
                              debug: bool = False):
    assert lowest_score >= 0, 'lower than zero is hard to implement'
    assert score_function in {'mean', 'mean-max'}
    # load the model and the tokenizer
    if device_id != 0:  # wait for the main process to download
      print(f'sleep {device_id}')
      time.sleep(60 * 5)
    config = TableBertConfig.from_file(config_path, **extra_config)
    device = torch.device(f'cuda:{device_id}')
    tokenizer_fast = config.tokenizer_fast_cls.from_pretrained(extra_config['base_model_name'])
    model = VanillaTableBert(config).to(device)
    ner = spacy.load('en_core_web_sm')
    pad_id = tokenizer_fast.pad_token_id
    print(f'complete download {device_id}')

    def pack(prep_list: List[Dict]):
      bs, seq_len = len(prep_list), max([len(p['ids']) for p in prep_list])
      packed_ids = np.full((bs, seq_len), dtype=np.int64, fill_value=pad_id)
      packed_attn_mask = np.zeros((bs, seq_len), dtype=np.int64)
      for idx, prep in enumerate(prep_list):
        packed_ids[idx, :len(prep['ids'])] = prep['ids']
        packed_attn_mask[idx, :len(prep['ids'])] = 1
      return torch.tensor(packed_ids).to(device), torch.tensor(packed_attn_mask).to(device)

    def run(ids: torch.Tensor,  # (bs, seq_len)
            attn_mask: torch.Tensor):  # (bs, seq_len)
      with torch.no_grad():
        reprs = []
        for batch in range(0, ids.size(0), gpu_batch_size):
          batch_ids = ids[batch:batch + gpu_batch_size]
          batch_attn_mask = attn_mask[batch:batch + gpu_batch_size]
          repr = model._bart.model(batch_ids, attention_mask=batch_attn_mask, return_dict=True).last_hidden_state
          reprs.append(repr)
        reprs = torch.cat(reprs, 0)
        reprs = reprs * attn_mask.unsqueeze(-1)  # mask out padding
        return reprs

    def get_tok2span(size: Tuple[int, int],   # (bs, seq_len)
                     prep_list: List[Dict],
                     pad_handle: str) -> Tuple[torch.Tensor, torch.Tensor]:  # (bs, seq_len), (bs, num_span)
      assert pad_handle in {'extend', 'zero'}  # different method to handle padding
      # convert
      tok2span = np.full(size, dtype=np.int64, fill_value=-1)  # (bs, seq_len) init with -1
      for idx, prep in enumerate(prep_list):
        for i, (start, end, _) in enumerate(prep['spans']):
          tok2span[idx, start:end] = i
      tok2span = torch.tensor(tok2span).to(device)
      # validate
      tok2span_mask = tok2span.ne(-1)  # (bs, seq_len)
      num_spans = torch.clamp(tok2span.max(-1)[0] + 1, min=0)  # (bs)
      max_num_spans = int(num_spans.max().item())
      span_mask = (torch.arange(max_num_spans).unsqueeze(0).to(device) < num_spans.unsqueeze(-1)).int()  # (bs, num_span)
      if pad_handle == 'extend':
        tok2span_noneg = tok2span * tok2span_mask + (~tok2span_mask) * num_spans.unsqueeze(-1)  # (bs, seq_len)
      elif pad_handle == 'zero':
        tok2span_noneg = tok2span * tok2span_mask  # (bs, seq_len)
      else:
        raise NotImplementedError
      return tok2span_noneg, span_mask

    def agg_repr(repr: torch.Tensor,  # (bs, seq_len, emb_size)
                 prep_list: List[Dict],
                 agg_func: str = 'mean'):
      bs, seq_len, emb_size = repr.size()
      tok2span, span_mask = get_tok2span((bs, seq_len), prep_list, pad_handle='extend')
      max_num_spans = span_mask.size(1)

      agg_func = eval(f'scatter_{agg_func}')
      tok2span = tok2span.unsqueeze(-1).expand(-1, -1, emb_size)  # (bs, seq_len, emb_size)
      span_repr = agg_func(
        repr, tok2span, dim=1,
        dim_size=max_num_spans + 1, fill_value=0).type(repr.dtype)  # (bs, max_num_spans + 1, emb_size)
      # remove the last "garbage" entry, mask out padding spans
      span_repr = span_repr[:, :-1] * span_mask.unsqueeze(-1)  # (bs, max_num_spans, emb_size)
      return span_repr, span_mask

    def agg_score(score: torch.Tensor,  # (n_tab, n_sent, n_tab_tok, n_sent_tok)
                  tab_prep_list: List[Dict],
                  sent_prep_list: List[Dict]):
      n_tab, n_sent, n_tab_tok, n_sent_tok = score.size()
      tab_tok2span, tab_span_mask = get_tok2span((n_tab, n_tab_tok), tab_prep_list, pad_handle='extend')  # (n_tab, n_tab_tok), (n_tab, n_tab_span)
      sent_tok2span, sent_span_mask = get_tok2span((n_sent, n_sent_tok), sent_prep_list, pad_handle='extend')  # (n_sent, n_sent_tok), (n_sent, n_sent_span)
      n_sent_span, n_tab_span = sent_span_mask.size(1), tab_span_mask.size(1)

      # max over sent tok
      sent_tok2span = sent_tok2span.unsqueeze(0).unsqueeze(-2).expand(n_tab, -1, n_tab_tok, -1)  # (n_tab, n_sent, n_tab_tok, n_sent_tok)
      # TODO: gpu support?
      score = scatter_max(score.detach().cpu(), sent_tok2span.cpu(), dim=-1, dim_size=n_sent_span + 1, fill_value=0)[0].to(score)  # (n_tab, n_sent, n_tab_tok, n_sent_span + 1)
      score = score[:, :, :, :-1] * sent_span_mask.unsqueeze(0).unsqueeze(-2)  # (n_tab, n_sent, n_tab_tok, n_sent_span)

      # mean over tab tok
      tab_tok2span = tab_tok2span.unsqueeze(1).unsqueeze(-1).expand(-1, n_sent, -1, n_sent_span)
      score = scatter_mean(score, tab_tok2span, dim=-2, dim_size=n_tab_span + 1, fill_value=0).type(score.dtype)  # (n_tab, n_sent, n_tab_span + 1, n_sent_span)
      score = score[:, :, :-1, :] * tab_span_mask.unsqueeze(1).unsqueeze(-1)  # (n_tab, n_sent, n_tab_span, n_sent_span)

      return score, tab_span_mask, sent_span_mask

    norm = lambda x: x / ((x * x).sum(-1, keepdim=True).sqrt() + 1e-10)

    while True:
      examples = input_queue.get()
      if type(examples) is str and examples == 'DONE':
        break
      for example in examples:
        try:
          key, document, raw_tables = example['key'], example['document'], example['tables']
          sents: List[str] = SlingExtractor.get_sentences(document, skip_numeric_sent=True)
          sents_prep: List[Dict] = SlingExtractor.prepare_sentences(sents, ner, tokenizer_fast, lower=lower, max_length=1024)
          tables_prep: List[Dict] = SlingExtractor.prepare_tables(raw_tables, tokenizer_fast, lower=lower, max_length=1024)
          if len(sents_prep) <= 0 or len(tables_prep) <= 0:
            continue

          sents_ids, sents_attn_mask = pack(sents_prep)  # (n_sent, n_sent_tok)
          table_ids, table_attn_mask = pack(tables_prep)  # (n_tab, n_tab_tok)

          sents_repr = run(sents_ids, sents_attn_mask)  # (n_sent, n_sent_tok, emb)
          tables_repr = run(table_ids, table_attn_mask)  # (n_tab, n_tab_tok, emb)

          if score_function == 'mean':  # avg span embedding and perform cosine
            # avg span emb
            sents_span_repr, sents_span_mask = agg_repr(sents_repr, sents_prep, agg_func='mean')  # (n_sent, n_sent_span, emb)
            tables_span_repr, tables_span_mask = agg_repr(tables_repr, tables_prep, agg_func='mean')  # (n_tab, n_tab_span, emb)
            # cosine
            sents_span_repr, tables_span_repr = norm(sents_span_repr), norm(tables_span_repr)
            scores = torch.einsum('ijk,xyk->ixjy', tables_span_repr, sents_span_repr)  # (n_tab, n_sent, n_tab_span, n_sent_span)
            mask = torch.einsum('ij,xy->ixjy', tables_span_mask, sents_span_mask)  # (n_tab, n_sent, n_tab_span, n_sent_span)
          elif score_function == 'mean-max':  # mean over table span tokens - max over sent span tokens
            # cosine
            sents_repr, tables_repr = norm(sents_repr), norm(tables_repr)
            scores_tok = torch.einsum('ijk,xyk->ixjy', tables_repr, sents_repr)  # (n_tab, n_sent, n_tab_tok, n_sent_tok)
            mask_tok = torch.einsum('ij,xy->ixjy', table_attn_mask, sents_attn_mask)  # (n_tab, n_sent, n_tab_tok, n_sent_tok)
            scores_tok = scores_tok * mask_tok
            scores, tables_span_mask, sents_span_mask = agg_score(scores_tok, tables_prep, sents_prep)
            mask = torch.einsum('ij,xy->ixjy', tables_span_mask, sents_span_mask)  # (n_tab, n_sent, n_tab_span, n_sent_span)
          else:
            raise NotImplementedError

          # remove low scores
          scores = scores * mask * scores.ge(lowest_score)

          # spasify
          n_table, n_sent, n_table_span, n_sent_span = mask.size()
          topk = min(topk_match, n_sent_span * n_table_span)
          sp_value, sp_ind = scores.view(n_table, n_sent, -1).topk(topk, -1)  # (n_tab, n_sent, topk)
          sp_ind_tablespan, sp_ind_sentspan = sp_ind // n_sent_span, sp_ind % n_sent_span  # (n_tab, n_sent, topk) not all valid
          sp_scores = torch.zeros_like(scores).view(n_table, n_sent, -1)  # (n_tab, n_sent, n_tab_span * n_sent_span)
          scores = sp_scores.scatter_(-1, sp_ind, sp_value).view(*scores.size())  # (n_tab, n_sent, n_tab_span, n_sent_span)

          # aggregate scores
          span_scores, sent_span_choices = scores.max(-1)  # (n_tab, n_sent, n_tab_span)
          sent_scores = span_scores.sum(-1)  # (n_tab, n_sent)
          final_scores, sent_choices = sent_scores.topk(min(topk_sent, n_sent), -1)  # (n_tab, topk_sent)

          # output
          for table_ind in range(n_table):
            raw_table = tables_prep[table_ind]['raw_table']
            for sent_ind in sent_choices[table_ind]:
              sent_ind = sent_ind.item()
              table_str = tables_prep[table_ind]['text']
              sent = sents_prep[sent_ind]['text']

              sentspan2tablespancore: Dict[Tuple[int, int], Tuple[int, int, float]] = defaultdict(lambda: (0, 0, lowest_score))
              for ts, ss in zip(sp_ind_tablespan[table_ind, sent_ind].cpu().numpy().tolist(),
                                sp_ind_sentspan[table_ind, sent_ind].cpu().numpy().tolist()):
                s = scores[table_ind, sent_ind, ts, ss].item()
                if s <= lowest_score:
                  continue
                start, end, label = sents_prep[sent_ind]['spans_char'][ss]
                tstart, tend, tlabel = tables_prep[table_ind]['spans_char'][ts]
                if s > sentspan2tablespancore[(start, end)][-1]:
                  sentspan2tablespancore[(start, end)] = (tstart, tend, s)

              if debug:
                print('*\t', sent.replace('\n', ' '))
                print('*\t', table_str.replace('\n', ' '))
                for (start, end), (tstart, tend, s) in sentspan2tablespancore.items():
                  print(sent[start:end], table_str[tstart:tend].replace('\n', ' '), s, sep='  |  ')
                print()

              mentions: List[Tuple[int, int]] = sorted(list(sentspan2tablespancore.keys()))
              if len(mentions) > 0:
                out = {
                  'uuid': f'dense_{key}_{table_ind}_{sent_ind}',
                  'metadata': {
                    'sent': sent,
                    'table': table_str,
                    'sentspan2tablespancore': list(sentspan2tablespancore.items()),
                  },
                  'table': raw_table,
                  'context_before': [sent],
                  'context_before_mentions': [mentions],
                  'context_after': []
                }
                output_queue.put(out)
        except KeyboardInterrupt as e:
          raise e
        except Exception as e:
          print('error')
          print(e)

  def gen_sentence_emb(self,
                       key2tables: Dict[str, List],
                       key_feild: str,
                       from_split: int,  # inclusive
                       to_split: int,  # exclusive
                       config_path: str,
                       extra_config: Dict[str, Any],
                       gpu_batch_size: int,
                       score_function: str,
                       topk_match: int,
                       topk_sent: int,
                       lower: bool,
                       output_file: str,
                       batch_size: int = 16,
                       num_threads: int = 1,
                       log_interval: int = 5000):
    torch.multiprocessing.set_start_method('spawn', force=True)
    mpw = MultiprocessWrapper(
      num_threads=num_threads,
      worker=functools.partial(self.gen_sentence_emb_worker,
                               config_path=config_path,
                               extra_config=extra_config,
                               gpu_batch_size=gpu_batch_size,
                               score_function=score_function,
                               topk_match=topk_match,
                               topk_sent=topk_sent,
                               lower=lower),
      writer=functools.partial(self.mention_write_worker, log_interval=log_interval),
      output_file=output_file,
      batch_size=batch_size,
      use_gpu=True)

    # read data
    coverted_keys: Set[str] = set()
    for sp in range(from_split, to_split):
      corpus = sling.Corpus(str(self.root_dir / self.lang / f'documents-0000{sp}-of-00010.rec'))
      for n, (doc_id, doc_raw) in tqdm(enumerate(corpus.input), disable=True):
        store = sling.Store(self_commons)
        doc_frame = store.parse(doc_raw)
        metadata = self.get_metadata(doc_frame)
        key = metadata[key_feild]
        if key not in key2tables or key in coverted_keys:
          continue
        coverted_keys.add(key)
        mpw.add_example({'key': key, 'document': doc_raw, 'tables': key2tables[key]})
    mpw.finish()

    print(f'#coverted keys {len(coverted_keys)}, #total keys {len(key2tables)}')
    print(f'uncovered keys {list(set(key2tables.keys()) - coverted_keys)[:10]}')

  def build_category_hierarchy(self, out_file: str):
    child2parents: Dict[str, List[str]] = {}
    for sp in range(0, 10):
      corpus = sling.Corpus(str(self.root_dir / self.lang / f'category-documents-0000{sp}-of-00010.rec'))
      for n, (wid, doc_raw) in tqdm(enumerate(corpus.input)):
        wid = wid.decode('utf-8')
        store = sling.Store(self_commons)
        doc_frame = store.parse(doc_raw)
        categories = self.get_categories(doc_frame)
        assert wid not in child2parents
        child2parents[wid] = categories
    with open(out_file, 'w') as fout:
      for c, ps in child2parents.items():
        fout.write('{}\t{}\n'.format(c, ','.join(ps)))

  @classmethod
  def get_top_category_by_request(cls,
                                  pageids: List[str],
                                  wc: WikipediaCategory,
                                  top_categories: Set[str],
                                  out_file: str = None,
                                  redirect: bool = False,
                                  follow_cate_hierachy_by_request: bool = False,
                                  use_revid: bool = False,
                                  cateid2name: Dict[str, str] = None,
                                  catename2id: Dict[str, str] = None,
                                  pageid2revid: Dict[str, str] = None):
    out_file = out_file or '/tmp/test'

    # revision id
    if use_revid:
      assert follow_cate_hierachy_by_request and not redirect, 'not implemented'
      with open(out_file, 'w') as fout:
        for pageid in pageids:
          try:
            revids = pageid2revid[pageid]
            pageid2cates = wc.pageid2cate([revids], use_revid=use_revid, max_limit=1)
            cates = list(pageid2cates.values())[0]
            promising_cate_title = WikipediaCategory.find_promising_cate(cates)
            _top_categories = set(cateid2name[c] for c in top_categories)
            top_category = wc.find_closest_top_category_by_request(promising_cate_title, _top_categories)
            top_category = catename2id[top_category]
            fout.write(f'{pageid}\t{top_category}\n')
          except:
            print(f'error {pageid}')
      return

    # redirect
    torawpageid: Dict[str, str] = {}
    if redirect:
      new_pageids: List[str] = []
      for pi in pageids:
        redirects = wc.pageid2redirect([pi])
        if len(redirects) > 0:
          new_pageids.append(redirects[0])
          torawpageid[redirects[0]] = pi
        else:
          new_pageids.append(pi)
      print(f'found redirects for {len(torawpageid)}/{len(pageids)}')
      pageids = new_pageids

    # get direct categories
    pageid2cates = wc.pageid2cate(pageids, max_limit=1)

    # get wikidata id of cetegoris
    if not redirect:
      all_cates = list(set(c for cates in pageid2cates.values() for c in cates))
      print(f'#categories {len(all_cates)}')
      title2wid = wc.titles2other(all_cates, field='wikidata')

    # find the top category
    with open(out_file, 'w') as fout:
      for pageid, cates in tqdm(pageid2cates.items()):
        promising_cate_title = WikipediaCategory.find_promising_cate(cates)
        if not redirect:
          cates = [title2wid[c] for c in cates if c in title2wid]
        try:
          if len(cates) <= 0:
            raise Exception('not categories as wikidata ids')
          if follow_cate_hierachy_by_request:
            _top_categories = set(cateid2name[c] for c in top_categories)
            top_category = wc.find_closest_top_category_by_request(promising_cate_title, _top_categories)
            top_category = [catename2id[top_category]]
          else:
            top_category: List[Tuple[str, int]] = wc.find_closest_top_category(cates, top_categories, track_num_path=True)
            max_num_path = max(map(itemgetter(1), top_category))
            top_category: List[str] = [c for c, n in top_category if n == max_num_path]
          top_category: str = ','.join(top_category)
          if pageid in torawpageid:
            pageid = torawpageid[pageid]
          fout.write(f'{pageid}\t{top_category}\n')
        except KeyboardInterrupt as e:
          raise e
        except:
          if pageid in torawpageid:
            print(f'error {torawpageid[pageid]}')
          else:
            print(f'error {pageid}')

  def get_top_category(self,
                       pageids: Set[str],
                       wc: WikipediaCategory,
                       top_categories: Set[str],
                       from_split: int = 0,
                       to_split: int = 10,
                       out_file: str = None):
    out_file = out_file or '/tmp/test'
    pageid2topcate: Dict[str, str] = {}
    with open(out_file, 'w') as fout:
      for sp in range(from_split, to_split):
        corpus = sling.Corpus(str(self.root_dir / self.lang / f'documents-0000{sp}-of-00010.rec'))
        for n, (doc_id, doc_raw) in tqdm(enumerate(corpus.input), disable=False):
          doc_id = doc_id.decode('utf-8')
          store = sling.Store(self_commons)
          doc_frame = store.parse(doc_raw)
          metadata = self.get_metadata(doc_frame)
          pageid = metadata['pageid']
          if pageid not in pageids:
            continue
          categories = self.get_categories(doc_frame)
          try:
            top_category: List[Tuple[str, int]] = \
              wc.find_closest_top_category(categories, top_categories, track_num_path=True)
            max_num_path = max(map(itemgetter(1), top_category))
            top_category: List[str] = [c for c, n in top_category if n == max_num_path]
            pageid2topcate[pageid] = top_category
            top_category: str = ','.join(top_category)
            fout.write(f'{pageid}\t{top_category}\n')
          except KeyboardInterrupt as e:
            raise e
          except Exception as e:
            print(f'error {pageid}')
    return pageid2topcate


def wikitq_overlap(test_wtq_path: Path,
                   train_wtq_path: Path,
                   topics: List[str],
                   topk: int = 100,
                   proportional: bool = False,
                   method: str = 'include'):
  assert method in {'include', 'overlap'}

  from spacy.lang.en import English
  nlp = English()
  tokenizer = nlp.tokenizer
  from transformers import BertTokenizer, BasicTokenizer
  #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  # count words
  test_topic2word2count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
  train_topic2word2count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
  for topic in topics:
    with open(str(test_wtq_path) + f'.{topic}', 'r') as test_fin, open(str(train_wtq_path) + f'.{topic}', 'r') as train_fin:
      for topic2word2count, fin in [(test_topic2word2count, test_fin), (train_topic2word2count, train_fin)]:
        for l in tqdm(fin):
          question = json.loads(l)['context_before'][0].lower()
          if isinstance(tokenizer, BasicTokenizer):
            tokens = tokenizer.tokenize(question)
          else:
            tokens = tokenizer(question)
          for t in tokens:
            if t.is_stop:
              continue
            topic2word2count[topic][t.text] += 1

  # choose top k words
  for topic in topics:
    if method == 'include':
      to_get_topk = [test_topic2word2count]
    elif method == 'overlap':
      to_get_topk = [test_topic2word2count, train_topic2word2count]
    else:
      raise NotImplementedError
    for tgt in to_get_topk:
      tops: List[Tuple[str, int]] = sorted(tgt[topic].items(), key=lambda x: -x[1])[:topk]
      sum_count = sum(map(itemgetter(1), tops)) or 1
      tgt[topic] = [(x[0], x[1] / sum_count) for x in tops]  # normalize
      print(f'{topic}: {tops[:10]}')

  # compute train/test stat
  for test_topic in topics:
    for train_topic in topics:
      if method == 'include':
        if proportional:
          o = np.sum([w[1] for w in test_topic2word2count[test_topic] if w[0] in train_topic2word2count[train_topic]])
        else:
          o = np.mean([w[0] in train_topic2word2count[train_topic] for w in test_topic2word2count[test_topic]])
      elif method == 'overlap':
        if proportional:
          raise NotImplementedError
        tests = set(map(itemgetter(0), test_topic2word2count[test_topic]))
        trains = set(map(itemgetter(0), train_topic2word2count[train_topic]))
        o = len(tests & trains) / (len(tests | trains) or 1)
      else:
        raise NotImplementedError
      print(o, end='\t')
    print('')


def select_by_embedding(prep_dir: Path,
                        out_file: Path,
                        num_files: int = 1,
                        score_thres: float = 1.0,
                        count_thres: int = 1,
                        debug: bool = False,
                        report: bool = False):
  # for bart-large: 0.7, 0.6, 0.5
  # for tapex-large: 0.8, 0.7, 0.6
  all_scores: List[float] = []
  key2scores: Dict[str, List[float]] = defaultdict(list)
  num_mentions: List[int] = []
  with open(out_file, 'w') as fout:
    for nf in range(num_files):
      key2examples: Dict[str, List[Tuple[Dict, float]]] = defaultdict(list)
      with open(prep_dir / f'split{nf}.jsonl', 'r') as fin:
        for l in tqdm(fin):
          l = json.loads(l)
          key = l['uuid'].rsplit('_', 1)[0]
          sent = l['metadata']['sent']
          table = l['metadata']['table']
          s2ts = l['metadata']['sentspan2tablespancore']
          mentions = []
          total_score = 0
          for (ss, se), (ts, te, s) in s2ts:
            key2scores[key].append(s)
            all_scores.append(s)
            if s >= score_thres:
              mentions.append((ss, se))
              total_score += s
              if debug:
                print(sent[ss:se], table[ts:te].replace('\n', ' '), s, sep=' | ')
          if len(mentions) >= count_thres:
            l['context_before_mentions'] = [mentions]
            key2examples[key].append((l, total_score))
          if len(mentions) and debug:
            input()
      if not report:
        for key, examples in key2examples.items():
          # sortec first by #mentions then by total score
          examples = sorted(examples, key=lambda x: (-len(x[0]['context_before_mentions'][0]), -x[1]))
          example = examples[0][0]
          num_mentions.append(len(example['context_before_mentions'][0]))
          fout.write(json.dumps(example) + '\n')
        print(f'count: {len(num_mentions)}, avg #mentions {np.mean(num_mentions)}')

  if not report:
    return
  # histogram
  plt.hist(all_scores, bins=100, weights=np.ones(len(all_scores)) / len(all_scores), cumulative=True)
  plt.savefig('test.png')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, choices=[
    'url_as_key', 'pageid_as_key', 'match_raw', 'build_category_hierarchy',
    'pageid2topic', 'assign_topic', 'wikitq_split_topic', 'wikitq_split_cate', 'wikitq_overlap',
    'sql2nl_with_retrieval', 'gen_sentence_emb', 'select_by_embedding'])
  parser.add_argument('--inp', type=Path, nargs='+')
  parser.add_argument('--out', type=Path)
  parser.add_argument('--topk', type=int, default=1, help='number of sentences used as context')
  parser.add_argument('--threads', type=int, default=1)
  parser.add_argument('--other', type=str, nargs='+')
  args = parser.parse_args()

  SEED = 2021
  random.seed(SEED)
  np.random.seed(SEED)

  if args.task == 'url_as_key':
    prep_file, sling_root = args.inp
    out_file = args.out
    filter_function = 'filter_function_min'
    get_method = 'bm25'
    log_interval = 1000
    batch_size = 128

    def filter_function_max(sent_with_mentions: List[Tuple[str, List[Tuple], List[List[Tuple]]]]):
      # return the sentence with max #mentions (return nothing if it's zero)
      top1 = sorted(sent_with_mentions, key=lambda x: -len(x[1]))[0]
      return [top1] if len(top1[1]) > 0 else []

    def filter_function_min(sent_with_mentions: List[Tuple[str, List[Tuple], List[List[Tuple]]]]):
      # return the sentence with min #mentions (execept those with zero mentions)
      sorted_swms = sorted(sent_with_mentions, key=lambda x: len(x[1]))
      for swm in sorted_swms:
        if len(swm[1]) > 0:
          return [swm]
      return []

    filter_function = eval(filter_function)
    url2tables = load_url2tables(prep_file)[0]
    se = SlingExtractor(sling_root)
    se.get_sentence_with_table(
      url2tables, key_feild='url', from_split=0, to_split=10, output_file=out_file,
      use_all_more_than_num_mentions=0, topk=args.topk, filter_function=filter_function, get_method=get_method,
      batch_size=batch_size, num_threads=args.threads, log_interval=log_interval)

  elif args.task == 'gen_sentence_emb':
    prep_file, sling_root = args.inp
    from_split, to_split, model_type = args.other
    out_file = args.out
    from_split, to_split = int(from_split), int(to_split)
    log_interval = 1000
    batch_size = 8
    gpu_batch_size = 4
    if model_type == 'bart_large':
      config_path = '/mnt/root/TaBERT/data/runs/bart_large/config.json'
      extra_config = {
        'objective_function': 'seq2seq',
        'base_model_name': 'facebook/bart-large',
        'load_model_from': None  # /mnt/root/TaBERT/data/runs/tapex_large/pytorch_model.bin
      }
    elif model_type == 'tapex_large':
      config_path = '/mnt/root/TaBERT/data/runs/tapex_large/config.json'
      extra_config = {
        'objective_function': 'seq2seq',
        'base_model_name': 'facebook/bart-large',
        'load_model_from': '/mnt/root/TaBERT/data/runs/tapex_large/pytorch_model.bin'
      }
    else:
      raise NotImplementedError
    url2tables = load_url2tables(prep_file)[0]
    se = SlingExtractor(sling_root)
    se.gen_sentence_emb(url2tables, 'url', from_split=from_split, to_split=to_split,
                        config_path=config_path,
                        extra_config=extra_config,
                        gpu_batch_size=gpu_batch_size,
                        score_function='mean-max',
                        topk_match=100,
                        topk_sent=5,
                        lower=True,
                        output_file=out_file,
                        batch_size=batch_size,
                        num_threads=args.threads,
                        log_interval=log_interval)

  elif args.task == 'select_by_embedding':
    prep_dir = args.inp[0]
    num_files = 10
    out_file = args.out
    select_by_embedding(prep_dir, out_file, num_files=num_files, score_thres=0.7, count_thres=1, report=False)

  elif args.task == 'pageid_as_key':
    prep_file, sling_root = args.inp
    out_file = args.out

    pageid2tables = load_pageid2tables(prep_file)
    se = SlingExtractor(sling_root)
    se.get_sentence_with_table(
      pageid2tables, key_feild='pageid', from_split=0, to_split=10, output_file=out_file,
      use_all_more_than_num_mentions=1, topk=args.topk, batch_size=16, num_threads=args.threads, log_interval=100)

  elif args.task == 'match_raw':
    prep_file = args.inp[0]
    out_file = args.out

    url2tables, raw_examples = load_url2tables(prep_file)
    match_raw(raw_examples, output_file=out_file, batch_size=16, num_threads=args.threads, log_interval=5000)

  elif args.task == 'build_category_hierarchy':
    sling_root = args.inp[0]
    out_file = args.out
    se = SlingExtractor(sling_root)
    se.build_category_hierarchy(out_file)

  elif args.task == 'pageid2topic':
    find_missing_from_sling = 3
    redirect = False
    hierachy_request = True
    use_revid = True
    wikipedia_cate_hierachy = 'data/wikipedia_category/sling_category_child2parents.txt'
    wtq_meta_file = Path('data/wikitablequestions/WikiTableQuestions/misc/table-metadata.tsv')

    cateid2name, catename2id = get_cateid2name(use_prefix=True)
    pageid2revid = WikiTQ.get_pageid2oldid(wtq_meta_file)

    if find_missing_from_sling:
      prep_files = args.inp[:-find_missing_from_sling]
      pageid2cate_files = args.inp[-find_missing_from_sling:]
    else:
      prep_files = args.inp[:-1]
      sling_root = args.inp[-1]
    out_file = args.out

    pids: Set[str] = set()
    for prep_file in prep_files:
      with open(prep_file, 'r') as fin:
        for l in fin:
          pid = json.loads(l)['pageid']
          pids.add(pid)

    if find_missing_from_sling:
      found_pids: Set[str] = set()
      for pageid2cate_file in pageid2cate_files:
        with open(pageid2cate_file, 'r') as fin:
          for l in fin:
            pageid, cates = l.rstrip('\n').split('\t')
            found_pids.add(pageid)

    wc = WikipediaCategory(child2parent_file=wikipedia_cate_hierachy, format='sling')
    top_categories: Set[str] = set(c for cs in topic2categories.values() for c in cs)
    print(f'#page ids {len(pids)}')

    if find_missing_from_sling:
      miss_pids = list(pids - found_pids)
      print(f'#missing page ids {len(miss_pids)}: {miss_pids}')
      SlingExtractor.get_top_category_by_request(
        miss_pids, wc, top_categories, out_file=out_file, redirect=redirect,
        follow_cate_hierachy_by_request=hierachy_request, use_revid=use_revid,
        cateid2name=cateid2name, catename2id=catename2id, pageid2revid=pageid2revid)
    else:
      se = SlingExtractor(sling_root)
      pageid2category: Dict[str, str] = se.get_top_category(
        pids, wc, top_categories, from_split=0, to_split=10, out_file=out_file)

  elif args.task == 'assign_topic':
    pageid2cate_files = args.inp
    out_file = args.out

    cate2topic: Dict[str, str] = {}
    for topic, cates in topic2categories.items():
      for cate in cates:
        cate2topic[cate] = topic

    pageid2cates: Dict[str, List[str]] = {}
    pageid2topic: Dict[str, str] = {}
    topic2count: Dict[str, int] = defaultdict(lambda: 0)
    numtopic2count: Dict[int, int] = defaultdict(lambda: 0)
    numcate2count: Dict[int, int] = defaultdict(lambda: 0)
    with open(out_file, 'w') as fout:
      for pageid2cate_file in pageid2cate_files:
        with open(pageid2cate_file, 'r') as fin:
          for l in fin:
            pageid, cates = l.rstrip('\n').split('\t')
            cates = cates.split(',')
            pageid2cates[pageid] = cates
            topics, counts = np.unique([cate2topic[cate] for cate in cates], return_counts=True)
            max_count = max(counts)
            topic: List[str] = [t for t, c in zip(topics, counts) if c == max_count]
            topic: str = np.random.choice(topic)
            pageid2topic[pageid] = topic
            topic2count[topic] += 1
            numtopic2count[len(topics)] += 1
            numcate2count[len(cates)] += 1
            fout.write(f'{pageid}\t{topic}\n')
    print('topic2count', sorted(topic2count.items(), key=lambda x: -x[1]))
    print('numtopic2count', sorted(numtopic2count.items(), key=lambda x: x[0]))
    print('numcate2count', sorted(numcate2count.items(), key=lambda x: x[0]))

    print('examples')
    pageid2cates: List[Tuple[str, List[str]]] = list(pageid2cates.items())
    random.shuffle(pageid2cates)
    cateid2name = get_cateid2name()[0]
    for pageid, cates in pageid2cates[:10]:
      cates = [cateid2name[c] for c in cates]
      print(f'https://en.wikipedia.org/?curid={pageid} {cates}')

  elif args.task == 'wikitq_split_topic':
    pageid2topic_file, wtq_path, prep_file = args.inp
    out_path = args.out
    wtq_id_key = 'uuid'

    pageid2topic: Dict[str, str] = dict(tuple(l.strip().split('\t')) for l in open(pageid2topic_file, 'r').readlines())
    wtq = WikiTQ(wtq_path)

    topic2file = {}
    topic2count = defaultdict(lambda: 0)
    with open(prep_file, 'r') as fin:
      for l in tqdm(fin):
        wtq_id = json.loads(l)[wtq_id_key]
        pageid = wtq.wtqid2pageid[wtq_id]
        if pageid in pageid2topic:
          topic = pageid2topic[pageid]
          if topic not in topic2file:
            topic2file[topic] = open(str(out_path) + f'.{topic.lower()}', 'w')
          topic2file[topic].write(l)
          topic2count[topic] += 1
    print('topic2count', sorted(topic2count.items(), key=lambda x: -x[1]))

  elif args.task == 'wikitq_split_cate':
    exclusive = True
    pageid2cate_file, wtq_path, prep_file = args.inp
    out_path = args.out

    pageid2cates: Dict[str, str] = dict(l.strip().split('\t') for l in open(pageid2cate_file, 'r').readlines())
    pageid2cates: Dict[str, List[str]] = {k: v.split(',') for k, v in pageid2cates.items()}
    wtq = WikiTQ(wtq_path)

    cate2file = {}
    cate2count = defaultdict(lambda: 0)
    with open(prep_file, 'r') as fin:
      for l in tqdm(fin):
        wtq_id = json.loads(l)['uuid']
        pageid = wtq.wtqid2pageid[wtq_id]
        if pageid in pageid2cates:
          cates = pageid2cates[pageid]
          if exclusive and len(cates) > 1:
            continue
          for cate in cates:
            if cate not in cate2file:
              cate2file[cate] = open(str(out_path) + f'.{cate}', 'w')
            cate2file[cate].write(l)
            cate2count[cate] += 1
    print('cate2count', sorted(cate2count.items(), key=lambda x: -x[1]))

  elif args.task == 'wikitq_overlap':
    split_dir = args.inp[0]
    split_by = 'cates'
    if split_by == 'topics':
      split_by = ['misc', 'people', 'politics', 'culture', 'sports']  # list(map(lambda x: x.lower(), topic2categories.keys()))
      print('\t'.join(split_by))
    elif split_by == 'cates':
      files = list(glob.glob(str(split_dir) + '/train.src.*'))  # get all train files
      sorted_files = sorted(files, key=lambda x: -os.path.getsize(x))  # sort by size
      split_by = [f.rsplit('.', 1)[1] for f in sorted_files]  # get category
      split_by = [c for c in split_by if
                  Path(str(split_dir / 'valid.src') + f'.{c}').exists() and
                  Path(str(split_dir / 'test.src') + f'.{c}').exists()]  # filter with dev/test
      cateid2name = get_cateid2name()[0]
      print('\t'.join(split_by))
      print('\t'.join([cateid2name[sb] for sb in split_by]))
    wikitq_overlap(split_dir / 'test.src', split_dir / 'train.src', split_by, topk=100, proportional=False, method='overlap')

  elif args.task == 'sql2nl_with_retrieval':
    topk = 1
    remove_keyword = True
    data = 'tapex'
    wtq_path, sling_root, prep_file = args.inp
    out_path = args.out

    wtq = WikiTQ(wtq_path)
    pageid2sqlnls: Dict[str, List[Tuple[str, str, str, Dict]]] = defaultdict(list)

    if data == 'wikitq':
      with open(prep_file, 'r') as fin:
        for l in fin:
          l = json.loads(l)
          wtqid = l['metadata']['ntid']
          sql = l['metadata']['sql']
          nl = l['metadata']['nl']
          pageid = wtq.wtqid2pageid[wtqid]
          pageid2sqlnls[pageid].append((wtqid, sql, nl, l))
    elif data == 'tapex':
      with open(prep_file, 'r') as fin:
        for l in fin:
          l = json.loads(l)
          wtqid = l['wtq_id']
          sql = l['context_before'][0]
          nl = 'dummy'
          pageid = wtq.wtqid2pageid[wtqid]
          pageid2sqlnls[pageid].append((wtqid, sql, nl, l))

    se = SlingExtractor(sling_root)
    se.match_sentence_with_sql(
      pageid2sqlnls, 0, 10, out_path, topk=topk, remove_keyword=remove_keyword, batch_size=1, num_threads=8)
