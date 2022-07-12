from typing import List, Dict, Tuple, Set
import argparse
from pathlib import Path
import json
import random
from tqdm import tqdm
from collections import defaultdict
import functools
import time
from spacy.lang.en import English
import re
import numpy as np
from multiprocessing import Process, Queue
from datasets import load_dataset
from omnitab.dataset_utils import BasicDataset
from utils.generate_sling_data import SlingExtractor


only_alphanumeric = re.compile('[\W_]+')


def load_url2tables(prep_file: Path) -> Dict[str, List]:
  table_count = 0
  url2tables: Dict[str, List] = defaultdict(list)
  with open(str(prep_file), 'r') as fin:
    for l in tqdm(fin):
      l = json.loads(l)
      table = l['table']
      url = l['metadata']['page_url']
      url2tables[url].append(table)
      table_count += 1
  url_count = len(url2tables)
  print(f'#url {url_count}, #table {table_count}')
  return url2tables


def annotate_sentence(urls: List[str],
                      documents: List[str],
                      nlp,
                      tables_li: List[List[Dict]],
                      use_all_more_than_num_mentions: int,
                      topk: int) -> List[Dict]:
  examples: List[Dict] = []

  for url, document, tables in zip(urls, documents, tables_li):
    # split by sentence
    document = nlp(document)
    sents: List[str] = [sent.text for sent in document.sents]

    for index, table in enumerate(tables):
      kept_sent_with_mentions: List[Tuple[str, List[Tuple], List[List[Tuple]]]] = []
      # find the topk best-matching sentence
      local_num_context_count = 0
      for sent_id, sent in enumerate(sents):
        if len(sent.split()) >= 512:  # skip long sentence
          continue
        if not sent.endswith(('!', '"', "'", '.', ':', ';', '?')):
          # rule 1: skip text without punctuations at the end (e.g., table snippets)
          continue
        sent_only_alphanumeric = only_alphanumeric.sub('', sent)
        sent_only_numberic = re.sub('[^0-9]', '', sent_only_alphanumeric)
        number_ratio = len(sent_only_numberic) / (len(sent_only_alphanumeric) or 1)
        if number_ratio > 0.8:
          # rule 2: skip text where most characters are numeric characters
          continue
        locations, location2cells = BasicDataset.get_mention_locations(sent, table['data'])
        mention_cells: List[List[Tuple[int, int]]] = [location2cells[ml] for ml in locations]
        data_used = sorted(list(set(mc for mcs in mention_cells for mc in mcs)))
        cover_ratio = np.sum([e - s for s, e in locations]) / (len(sent) or 1)
        if cover_ratio > 0.8:
          # rule 3: skip text that has over 80% of overlap with the table, which is likely to be table snippets
          continue
        if not use_all_more_than_num_mentions:
          kept_sent_with_mentions.append((sent, locations, mention_cells))
        elif use_all_more_than_num_mentions and len(locations) >= use_all_more_than_num_mentions:
          table['data_used'] = data_used
          examples.append({
            'uuid': f'sling_{url}_{index}_{local_num_context_count}',
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
        # sort by number of mentions
        kept_sent_with_mentions = sorted(kept_sent_with_mentions, key=lambda x: -len(x[1]))[:topk]
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
          'uuid': f'sling_{url}_{index}',
          'table': table,
          'context_before': [merge_sent],
          'context_before_mentions': [merge_locations],
          'context_before_mentions_cells': [merge_mention_cells],
          'context_after': []
        })
  return examples


def match_sentence_with_table_worker(input_queue: Queue,
                                     output_queue: Queue,
                                     use_all_more_than_num_mentions: int,
                                     topk: int):
  nlp = English()  # just the language with no table_bert
  nlp.add_pipe('sentencizer')
  func = functools.partial(annotate_sentence, nlp=nlp,
                           use_all_more_than_num_mentions=use_all_more_than_num_mentions, topk=topk)
  while True:
    args = input_queue.get()
    if type(args) is str and args == 'DONE':
      break
    for example in func(**args):
      output_queue.put(example)


def match_sentence_with_table(self,
                              url2tables: Dict[str, List],
                              output_file: str,
                              use_all_more_than_num_mentions: int = 0,
                              topk: int = 1,
                              batch_size: int = 16,
                              num_threads: int = 1,
                              log_interval: int = 5000):

  input_queue = Queue()
  output_queue = Queue()
  processes = []

  # start processes
  for _ in range(num_threads):
    p = Process(target=functools.partial(match_sentence_with_table_worker,
                                         use_all_more_than_num_mentions=use_all_more_than_num_mentions,
                                         topk=topk),
                args=(input_queue, output_queue))
    p.daemon = True
    p.start()
    processes.append(p)
  write_p = Process(target=functools.partial(SlingExtractor.write_worker, log_interval=log_interval),
                    args=(output_file, output_queue))
  write_p.start()

  # read data
  coverted_urls: Set[str] = set()
  urls: List[str] = []
  doc_li: List[str] = []
  tables_li: List[List[Dict]] = []

  dataset = load_dataset('c4', 'en')  # TODO: use cache dir

  for c4example in dataset:
    url = c4example['url']
    doc = c4example['text']

    if url not in url2tables:
      continue
    if url in coverted_urls:
      print(f'{url} shows multiple times')
    coverted_urls.add(url)

    urls.append(url)
    doc_li.append(doc)
    tables_li.append(url2tables[url])

    if len(urls) >= batch_size:
      input_queue.put({'urls': urls, 'documents': doc_li, 'tables_li': tables_li})
      urls = []
      doc_li = []
      tables_li = []

  for _ in processes:
    input_queue.put('DONE')
  for p in processes:
    p.join()
  output_queue.put('DONE')
  write_p.join()

  print(f'#coverted urls {len(coverted_urls)}, #total urls {len(url2tables)}')
  print(f'uncovered urls {list(set(url2tables.keys()) - coverted_urls)[:10]}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--inp', type=Path, nargs='+')
  parser.add_argument('--out', type=Path)
  parser.add_argument('--topk', type=int, default=1, help='number of sentences used as context')
  parser.add_argument('--threads', type=int, default=40)
  args = parser.parse_args()

  SEED = 2021
  random.seed(SEED)
  np.random.seed(SEED)

  prep_file = args.inp[0]
  out_file = args.out

  url2tables = load_url2tables(prep_file)

  match_sentence_with_table(
    url2tables, output_file=out_file,
    use_all_more_than_num_mentions=3, topk=args.topk, batch_size=16, num_threads=args.threads, log_interval=5000)
