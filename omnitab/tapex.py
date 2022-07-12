from typing import List, Tuple
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from omnitab.dataset_utils import BasicDataset


class Tapex(BasicDataset):
  def __init__(self, source_file: str):
    self.source_file = source_file
    self.target_file = source_file.replace('src', 'tgt')
    self.examples = self.load(self.source_file, self.target_file)

  @staticmethod
  def load(src_file: str, tgt_file):
    examples: List[Tuple[str, str, str]] = []
    with open(src_file, 'r') as sfin, open(tgt_file, 'r') as tfin:
      for s in sfin:
        s = s.rstrip('\n')
        cp = s.find('col :')
        assert cp != -1, 'TAPEX file format error'
        context = s[:cp].strip()
        table = s[cp:]
        tgt = tfin.readline().strip()
        examples.append((context, table, tgt))
    return examples

  @staticmethod
  def linearize(header: List[str], data: List[List[str]]) -> str:
    header_str = 'col : ' + ' | '.join(header)
    row_strs: List[str] = [header_str]
    for row_idx, row in enumerate(data):
      row_str = f'row {row_idx + 1} : ' + ' | '.join(row)
      row_strs.append(row_str)
    final = ' '.join(row_strs)
    return final

  def convert_to_tabert_format(self, output_path: str):
    num_mentions: List[int] = []
    with open(output_path, 'w') as fout:
      for line_id, (context, table, target) in tqdm(enumerate(self.examples)):
        # parse table
        table_header: List[str] = list(map(lambda x: x.strip(),
                                           table.split('row 1 :', 1)[0].rstrip()[len('col :'):].split('|')))
        table_data: List[List[str]] = []
        row_ind = 1
        while True:
          i_ind = table.find(f'row {row_ind} :')
          ip1_ind = table.find(f'row {row_ind + 1} :')
          if ip1_ind == -1:  # last row
            row: List[str] = list(map(lambda x: x.strip(),
                                      table[i_ind:].split(':', 1)[1].strip().split('|')))
          else:
            row: List[str] = list(map(lambda x: x.strip(),
                                      table[i_ind:ip1_ind].split(':', 1)[1].strip().split('|')))
            row_ind += 1
          assert len(row) == len(table_header), f'TAPEX file format error {row} {table_header} {table}'
          table_data.append(row)
          if ip1_ind == -1:
            break

        # overall dict
        td = {
          'uuid': f'tapex_{line_id}',
          'table': {'caption': '', 'header': [], 'data': table_data, 'data_used': [], 'used_header': []},
          'context_before': [context],
          'context_after': [],
          'context_before_mentions': [],
          'context_before_mentions_cells': [],
          'answers': [target]  # TODO: split into multiple answers?
        }

        # header dict
        for cname in table_header:
          td['table']['header'].append({
            'name': cname,
            'name_tokens': None,
            'type': 'text',
            'sample_value': {'value': None, 'tokens': [], 'ner_tags': []},
            'sample_value_tokens': None,
            'is_primary_key': False,
            'foreign_key': None,
            'used': False,
            'value_used': False,
          })

        # compute overlap
        mention_locations, mention_cells = self.get_mention_locations(context, [table_header] + table_data)
        mention_cells = [mention_cells[ml] for ml in mention_locations]
        td['context_before_mentions'].append(mention_locations)
        td['context_before_mentions_cells'].append(mention_cells)
        num_mentions.append(len(mention_locations))

        if line_id > 0 and line_id % 20000 == 0:
          print(f'avg #mentions {np.mean(num_mentions)}')
        fout.write(json.dumps(td) + '\n')

    print(f'avg #mentions {np.mean(num_mentions)}')
