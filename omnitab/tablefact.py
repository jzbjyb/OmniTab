from typing import List, Dict, Set, Tuple
import json
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm
import csv
import numpy as np
from omnitab.dataset_utils import BasicDataset


class TableFact(BasicDataset):
    def __init__(self, root_dir: Path):
        self.table_dir = root_dir / 'all_csv'
        self.full_data = self.load(root_dir / 'full_cleaned.json')
        self.train_data = self.partition(root_dir / 'train_examples.json')
        self.dev_data = self.partition(root_dir / 'val_examples.json')
        self.test_data = self.partition(root_dir / 'test_examples.json')

    def load(self, example_file: Path):
        with open(example_file, 'r') as fin:
            data = json.load(fin)
            return data

    def partition(self, example_file: Path):
        with open(example_file, 'r') as fin:
            data = json.load(fin)
            return {k: self.full_data[k] for k in data}

    @staticmethod
    def get_table(filename: str):
        rows = []
        with open(filename, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter='#')
            for row in csv_reader:
                rows.append(row)
        header = rows[0]  # assume the first row is header
        data = rows[1:]
        return header, data

    @staticmethod
    def parse_context(context: str):
        tok_idx = 0
        raw_sentence: List[str] = []
        mentions: List[Tuple[int, int]] = []  # char index of mentions
        mentions_cells: List[Tuple[int, int]] = []
        for i, piece in enumerate(context.split('#')):
            if i % 2 == 1:  # entity
                entity, idx = piece.rsplit(';', 1)
                row_ind, col_ind = idx.split(',')
                row_ind, col_ind = int(row_ind), int(col_ind)
                if row_ind != -1 and col_ind != -1:
                    mentions.append((tok_idx, tok_idx + len(entity)))
                    mentions_cells.append((row_ind, col_ind))
                raw_sentence.append(entity)
                tok_idx += len(entity)
            else:  # no entity
                raw_sentence.append(piece)
                tok_idx += len(piece)
        return ''.join(raw_sentence), mentions, mentions_cells

    def get_page_ids(self, split: str):
        data = getattr(self, '{}_data'.format(split))
        return set(tid.split('-')[1] for tid in data)

    def convert_to_tabert_format(self, split: str, output_path: Path):
        count = num_rows = num_cols = num_used_rows = num_used_cols = 0
        numrows2count = defaultdict(lambda: 0)
        numusedcells2count = defaultdict(lambda: 0)
        nummentions2count = defaultdict(lambda: 0)
        find_mention_ratios: List[float] = []
        data = getattr(self, '{}_data'.format(split))
        with open(output_path, 'w') as fout:
            for table_id in tqdm(data):
                example = data[table_id]
                caption = example[3]
                # parse table
                header, table_data = self.get_table(self.table_dir / table_id)
                numrows2count[len(table_data)] += 1
                # get types
                header_types = ['real' if self.is_number(cell.lower().strip()) else 'text'
                                for cell in table_data[0]] if len(table_data) > 0 else ['text'] * len(header)
                assert len(header_types) == len(header)
                for context_id, (context, label) in enumerate(zip(example[0], example[1])):
                    if not label:
                        continue
                    context, mentions, mentions_cells = self.parse_context(context)
                    highlighted_cells = sorted(list(set(mentions_cells)))
                    mentions_cells = [[(r - 1, c)] if r > 0 else [] for r, c in mentions_cells]
                    assert len(mentions) == len(mentions_cells)
                    col2rows: Dict[int, Set[int]] = defaultdict(set)
                    row2count: Dict[int, int] = defaultdict(lambda: 0)
                    data_used: List[Tuple[int, int]] = []
                    for ri, ci in highlighted_cells:
                        col2rows[ci].add(ri)
                        row2count[ri] += 1
                        if ri > 0:  # skip header
                            data_used.append((ri - 1, ci))
                    td = {
                        'uuid': f'tablefact_{split}_{table_id}_{context_id}',
                        'table': {'caption': caption, 'header': [], 'data': table_data, 'data_used': data_used,
                                  'used_header': []},
                        'context_before': [context],
                        'context_before_mentions': [mentions],
                        'context_before_mentions_cells': [mentions_cells],
                        'context_after': []
                    }
                    num_rows += len(table_data)
                    num_used_rows += len(set(row2count.keys()) - {0})  # remove header
                    numusedcells2count[len(highlighted_cells)] += 1
                    nummentions2count[len(mentions)] += 1
                    find_mention_ratios.append(len(mentions) / (len(highlighted_cells) or 1))

                    # extract value and used
                    for col_ind, (cname, ctype) in enumerate(zip(header, header_types)):
                        used_rows = list(col2rows[col_ind] - {0})  # remove the header
                        td['table']['header'].append({
                            'name': cname,
                            'name_tokens': None,
                            'type': ctype,
                            'sample_value': {'value': None, 'tokens': [], 'ner_tags': []},
                            'sample_value_tokens': None,
                            'is_primary_key': False,
                            'foreign_key': None,
                            'used': 0 in col2rows[col_ind],
                            'value_used': len(used_rows) > 0,
                        })
                        num_used_cols += int(len(col2rows[col_ind]) > 0)
                        if len(used_rows) > 0:
                            value = table_data[random.choice(used_rows) - 1][col_ind]  # remove the header
                        else:
                            value = table_data[random.randint(0, len(table_data) - 1)][col_ind]
                        td['table']['header'][-1]['sample_value']['value'] = value
                    num_cols += len(td['table']['header'])
                    count += 1
                    fout.write(json.dumps(td) + '\n')
        print('total count {}, used rows {}/{}, used columns {}/{}'.format(
            count, num_used_rows, num_rows, num_used_cols, num_cols))
        print(f'#rows -> count {sorted(numrows2count.items())}')
        print(f'#used cells -> count {sorted(numusedcells2count.items())}')
        print(f'#mentions -> count {sorted(nummentions2count.items())}')
        print(f'find mention ratio {np.mean(find_mention_ratios)}')
