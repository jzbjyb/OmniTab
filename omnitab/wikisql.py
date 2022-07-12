from typing import List, Dict
import json
from pathlib import Path
import random
from tqdm import tqdm
import re
from collections import defaultdict
import logging
import numpy as np
from omnitab.dataset_utils import BasicDataset


class WikiSQL(BasicDataset):
    AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    COND_OPS = ['=', '>', '<', 'OP']

    def __init__(self, root_dir: Path):
        for split in ['train', 'dev', 'test']:
            setattr(self, f'{split}_data', self.load(root_dir / f'{split}.jsonl', root_dir / f'{split}.tables.jsonl'))

    def load(self, example_file: str, table_file: str):
        id2table: Dict[str, Dict] = {}
        with open(table_file, 'r') as fin:
            for l in fin:
                l = json.loads(l)
                id2table[l['id']] = l
        data: List[Dict] = []
        with open(example_file, 'r') as fin:
            for l in fin:
                l = json.loads(l)
                l['table'] = id2table[l['table_id']]
                data.append(l)
        return data

    @staticmethod
    def convert_to_human_readable(sel, agg, columns, conditions):
        # Make SQL query string. Based on https://github.com/salesforce/WikiSQL/blob/c2ed4f9b22db1cc2721805d53e6e76e07e2ccbdc/lib/query.py#L10
        rep = 'SELECT {agg} {sel} FROM table'.format(
            agg=WikiSQL.AGG_OPS[agg], sel=columns[sel] if columns is not None else 'col{}'.format(sel))
        if conditions:
            rep += ' WHERE ' + ' AND '.join(
                ['{} {} {}'.format(columns[i], WikiSQL.COND_OPS[o], v) for i, o, v in conditions])
        return ' '.join(rep.split())

    @staticmethod
    def normalize_rows(rows: List[List]):
        return [[str(cell).replace(' ,', ',') for cell in row] for row in rows]

    def get_page_ids(self, split: str):
        data = getattr(self, '{}_data'.format(split))
        return set(e['table_id'].split('-')[1] for e in data)

    def convert_to_tabert_format(self, split: str, output_path: Path, add_sql: bool = False):
        count = num_rows = num_cols = num_used_cols = 0
        numrows2count = defaultdict(lambda: 0)
        numusedcells2count = defaultdict(lambda: 0)
        nummentions2count = defaultdict(lambda: 0)
        find_mention_ratios: List[float] = []
        condition_found: List[bool] = []
        data = getattr(self, '{}_data'.format(split))
        with open(output_path, 'w') as fout:
            for idx, example in tqdm(enumerate(data)):
                td = {
                    'uuid': f'wsql_{split}_{idx}',
                    'table': {'caption': '', 'header': [], 'data': [], 'data_used': [], 'used_header': []},
                    'context_before': [],
                    'context_after': [],
                    'context_before_mentions': [],
                    'context_before_mentions_cells': [],
                }
                question = example['question']
                td['context_before'].append(question)
                td['table']['data'] = self.normalize_rows(example['table']['rows'])
                num_rows += len(td['table']['data'])
                numrows2count[len(td['table']['data'])] += 1

                # extract column name
                for cname, ctype in zip(example['table']['header'], example['table']['types']):
                    td['table']['header'].append({
                        'name': '',
                        'name_tokens': None,
                        'type': 'text',
                        'sample_value': {'value': None, 'tokens': [], 'ner_tags': []},
                        'sample_value_tokens': None,
                        'is_primary_key': False,
                        'foreign_key': None,
                        'used': False,
                        'value_used': False,
                    })
                    td['table']['header'][-1]['name'] = cname
                    td['table']['header'][-1]['type'] = ctype if ctype == 'text' else 'real'
                num_cols += len(td['table']['header'])

                # extract value, used, and data_used
                sql = example['sql']
                num_used_cols += len(set([sql['sel']] + [c[0] for c in sql['conds']]))
                td['table']['header'][sql['sel']]['used'] = True
                for col_ind, _, cond in sql['conds']:
                    td['table']['header'][col_ind]['used'] = True
                    cond = str(cond).replace(' ,', ',')
                    conds = [cond, re.sub(r'\.0$', '', cond), cond.title()]  # match candidates
                    column_data = [row[col_ind] for row in td['table']['data']]
                    value = None
                    row_ind = None
                    for cond in conds:
                        for i, v in enumerate(column_data):
                            if cond == v or cond.lower() == v.lower() or self.float_eq(cond, v):
                                value = v
                                row_ind = i
                                break
                        if value is not None:
                            break
                    if value is None:
                        logging.warn(f'{conds} not in data {column_data} for {question}')
                        td['table']['header'][col_ind]['sample_value']['value'] = random.choice(column_data)
                        condition_found.append(False)
                    else:
                        td['table']['header'][col_ind]['sample_value']['value'] = value
                        td['table']['header'][col_ind]['value_used'] = True
                        td['table']['data_used'].append((row_ind, col_ind))
                        condition_found.append(True)
                td['table']['data_used'] = sorted(list(set(td['table']['data_used'])))

                # extract mentions
                hl_cells = td['table']['data_used']
                mention_locations, mention_cells = self.get_mention_locations(question, td['table']['data'], set(hl_cells))
                mention_cells = [mention_cells[ml] for ml in mention_locations]
                td['context_before_mentions'].append(mention_locations)
                td['context_before_mentions_cells'].append(mention_cells)
                numusedcells2count[len(hl_cells)] += 1
                nummentions2count[len(mention_locations)] += 1
                find_mention_ratios.append(len(mention_locations) / (len(hl_cells) or 1))

                if add_sql:  # extract sql
                    td['sql'] = self.convert_to_human_readable(
                        sql['sel'], sql['agg'], example['table']['header'], sql['conds'])

                count += 1
                fout.write(json.dumps(td) + '\n')
        print('total count {}, used columns {}/{}'.format(count, num_used_cols, num_cols))
        print(f'#rows -> count {sorted(numrows2count.items())}')
        print(f'#used cells -> count {sorted(numusedcells2count.items())}')
        print(f'#mentions -> count {sorted(nummentions2count.items())}')
        print(f'find mention ratio {np.mean(find_mention_ratios)}')
        print(f'find condition ratio {np.mean(condition_found)}')
