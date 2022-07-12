from typing import List, Dict, Set, Tuple, Any
import json
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm
import numpy as np
from omnitab.dataset_utils import BasicDataset


class Totto(BasicDataset):
    def __init__(self, root_dir: Path):
        self.train_data = self.load(root_dir / 'totto_train_data.jsonl')
        self.dev_data = self.load(root_dir / 'totto_dev_data.jsonl')

    def load(self, filename: Path):
        data: List[Dict] = []
        with open(filename, 'r') as fin:
            for l in fin:
                data.append(json.loads(l))
        return data

    @staticmethod
    def dummy_cell(is_header: bool = False):
        return {
            'column_span': 1,
            'is_header': is_header,
            'row_span': 1,
            'value': '',
        }

    @staticmethod
    def is_valid_cell(cell):
        return cell['row_span'] == 1 and cell['column_span'] == 1

    @staticmethod
    def is_valid_header(cell):
        return cell['is_header'] and Totto.is_valid_cell(cell)

    @staticmethod
    def visualize(table, max_cell_len: int = 10):
        for row in table:
            for cell in row:
                print('{}\t'.format(str(cell['value'])[:max_cell_len].replace('\n', ' ')), end='')
            print('')

    @staticmethod
    def table_dict_to_list(table, strict: bool = False, miss_value: Any = None, truncate_by_first_row: bool = False):
        table_list: List[List[Dict]] = []
        if len(table) <= 0:
            return table_list
        total_num_rows = np.max(list(table.keys())) + 1
        total_num_cols = (np.max(list(table[0].keys())) + 1) if truncate_by_first_row else (
                np.max([c for r in table for c in table[r]]) + 1)
        for i in range(total_num_rows):
            table_list.append([])
            assert not strict or total_num_cols == len(
                table[i]), 'number of columns inconsistent between different rows'
            for j in range(total_num_cols):
                if i in table and j in table[i]:
                    table_list[-1].append(table[i][j])
                else:
                    table_list[-1].append(miss_value)
        if len(table[0]) < total_num_cols:
            Totto.visualize(table_list)
            raise Exception('header incomplete')
        return table_list

    @staticmethod
    def expand_table(table, highlighted_cells: Set[Tuple[int, int]] = set()):
        _table: Dict[int, Dict[int, Dict]] = defaultdict(dict)  # expanded table, row_idx -> col_idx -> cell
        _highlighted_cells: Set[Tuple[int, int]] = set()  # expanded highlighted cells
        idx_convert: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
        _row_idx = _col_idx = 0  # index of the expanded table
        row_idx = col_idx = 0  # index to the original collapsed table
        is_expanded = False
        while True:
            _has_table_row = _row_idx in _table
            has_table_row = row_idx < len(table)
            if not _has_table_row and not has_table_row:  # both expanded and original tables are exhausted
                break
            _col_idx = col_idx = 0  # reset col idx
            visited_col_idx = set()
            use_table = False  # whether or not the current row of the original table is used
            while True:
                if _has_table_row and _col_idx in _table[_row_idx]:  # expanded table first
                    visited_col_idx.add(_col_idx)
                    _col_idx += 1
                elif has_table_row and col_idx < len(table[row_idx]):  # original table second
                    cell = table[row_idx][col_idx]
                    rs = cell['row_span']
                    cs = cell['column_span']
                    if rs > 1 or cs > 1:
                        _cell = deepcopy(cell)
                        is_expanded = True
                    else:
                        _cell = cell
                    _cell['row_span'] = 1
                    _cell['column_span'] = 1
                    for i in range(rs):
                        for j in range(cs):
                            # the same position could be visited multiple times, which means the table might be corrupted
                            # but we ignore this because it's not a large portion and it's impossible to fix it unless we
                            # download the row tables from Wikipedia ourselves.
                            _table[_row_idx + i][_col_idx + j] = _cell
                            idx_convert[(row_idx, col_idx)].add((_row_idx + i, _col_idx + j))
                            visited_col_idx.add(_col_idx + j)
                            if (row_idx, col_idx) in highlighted_cells:
                                # TODO: some highlighted cells overflow because of the following truncate
                                _highlighted_cells.add((_row_idx + i, _col_idx + j))
                    _col_idx += cs
                    col_idx += 1
                    use_table = True
                else:  # exhauseted row
                    if has_table_row and len(table[row_idx]) == 0:  # make sure to mark empty row used
                        use_table = True
                    _row_idx += 1
                    row_idx += int(use_table)
                    break
        _table_list = Totto.table_dict_to_list(_table, miss_value=Totto.dummy_cell(is_header=False),
                                               truncate_by_first_row=True)
        return _table_list, _highlighted_cells, is_expanded, idx_convert

    @staticmethod
    def get_header_rows_and_merge(table: List[List[Dict]], merge_sym: str = ' // '):
        # find the number of header rows from the top of the table and merge
        header_rows = []
        for row_idx, row in enumerate(table):
            # a row is a header row if all cells are marked as header
            # otherwise, we treat is as a data row
            header_sub = [cell for cell in row if cell['is_header']]
            if len(header_sub) == len(row):
                header_rows.append(row_idx)
            else:
                break
        num_hr = len(header_rows)
        merged_hr: List[Dict] = []
        if num_hr >= 1:
            for col_idx, cell in enumerate(table[header_rows[0]]):
                _cell = deepcopy(cell)
                new_value = [table[row_idx][col_idx]['value'] for row_idx in header_rows]
                _cell['value'] = merge_sym.join(new_value)
                merged_hr.append(_cell)
        return num_hr, merged_hr

    def get_page_titles(self, split: str):
        data = getattr(self, '{}_data'.format(split))
        return set(e['table_page_title'] for e in data)

    def convert_to_tabert_format(self,
                                 split: str,
                                 output_path: Path):
        count = num_rows = num_cols = num_used_rows = num_used_cols = 0
        data = getattr(self, '{}_data'.format(split))

        cases = {'inner_header': 0, 'no_header': 0, 'no_data': 0, 'fake_header': 0,
                 'multi_header': 0, 'partial_header': 0, 'expand': 0}
        numrows2count = defaultdict(lambda: 0)
        numusedcells2count = defaultdict(lambda: 0)
        nummentions2count = defaultdict(lambda: 0)
        find_mention_ratios: List[float] = []
        num_sent_ann: List[int] = []

        with open(output_path, 'w') as fout, open(str(output_path) + '.raw', 'w') as rfout:
            for example in tqdm(data):
                td = {
                    'uuid': None,
                    'metadata': {
                        'page_title': example['table_page_title'],
                        'page_url': example['table_webpage_url'],
                        'section_title': example['table_section_title'],
                        'section_text': example['table_section_text'],
                        'sentence_annotations': example['sentence_annotations'],
                    },
                    'table': {'caption': '', 'header': [], 'data': [], 'data_used': [], 'used_header': []},
                    'context_before': [],
                    'context_before_mentions': [],
                    'context_before_mentions_cells': [],
                    'context_after': []
                }

                num_sent_ann.append(len(example['sentence_annotations']))
                td['uuid'] = 'totto_{}'.format(example['example_id'])
                context = example['sentence_annotations'][0]['final_sentence']  # use the first annotated sentence
                table = example['table']
                highlighted_cells = set(map(tuple, example['highlighted_cells']))  # dedup
                td['context_before'].append(context)
                mention_locations, ml2cells = self.get_mention_locations(
                    context, [[c['value'] for c in r] for r in table], highlighted_cells)
                td['context_before_mentions'].append(mention_locations)
                numusedcells2count[len(highlighted_cells)] += 1
                nummentions2count[len(mention_locations)] += 1
                find_mention_ratios.append(len(mention_locations) / (len(highlighted_cells) or 1))

                # expand table
                expand_table, expand_highlighted_cells, is_expand, idx_convert = Totto.expand_table(table, highlighted_cells)
                cases['expand'] += int(is_expand)
                ml2cells = {k: list(set(_i for i in cs for _i in idx_convert[i])) for k, cs in ml2cells.items()}

                # merge header
                row2count: Dict[int, int] = defaultdict(lambda: 0)
                col2rows: Dict[int, Set] = defaultdict(set)
                num_hr, merged_header = Totto.get_header_rows_and_merge(expand_table)
                cases['multi_header'] += int(num_hr > 1)
                if num_hr <= 0:  # table must contain at least one header row
                    cases['no_header'] += 1
                    continue
                if len(expand_table) < num_hr + 1:  # table must contain at least one data row
                    cases['no_data'] += 1
                    continue
                for r, c in expand_highlighted_cells:
                    row2count[r] += 1
                    col2rows[c].add(r)
                num_rows += len(expand_table)
                num_cols += len(expand_table[0])
                num_used_rows += len(row2count)
                num_used_cols += len(col2rows)
                # highlighted cells could be header rows, which mean that header rows are not a real header.
                # See example with id 3182391131405542101 for an example
                fake_header = np.any([i in row2count for i in range(num_hr)])
                cases['fake_header'] += int(fake_header)

                # extract data and data_used
                inner_header = False
                partial_header = False
                for row in expand_table[num_hr:]:
                    header_sub = [cell['value'] for cell in row if cell['is_header']]
                    if len(header_sub) == len(row):
                        inner_header = True
                    elif len(header_sub) > 0:
                        partial_header = True
                    r = [cell['value'] for cell in row]
                    td['table']['data'].append(r)
                cases['inner_header'] += int(inner_header)
                cases['partial_header'] += int(partial_header)
                td['table']['data_used'] = sorted([(r - num_hr, c) for r, c in expand_highlighted_cells if r >= num_hr])
                numrows2count[len(td['table']['data'])] += 1

                # link mention with table data
                td['context_before_mentions_cells'].append(
                    [[(r - num_hr, c) for r, c in ml2cells[ml] if r >= num_hr] for ml in mention_locations])

                # extract column name
                for cell in merged_header:
                    td['table']['header'].append({
                        'name': cell['value'],
                        'name_tokens': None,
                        'type': 'text',
                        'sample_value': {'value': None, 'tokens': [], 'ner_tags': []},
                        'sample_value_tokens': None,
                        'is_primary_key': False,
                        'foreign_key': None,
                        'used': False,
                        'value_used': False
                    })

                # extract column type, value, and used
                for col_idx, header in enumerate(td['table']['header']):
                    if col_idx in col2rows:  # highlight column
                        row_idx = random.choice(list(col2rows[col_idx]))
                        header['used'] = len([i for i in col2rows[col_idx] if i < num_hr]) > 0  # use the header
                        header['value_used'] = len([i for i in col2rows[col_idx] if i >= num_hr]) > 0  # use the values
                        value = expand_table[row_idx][col_idx]['value']
                    else:  # sample from all data
                        all_values = [expand_table[i][col_idx]['value']
                                      for i in range(0 if fake_header else num_hr, len(expand_table))
                                      if expand_table[i][col_idx]['value']]  # choose non-empty values
                        value = random.choice(all_values) if len(all_values) > 0 else None
                    header['type'] = 'real' if Totto.is_number(value) else 'text'
                    header['sample_value']['value'] = value

                # write
                fout.write(json.dumps(td) + '\n')
                rfout.write(json.dumps(example) + '\n')
                count += 1
        print('total count {}, used rows {}/{}, used columns {}/{}'.format(
            count, num_used_rows, num_rows, num_used_cols, num_cols))
        print('out of {}, {}'.format(len(data), cases))
        print(f'#rows -> count {sorted(numrows2count.items())}')
        print(f'#used cells -> count {sorted(numusedcells2count.items())}')
        print(f'#mentions -> count {sorted(nummentions2count.items())}')
        print(f'find mention ratio {np.mean(find_mention_ratios)}')
        print(f'#sentence annotations {np.mean(num_sent_ann)}')
