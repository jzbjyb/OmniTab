from typing import List, Dict, Set
import json
from pathlib import Path
from tqdm import tqdm
import re
from enum import Enum
import numpy as np
from omnitab.dataset_utils import BasicDataset


class ColumnType(Enum):
    ENTITY = 'entity'
    NON_ENTITY = 'non-entity'


class TurlData(BasicDataset):
    def __init__(self, root_dir: Path):
        self.table_dir = root_dir / 'all_csv'
        self.train_data = self.load(root_dir / 'train_tables_100k.jsonl')  # load a samll subset
        self.dev_data = self.load(root_dir / 'dev_tables.jsonl')
        self.test_data = self.load(root_dir / 'test_tables.jsonl')

    def load(self, example_file: Path):
        data: List[Dict] = []
        overflow = 0
        with open(example_file, 'r') as fin:
            for l in tqdm(fin, desc=f'loading {example_file}'):
                l = json.loads(l)
                l['context'], of = self.get_context(l['pgTitle'], l['sectionTitle'], l['tableCaption'])
                overflow += int(of)
                l['columns'], l['table_data'] = self.load_table_index_by_column(l)
                data.append(l)
        print(f'{example_file} has {overflow} overflow examples')
        return data

    @staticmethod
    def get_context(page_title: str, section_title: str, caption: str, max_length: int = 50, add_punc: bool = True) -> str:
        overflow = False
        li = [page_title, section_title, caption]
        for i in range(len(li)):  # preprocess text
            li[i] = li[i].strip()
            if add_punc:
                li[i] = BasicDataset.add_punc(li[i])
        page_title, section_title, caption = li
        toks = []
        if len(page_title) > 0:
            sp = page_title.split(' ')
            overflow = overflow or len(sp) > max_length
            toks += sp[:max_length]
        if len(section_title) > 0:
            sp = section_title.split(' ')
            overflow = overflow or len(sp) > max_length
            toks += sp[:max_length]
        if len(caption) > 0 and caption != section_title:
            sp = caption.split(' ')
            overflow = overflow or len(sp) > max_length
            toks += sp[:max_length]
        return ' '.join(toks), overflow

    @staticmethod
    def load_table_index_by_column(example: Dict):
        num_columns = len(example['processed_tableHeaders'])
        num_rows = len(example['tableData'])
        assert num_columns == len(example['column_type']), '#columns inconsistent'
        assert 0 <= example['subject_column'] < num_columns, 'subject column index out of bound'
        columns: List[Dict] = [{'name': cn,
                                'entity_type': ColumnType.ENTITY if ct == 0 else ColumnType.NON_ENTITY,
                                'value_type': None,
                                'is_subject': example['subject_column'] == col_idx,
                                'data': []}
                               for col_idx, (cn, ct) in enumerate(zip(example['processed_tableHeaders'], example['column_type']))]
        assert num_rows == len(example['entityCell']), '#rows inconsistent'
        table_data: List[List[str]] = []
        for row, row_is_entity in zip(example['tableData'], example['entityCell']):
            assert len(row) == num_columns and len(row_is_entity) == num_columns, '#cells in a row inconsistent'
            table_data.append([])
            for col_idx, (cell, is_entity) in enumerate(zip(row, row_is_entity)):
                cell['is_entity'] = is_entity
                columns[col_idx]['data'].append(cell)
                if columns[col_idx]['value_type'] is None:  # set value type using the first row
                    columns[col_idx]['value_type'] = 'real' if TurlData.is_number(cell['text']) else 'text'
                table_data[-1].append(str(cell['text']))
        return columns, table_data

    @staticmethod
    def iter_sub_obj(example: Dict, only_entity: bool = True, min_entity_pair: int = 3):
        columns = example['columns']
        sub_col_idx = example['subject_column']
        sub_col_type = columns[sub_col_idx]['entity_type']
        # TODO: add filtering (at least 3 valid sub-obj entity pairs)
        for obj_col_idx in range(len(columns)):
            if obj_col_idx == sub_col_idx:
                continue
            obj_col_type = columns[obj_col_idx]['entity_type']
            if only_entity and not (sub_col_type == ColumnType.ENTITY and obj_col_type == ColumnType.ENTITY):
                continue
            sub_cells = []
            obj_cells = []
            for sub_cell, obj_cell in zip(columns[sub_col_idx]['data'], columns[obj_col_idx]['data']):
                if only_entity and not (sub_cell['is_entity'] and obj_cell['is_entity']):
                    continue
                sub_cells.append(sub_cell)
                obj_cells.append(obj_cell)
            if len(sub_cells) < min_entity_pair:
                 continue
            yield sub_col_idx, sub_cells, obj_col_idx, obj_cells

    def get_page_titles(self, split: str):
        data = getattr(self, '{}_data'.format(split))
        return set(e['pgTitle'] for e in data)

    def convert_to_tabert_format(self, split: str, output_path: Path, task: str, avoid_titles: Set[str] = set()):
        assert task in {'cell_filling', 'row_population', 'schema_augmentation'}
        count = num_rows = num_cols = 0
        num_rows_per_example: List[int] = []
        data = getattr(self, '{}_data'.format(split))
        with open(output_path, 'w') as fout:
            for idx, example in tqdm(enumerate(data)):
                if example['pgTitle'] in avoid_titles:
                    continue
                td = {
                    'uuid': f"turl_{task}_{split}_{example['_id']}",
                    'table': {
                        'caption': example['tableCaption'],
                        'header': [],
                        'data': example['table_data'],
                        'used_header': []},
                    'context_before': [example['context']],
                    'context_after': []}
                num_rows += len(example['table_data'])
                num_cols += len(example['columns'])

                if task == 'cell_filling':  # sample subject-object pairs
                    for sub_col_idx, sub_cells, obj_col_idx, obj_cells in self.iter_sub_obj(example):
                        num_rows_per_example.append(len(sub_cells))
                        td['table']['header'] = []  # init header with empty list
                        for sub_cell, obj_cell in zip(sub_cells, obj_cells):
                            td['table']['header'].append({  # append subject
                                'name': example['columns'][sub_col_idx]['name'],
                                'name_tokens': None,
                                'type': example['columns'][sub_col_idx]['value_type'],
                                'sample_value': {'value': sub_cell['text'], 'tokens': [], 'ner_tags': []},
                                'sample_value_tokens': None,
                                'is_primary_key': False,
                                'foreign_key': None,
                                'used': False,
                                'value_used': False,
                            })
                            td['table']['header'].append({  # append object
                                'name': example['columns'][obj_col_idx]['name'],
                                'name_tokens': None,
                                'type': example['columns'][obj_col_idx]['value_type'],
                                'sample_value': {'value': obj_cell['text'], 'tokens': [], 'ner_tags': []},
                                'sample_value_tokens': None,
                                'is_primary_key': False,
                                'foreign_key': None,
                                'used': False,
                                'value_used': False,
                            })
                        count += 1
                        fout.write(json.dumps(td) + '\n')
                elif task == 'schema_augmentation':
                    for column in example['columns']:
                        td['table']['header'].append({
                            'name': column['name'],
                            'name_tokens': None,
                            'type': column['value_type'],
                            'sample_value': {'value': None, 'tokens': [], 'ner_tags': []},
                            'sample_value_tokens': None,
                            'is_primary_key': False,
                            'foreign_key': None,
                            'used': False,
                            'value_used': False,
                        })
                    count += 1
                    fout.write(json.dumps(td) + '\n')
                else:
                    raise NotImplementedError
            print(f'total count {count}, #columns {num_cols}, #rows {num_rows} #rows per example {np.mean(num_rows_per_example)}')
