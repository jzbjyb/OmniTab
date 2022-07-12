from typing import Dict, List, Tuple, Union
import os
import json
import sqlite3
import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
from table_bert import Table, Column
from omnitab.dataset_utils import BasicDataset


class Spider(BasicDataset):
    def __init__(self, root_dir: Path):
        self.train_file = root_dir / 'train_spider.json'
        self.dev_file = root_dir / 'dev.json'
        self.db_file = root_dir / 'tables.json'
        self.db_path_pattern = os.path.join(str(root_dir), 'database') + '/{}/{}.sqlite'
        self.load()

    @staticmethod
    def convert_to_table(table_name: str, columns: List[Tuple[str, str, Union[None, str]]]) -> Table:
        table = Table(
            id=table_name,
            header=[Column(cn, ct, sample_value=value if value is not None else '') for cn, ct, value in columns],
            data=[]
        )
        return table

    def load(self):
        dbs = json.load(open(self.db_file, 'r'))
        self.dbs: Dict[str, Dict] = {}
        for db in dbs:
            self.dbs[db['db_id']] = db
        self.train_data = json.load(open(self.train_file, 'r'))
        self.dev_data = json.load(open(self.dev_file, 'r'))
        print('#DB {}, #train {}, #dev {}'.format(len(self.dbs), len(self.train_data), len(self.dev_data)))

    def load_database(self, db_id: str, table_name: str):
        db_path = self.db_path_pattern.format(db_id, db_id)
        con = sqlite3.connect(db_path)
        con.text_factory = lambda b: b.decode(errors='ignore')
        try:
            df = pd.read_sql_query('SELECT * from {}'.format(table_name), con)
        except Exception as e:
            print('bug in DB "{}" table "{}"'.format(db_id, table_name))
            raise e
        return df

    def sample_from_table_df(self, df):
        if len(df) == 0:
            return None
        return df.iloc[random.randint(0, len(df) - 1)]

    def sample_negative(self, split: str, output_path: Path, neg_num: int = 9):
        assert hasattr(self, 'db2table2column')
        all_tables = ['{}|||{}'.format(db, tn) for db in self.db2table2column for tn in self.db2table2column[db]]
        with open(output_path, 'w') as fout:
            for example in tqdm(getattr(self, '{}_data'.format(split))):
                db_id = example['db_id']
                question = example['question']
                for tn in self.db2table2column[db_id]:
                    # positive
                    fout.write('{}\t{}\t{}\t{}\n'.format(question, db_id, tn, 1))
                    # negative
                    samples = np.random.choice(all_tables, neg_num + len(self.db2table2column[db_id]), replace=False)
                    samples = [s for s in samples if not s.startswith(db_id + '|||')][:neg_num]
                    assert len(samples) == neg_num
                    for s in samples:
                        n_db_id, n_tn = s.split('|||', 1)
                        fout.write('{}\t{}\t{}\t{}\n'.format(question, n_db_id, n_tn, 0))

    def gen_sql2nl_data(self, split: str, output_path: Path):
        with open(output_path, 'w') as fout:
            for eid, example in tqdm(enumerate(getattr(self, '{}_data'.format(split)))):
                sql = example['query']
                nl = example['question']
                td = {
                    'uuid': f'spider_{eid}',
                    'metadata': {
                        'sql': sql,
                        'nl': nl,
                    },
                    'table': {'caption': '', 'header': [], 'data': [], 'data_used': [], 'used_header': []},
                    'context_before': [nl],
                    'context_after': []
                }
                fout.write(json.dumps(td) + '\n')

    def convert_to_tabert_format(self, split: str, output_path: Path):
        all_types = set()
        no_row_count = 0
        self.db2table2column: Dict[str, Dict[str, List[Tuple[str, str, Union[None, str]]]]] = {}
        for example in tqdm(getattr(self, '{}_data'.format(split))):
            db_id = example['db_id']
            db = self.dbs[db_id]
            if db_id in self.db2table2column:
                continue
            self.db2table2column[db_id] = {}
            for id, (tn, tno) in enumerate(zip(db['table_names'], db['table_names_original'])):
                self.db2table2column[db_id][tn] = []
                row = self.sample_from_table_df(self.load_database(db_id, tno))
                if row is None:
                    # raise Exception('DB "{}" table "{}" has no rows'.format(db_id, tno))
                    no_row_count += 1
                for cn, cno, ct in zip(db['column_names'], db['column_names_original'], db['column_types']):
                    all_types.add(ct)
                    # convert to tabert type
                    if ct in {'number', 'time'}:
                        ct = 'real'
                    else:
                        ct = 'text'
                    if cn[0] != id:
                        continue
                    assert cn[0] == cno[0], 'column_names and column_names_original are not consistent in terms of order'
                    cn, cno = cn[1], cno[1]
                    if row is None:
                        self.db2table2column[db_id][tn].append((cn, ct, None))
                    else:
                        self.db2table2column[db_id][tn].append((cn, ct, str(row[cno])))
        with open(output_path, 'w') as fout:
            json.dump(self.db2table2column, fout, indent=2)
        print('#tables without data {}'.format(no_row_count))
        print('All types {}'.format(all_types))
