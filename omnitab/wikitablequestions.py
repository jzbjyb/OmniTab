from typing import List, Dict, Set, Tuple, Any
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import csv
from omnitab.dataset_utils import BasicDataset


class WikiTQ(BasicDataset):
    def __init__(self, root_dir: Path):
        # TODO: previously we were using split 5
        self.train_data = self.load(root_dir / 'data' / 'random-split-1-train.tsv')
        self.dev_data = self.load(root_dir / 'data' / 'random-split-1-dev.tsv')
        self.test_data = self.load(root_dir / 'data' / 'pristine-unseen-tables.tsv')
        self.table_root_dir = root_dir
        self.misc_file = root_dir / 'misc' / 'table-metadata.tsv'
        self.wtqid2tableid = self.get_wtqid2tableid()
        self.tableid2pageid = self.get_tableid2pageid(self.misc_file)
        self.wtqid2pageid = {wtqid: self.tableid2pageid[tableid] for wtqid, tableid in self.wtqid2tableid.items()
                             if tableid in self.tableid2pageid}
        self.pageid2wtqids = self.get_pageid2wtqids()

    def get_table(self, table_id: str):
        filename = self.table_root_dir / table_id
        with open(filename, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter=',', doublequote=False, escapechar='\\')
            header = [str(c) for c in next(csv_reader)]
            data = [[str(c) for c in row] for row in csv_reader]
            for row in data:
                assert len(row) == len(header), f'{filename} format error for line {row} with #{len(row)}'
            header_types = ['text'] * len(header)
            if len(data) > 0:
                header_types = ['real' if WikiTQ.is_number(cell) else 'text' for cell in data[0]]
            return header, header_types, data

    @staticmethod
    def get_tableid2pageid(mis_file: Path):
        tableid2pageid: Dict[str, str] = {}
        with open(mis_file, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter='\t')
            _ = next(csv_reader)  # skip tsv head
            for row in csv_reader:
                tableid, pageid = row[:2]
                assert tableid not in tableid2pageid, 'duplicate table id'
                tableid2pageid[tableid] = pageid
        return tableid2pageid

    @staticmethod
    def get_pageid2oldid(mis_file: Path, pageids: List[str] = None):
        pageid2oldid: Dict[str, str] = {}
        with open(mis_file, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter='\t')
            _ = next(csv_reader)  # skip tsv head
            for row in csv_reader:
                tableid, pageid, oldid = row[:3]
                if not pageids or pageid in pageids:
                    pageid2oldid[pageid] = oldid
        return pageid2oldid

    def get_wtqid2tableid(self):
        wtqid2tableid: Dict[str, str] = {}
        for split in ['train', 'dev', 'test']:
            data = getattr(self, f'{split}_data')
            wtqid2tableid.update({e['id']: e['table_id'] for e in data})
        return wtqid2tableid

    def get_pageid2wtqids(self):
        pageid2wtqids: Dict[str, List[str]] = defaultdict(list)
        for wtqid, tableid in self.wtqid2tableid.items():
            pageid = self.tableid2pageid[tableid]
            pageid2wtqids[pageid].append(wtqid)
        return pageid2wtqids

    def load(self, filename: Path):
        numans2count: Dict[int, int] = defaultdict(lambda: 0)
        data: List[Dict] = []
        with open(filename, 'r') as fin:
            #csv_reader = csv.reader(fin, delimiter='\t')
            #_ = next(csv_reader)  # skip tsv head
            _ = fin.readline()  # skip tsv head
            #for row in csv_reader:
            for row in fin:
                id, utterance, table_id, targets = row.rstrip('\n').split('\t')
                targets = targets.split('|')
                numans2count[len(targets)] += 1
                data.append({'id': id, 'utterance': utterance, 'table_id': table_id, 'targets': targets})
        return data

    def convert_to_tabert_format(self, split: str, output_path: Path):
        count = num_rows = num_cols = num_used_cols = 0
        data = getattr(self, '{}_data'.format(split))
        with open(output_path, 'w') as fout:
            for idx, example in tqdm(enumerate(data)):
                td = {
                    'uuid': None,
                    'pageid': self.tableid2pageid[example['table_id']],
                    'table': {'caption': '', 'header': [], 'data': [], 'used_header': []},
                    'context_before': [],
                    'context_after': []
                }
                td['uuid'] = f'wtq_{split}_{idx}'
                question = example['utterance']
                table_header, header_types, table_data = self.get_table(example['table_id'])

                td['context_before'].append(question)
                td['table']['data'] = table_data
                num_rows += len(td['table']['data'])

                # extract column name
                for cname, ctype in zip(table_header, header_types):
                    td['table']['header'].append({
                        'name': cname,
                        'name_tokens': None,
                        'type': ctype,
                        'sample_value': {'value': None, 'tokens': [], 'ner_tags': []},
                        'sample_value_tokens': None,
                        'is_primary_key': False,
                        'foreign_key': None,
                        'used': False,
                        'value_used': False,
                    })
                num_cols += len(td['table']['header'])

                # TODO: use string matching to find used columns and cells
                count += 1
                fout.write(json.dumps(td) + '\n')
        print('total count {}, used columns {}/{}'.format(count, num_used_cols, num_cols))
