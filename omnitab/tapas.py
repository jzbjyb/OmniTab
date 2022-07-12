from typing import List, Dict, Set
import json
from pathlib import Path
from tqdm import tqdm
import re
from omnitab.dataset_utils import BasicDataset


class TapasTables(BasicDataset):
    def __init__(self, root_dir: Path):
        self.data_path = root_dir / 'interactions.txtpb'

    def hash_text(self, text: str):
        text = re.sub('\s+', '', text.lower())  # lower and remove spaces
        text = re.sub(r'[^\w]', '', text)  # remove punc
        return text

    def _contains(self, seen: Set[str], text: str):
        text = self.hash_text(text)
        if text in seen:
            return True
        for s in seen:
            if text in s:
                return True
        return False

    def get_context(self, example):
        id2q: Dict[str, str] = {'TITLE': None, 'DESCRIPTION': None, 'SEGMENT_TITLE': None, 'SEGMENT_TEXT': None,
                                'CAPTION': None}
        for q in example.questions:
            if q.id not in id2q:
                raise ValueError('text key not found')
            id2q[q.id] = BasicDataset.add_punc(q.original_text)
        # first caption, then segment, then title (prefer desc than title)
        used: Set[str] = set()
        used_li: List[str] = []
        if id2q['CAPTION']:
            used.add(self.hash_text(id2q['CAPTION']))
            used_li.append(id2q['CAPTION'])
        for desc, title in [(id2q['SEGMENT_TEXT'], id2q['SEGMENT_TITLE']), (id2q['DESCRIPTION'], id2q['TITLE'])]:
            if desc and not self._contains(used, desc):
                used.add(self.hash_text(desc))
                used_li.append(desc)
            elif title and not self._contains(used, title):
                used.add(self.hash_text(title))
                used_li.append(title)
        return ' '.join(used_li), id2q['CAPTION']

    def get_table(self, example):
        header = [c.text for c in example.table.columns]
        data = [[c.text for c in row.cells] for row in example.table.rows]
        assert len(data) > 0
        for row in data:
            assert len(row) == len(header)
        header_type = ['real' if BasicDataset.is_number(c) else 'text' for c in data[0]]
        return header, header_type, data

    def convert_to_tabert_format(self, split: str, output_path: Path):
        from google.protobuf import text_format
        from table_bert.tapas_protos import interaction_pb2
        with open(self.data_path, 'r') as fin, open(output_path, 'w') as fout:
            for l in tqdm(fin):
                example = text_format.Parse(l, interaction_pb2.Interaction())
                header, header_type, data = self.get_table(example)
                context, caption = self.get_context(example)
                td = {
                    'uuid': f'tapas_{example.id}',
                    'table': {'caption': caption or '', 'header': [], 'data': data, 'data_used': [], 'used_header': []},
                    'context_before': [context],
                    'context_before_mentions': [],
                    'context_after': []
                }
                for column, column_type in zip(header, header_type):
                    td['table']['header'].append({
                        'name': column,
                        'name_tokens': None,
                        'type': column_type,
                        'sample_value': {'value': None, 'tokens': [], 'ner_tags': []},
                        'sample_value_tokens': None,
                        'is_primary_key': False,
                        'foreign_key': None,
                        'used': False,
                        'value_used': False
                    })
                fout.write(json.dumps(td) + '\n')
