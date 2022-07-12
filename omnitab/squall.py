from typing import Dict, Set, Tuple, List
import json
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from omnitab.dataset_utils import BasicDataset
from omnitab.wikitablequestions import WikiTQ


class Squall(BasicDataset):
  # manually verified
  KEYWORDS: Set[str] = {'select', 'from', 'where', 'by', 'count', 'limit', 'order', 'desc', 'id', 'and',
                        'in', 'asc', 'group', 'sum', 'max', 'abs', 'min', 'distinct', 'null', 'is', 'not',
                        'having', 'avg', 'or', 'present_ref', 'agg', 'union',
                        '=', '(', ')', '*', ',', '>', '<', '!=', '+', '-', '>=', '<=', '/'}

  def __init__(self, json_file: Path, wikitq: WikiTQ = None):
    self.wtqid2example, self.keyword2count = self.load(json_file, wikitq=wikitq)

  @staticmethod
  def load(filepath: Path, wikitq: WikiTQ = None, fromw: str = ' from w') -> Tuple[Dict[str, Dict], Dict[str, int]]:
    keyword2count: Dict[str, int] = defaultdict(lambda: 0)
    wtqid2example: Dict[str, Dict] = {}
    oob = count_fromw = 0
    with open(filepath, 'r') as fin:
      data = json.load(fin)
      for example in data:
        wtqid = example['nt']
        # use either WikiTQ or the preprocessed columns in SQUALL
        if wikitq:
          columns = wikitq.get_table(wikitq.wtqid2tableid[wtqid])[0]
        else:
          columns = [c[0] for c in example['columns']]
        nl: str = ' '.join(example['nl'])
        sql = []
        sql_kw: List[str] = []
        for t in example['sql']:
          if t[0] == 'Column':  # match the column index to the corresponding name
            ci = int(t[1].split('_', 1)[0][1:]) - 1
            if ci < len(columns):
              sql.append(columns[ci])
            else:  # TODO: squall annotation error?
              sql.append(t[1])
              oob += 1
          else:
            sql.append(t[1])
            if t[0] == 'Keyword':
              keyword2count[t[1]] += 1
              sql_kw.append(t[1])
        sql = ' '.join(sql)
        has_fromw = sql.find(fromw + ' ') >= 0 or sql.endswith(fromw)
        count_fromw += int(has_fromw)
        sql = sql.replace(fromw + ' ', ' ')
        if sql.endswith(fromw):
          sql = sql[:-len(fromw)]
        wtqid2example[wtqid] = {
          'raw': example,
          'sql_raw': example['sql'],
          'nl_raw': example['nl'],
          'sql_kw': sql_kw,
          'nl': nl,
          'sql': sql
        }
    print(f'total: {len(wtqid2example)}; column out of bound: {oob}; #examples with "{fromw}": {count_fromw}')
    return wtqid2example, keyword2count

  def gen_sql2nl_data(self, output_path: Path, wtqid2example: Dict[str, Dict] = None):
    used_wtqids: Set[str] = set()
    with open(output_path, 'w') as fout:
      for eid, (wtqid, example) in tqdm(enumerate(self.wtqid2example.items())):
        if wtqid2example and wtqid not in wtqid2example:
          continue
        used_wtqids.add(wtqid)
        sql = example['sql']
        nl = example['nl']
        table = {'caption': '', 'header': [], 'data': [], 'data_used': [], 'used_header': []}
        if wtqid2example:
            table = wtqid2example[wtqid]['table']
        td = {
          'uuid': f'squall_{eid}',
          'metadata': {
            'wtqid': wtqid,
            'sql': sql,
            'nl': nl,
          },
          'table': table,
          'context_before': [nl],
          'context_after': []
        }
        fout.write(json.dumps(td) + '\n')
    if wtqid2example:
      print(f'found sql for {len(used_wtqids)} out of {len(wtqid2example)}')
      print(f'example ids without sql {list(set(wtqid2example.keys()) - used_wtqids)[:10]}')

  def get_subset(self, wtq_prep_path: Path, output_path: Path):
    wtqid2example: Dict[str, Dict] = {}
    with open(wtq_prep_path, 'r') as fin:
      for l in fin:
        example = json.loads(l)
        wtqid = example['uuid']
        assert wtqid not in wtqid2example, 'duplicate wtqid'
        wtqid2example[wtqid] = example
    self.gen_sql2nl_data(output_path, wtqid2example=wtqid2example)
