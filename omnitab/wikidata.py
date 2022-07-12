import time
from typing import Dict, List, Set, Union, Tuple
from collections import defaultdict
from tqdm import tqdm
import requests

_topic2categories: Dict[str, List[str]] = {
  'Politics': ['Crime', 'Geography', 'Government', 'Law', 'Military', 'Policy',
               'Politics', 'Society', 'World'],
  'Culture': ['Entertainment', 'Events', 'History', 'Human behavior', 'Humanities',
              'Life', 'Culture', 'Mass media', 'Music', 'Organizations'],
  'Sports': ['Sports'],
  'People': ['People'],
  'Misc': ['Academic disciplines', 'Business', 'Concepts', 'Economy', 'Education',
           'Energy', 'Engineering', 'Food and drink', 'Health', 'Industry', 'Knowledge',
           'Language', 'Mathematics', 'Mind', 'Objects', 'Philosophy', 'Religion',
           'Nature', 'Science and technology', 'Universe']
}
topic2categories: Dict[str, List[str]] = {
  'Politics': ['Q6478924', 'Q1457673', 'Q54070', 'Q4026563', 'Q5850187', 'Q6277256',
               'Q4103183', 'Q1457756', 'Q7386634'],
  'Culture': ['Q6337045', 'Q7214908', 'Q1457595', 'Q6697416', 'Q6172603', 'Q5550747',
              'Q2944929', 'Q1458390', 'Q8255', 'Q5613113'],
  'Sports': ['Q1457982'],
  'People': ['Q4047087'],
  'Misc': ['Q6642719', 'Q6353120', 'Q5550686', 'Q9715089', 'Q4103249', 'Q8413436', 'Q4057258', 'Q5645580',
           'Q7486603', 'Q6528585', 'Q2945448', 'Q1458484', 'Q4619', 'Q6643238', 'Q6576895', 'Q1983674',
           'Q1457903', 'Q4049293', 'Q9795196', 'Q21079384']
}

class WikipediaCategory(object):
  PREFIX = 'Category:'
  PAGEID2OTHER_URL = 'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&pageids={}&format=json'
  PAGEID2CATE_URL = 'https://en.wikipedia.org/w/api.php?action=query&prop=categories&pageids={}&clshow=!hidden&cllimit=50&clprop=sortkey&format=json'
  REVID2CATE_URL = 'https://en.wikipedia.org/w/api.php?action=query&prop=categories&revids={}&clshow=!hidden&cllimit=50&clprop=sortkey&format=json'
  TITLE2CATE_URL = 'https://en.wikipedia.org/w/api.php?action=query&prop=categories&titles={}&clshow=!hidden&cllimit=50&clprop=sortkey&format=json'
  PAGEID2REDIRECT_URL = 'https://en.wikipedia.org/w/api.php?action=query&pageids={}&redirects&format=json'
  TITLE2OTHER_URL = 'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles={}&format=json'
  WIKIDATA_ID2NAME = 'https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids={}&languages=en&format=json'
  max_limit = 50
  wait = 1.0

  def __init__(self, parent2child_file: str = None, child2parent_file: str = None, format: str = 'external'):
    if parent2child_file:
      self.parent2child: Dict[str, List[str]] = {}
      self.child2parent: Dict[str, List[str]] = {}
      with open(parent2child_file, 'r') as fin:
        for l in tqdm(fin):
          assert format == 'external'
          parent, childs = l.strip().split(',', 1)
          childs = list(set(childs[2:-1].split(',')))  # remove duplicates
          self.parent2child[parent] = childs
          for child in childs:
            if child not in self.child2parent:
              self.child2parent[child] = []
            #assert parent not in self.child2parent[child], f'{parent} -> {child} appears multiple times'
            self.child2parent[child].append(parent)
    elif child2parent_file:
      self.parent2child: Dict[str, List[str]] = {}
      self.child2parent: Dict[str, List[str]] = {}
      with open(child2parent_file, 'r') as fin:
        for l in tqdm(fin):
          assert format == 'sling'
          child, parents = l.rstrip('\n').split('\t', 1)
          parents = list(set(parents.split(',')))  # remove duplicates
          self.child2parent[child] = parents
          for parent in parents:
            if parent not in self.parent2child:
              self.parent2child[parent] = []
            #assert child not in self.parent2child[parent], f'{parent} -> {child} appears multiple times'
            self.parent2child[parent].append(child)

  @classmethod
  def batcher(cls, examples: List, max_limit: int = None, wait: int = None, disable: bool = False):
    max_limit = max_limit or cls.max_limit
    wait = wait or cls.wait
    for i in tqdm(range(0, len(examples), max_limit), disable=disable):
      yield examples[i:i + max_limit]
      time.sleep(wait)

  @classmethod
  def find_promising_cate(cls, cates: List[str]) -> str:
    skips: List[str] = ['category', 'categories', 'wikipedia', 'wikiproject', 'articles with', 'pages with']
    remains = []
    for cate in cates:
      _cate = cate.lstrip('Category:').lower()
      _skip = False
      for skip in skips:
        if skip in _cate:
          _skip = True
          break
      if _skip:
        continue
      remains.append(cate)
    return remains[0]

  def find_all_childs_recursively(self, parent: str) -> List[Set[str]]:
    layers = [{parent}]
    all_childs = {parent}
    to_extend = {parent}
    while len(to_extend) > 0:  # bfs
      _to_extend = set()
      for t in to_extend:
        if t in self.parent2child:
          n = set(self.parent2child[t]) - all_childs
          _to_extend.update(n)
          all_childs.update(n)
      to_extend = _to_extend
      layers.append(to_extend)
    return layers

  def find_closest_top_category(self,
                                categories: List[str],
                                top_categories: Set[str],
                                track_num_path: bool = False) -> Union[List[str], List[Tuple[str, int]]]:
    to_go: Dict[str, int] = {c: 1 for c in categories}  # track the number of path (initialized with 1)
    visited: Set[str] = set()
    while len(to_go) > 0:
      reached = set(to_go.keys()) & top_categories
      if len(reached) > 0:
        if track_num_path:
          return [(r, to_go[r]) for r in reached]
        else:
          return list(reached)
      _to_go: Dict[str, int] = defaultdict(lambda: 0)
      for tg in to_go:
        visited.add(tg)
        if tg not in self.child2parent:
          continue
        for _tg in self.child2parent[tg]:
          _to_go[_tg] = _to_go[_tg] + to_go[tg]
      _to_go = {k: v for k, v in _to_go.items() if k not in visited}
      to_go = _to_go
    raise Exception("can't reach top categories")

  def find_closest_top_category_by_request(self, cate: str, top_categories: Set[str], max_try: int = 50):  # all in title format
    tries = 0
    while cate:
      if cate in top_categories:
        return cate
      if tries >= max_try:
        raise Exception("reach max try")
      next = self.title2cate([cate], disable=True)
      tries += 1
      if len(next) <= 0:
        cate = None
      else:
        matches = list(set(next[cate]) & top_categories)
        if len(matches) > 0:  # find top
          cate = matches[0]
        else:
          next_cate = self.find_promising_cate(next[cate])
          cate = next_cate
    raise Exception("can't reach top categories")

  @classmethod
  def titles2other(cls, titles: List[str], field: str, **kwargs) -> Dict[str, str]:
    title2other: Dict[str, str] = {}
    for batch in cls.batcher(titles, **kwargs):
      url = cls.TITLE2OTHER_URL.format('|'.join(batch))
      r = requests.get(url)
      r = r.json()['query']['pages']
      for k, v in r.items():
        title = v['title']
        if field == 'pageid':
          v = k
        elif field == 'wikidata':
          if 'wikibase_item' not in v['pageprops']:
            continue
          v = v['pageprops']['wikibase_item']
        else:
          raise NotImplementedError
        title2other[title] = v
    return title2other

  @classmethod
  def pageids2other(cls, pageids: List[str], field: str, **kwargs):
    results: Set[str] = set()
    for batch in cls.batcher(pageids, **kwargs):
      if field in {'wikidata', 'title'}:
        url = cls.PAGEID2OTHER_URL.format('|'.join(batch))
      else:
        raise NotImplementedError
      r = requests.get(url)
      for k, v in r.json()['query']['pages'].items():
        if field == 'wikidata':
          if 'pageprops' not in v or 'wikibase_item' not in v['pageprops']:
            continue
          results.add(v['pageprops']['wikibase_item'])
        elif field == 'title':
          if 'title' not in v:
            continue
          results.add(v['title'])
    return results

  @classmethod
  def wikidataid2name(cls, wids: List[str], **kwargs):
    wid2name: Dict[str, str] = {}
    for batch in cls.batcher(wids, **kwargs):
      url = cls.WIKIDATA_ID2NAME.format('|'.join(batch))
      r = requests.get(url)
      for wid, v in r.json()['entities'].items():
        wid2name[wid] = v['labels']['en']['value']
    return wid2name

  @classmethod
  def wikidataid2pageid(cls, wids: List[str], **kwargs):
    wid2name = cls.wikidataid2name(wids, **kwargs)
    assert len(wids) == len(wid2name)
    titles = list(set(wid2name.values()))
    title2pageid = cls.titles2other(titles, field='pageid', **kwargs)
    assert len(titles) == len(title2pageid)
    return list(set(title2pageid.values()))

  @classmethod
  def pageid2cate(cls, pageids: List[str], use_revid: bool = False, **kwargs):
    pageid2cates: Dict[str, List[str]] = {}
    for batch in cls.batcher(pageids, **kwargs):
      if use_revid:
        url = cls.REVID2CATE_URL.format('|'.join(batch))
      else:
        url = cls.PAGEID2CATE_URL.format('|'.join(batch))
      r = requests.get(url).json()
      r = r['query']['pages']
      for k, v in r.items():
        if 'missing' in v:
          continue
        if 'categories' not in v:
          continue
        cates = [c['title'] for c in sorted(v['categories'], key=lambda x: x['sortkey'])]
        pageid2cates[k] = cates
    return pageid2cates

  @classmethod
  def title2cate(cls, titles: List[str], **kwargs):
    title2cates: Dict[str, List[str]] = {}
    for batch in cls.batcher(titles, **kwargs):
      url = cls.TITLE2CATE_URL.format('|'.join(batch))
      r = requests.get(url)
      r = r.json()['query']['pages']
      for k, v in r.items():
        if 'missing' in v:
          continue
        if 'categories' not in v:
          continue
        cates = [c['title'] for c in sorted(v['categories'], key=lambda x: x['sortkey'])]
        title2cates[v['title']] = cates
    return title2cates

  @classmethod
  def pageid2redirect(cls, pageids: List[str], **kwargs):
    redirects: List[str] = []
    for batch in cls.batcher(pageids, **kwargs):
      url = cls.PAGEID2REDIRECT_URL.format('|'.join(batch))
      r = requests.get(url).json()['query']['pages']
      r = {k: v for k, v in r.items() if 'missing' not in v}
      # TODO: the order might not be corresponding and some pageid might not have redirects
      redirects = list(r.keys())
    return redirects


def convert_title_to_pageid():
  topic2categories: Dict[str, List[str]] = {}
  for topic, categories in _topic2categories.items():
    categories = [f'{WikipediaCategory.PREFIX}{c}' for c in categories]
    title2pageid = WikipediaCategory.titles2other(categories, field='wikidata')
    topic2categories[topic] = [title2pageid[c] for c in categories]
  print(topic2categories)


def get_cateid2name(use_prefix: bool = False):
  cateid2name: Dict[str, str] = {}
  catename2id: Dict[str, str] = {}
  for topic in topic2categories:
    assert len(topic2categories[topic]) == len(_topic2categories[topic])
    for id, name in zip(topic2categories[topic], _topic2categories[topic]):
      if use_prefix:
        name = WikipediaCategory.PREFIX + name
      cateid2name[id] = name
      catename2id[name] = id
  return cateid2name, catename2id


if __name__ == '__main__':
  convert_title_to_pageid()
