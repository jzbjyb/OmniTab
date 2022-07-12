from typing import Dict, List, Tuple, Set
from collections import defaultdict
import json
import csv
import re
import random
from nltk.corpus import stopwords
from difflib import SequenceMatcher
import timeout_decorator


class BasicDataset(object):
    STOPWORDS = None
    @staticmethod
    def get_stopwords():
        if BasicDataset.STOPWORDS is None:
            BasicDataset.STOPWORDS = set(stopwords.words('english'))
        return BasicDataset.STOPWORDS

    @staticmethod
    def add_punc(text, punc: str = '.'):
        text = text.strip()
        if len(text) > 0 and re.search('[?.!"]$', text) is None:
            return text + punc
        return text

    @staticmethod
    def is_number(s):
        if s is None:
            return False
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def float_eq(a: str, b: str):
        try:
            a = float(a)
            b = float(b)
            return a == b
        except ValueError:
            return False

    @staticmethod
    def locate_in_table(values: List[str], table: List[List[str]]):
        locations: List[Tuple[int, int]] = []
        values = set([re.sub('[\W_]+', ' ', v.strip().lower()) for v in values])  # only keep alphanumerics
        for row_idx, row in enumerate(table):
            for col_idx, cell in enumerate(row):
                if re.sub('[\W_]+', ' ', cell.strip().lower()) in values:
                    locations.append((row_idx, col_idx))
        return locations

    @staticmethod
    def add_answer(prep_file: str, answer_file: str, out_file: str, string_match: bool = False):
        numans2count: Dict[int, int] = defaultdict(lambda: 0)
        matched = False
        found_count = total_count = 0
        with open(prep_file, 'r') as pfin, open(answer_file, 'r') as afin, open(out_file, 'w') as fout:
            csv_reader = csv.reader(afin, delimiter='\t')
            _ = next(csv_reader)  # skip tsv head
            for i, l in enumerate(pfin):
                l = json.loads(l)
                al = next(csv_reader)
                coord = eval(al[5])
                anss = eval(al[6])
                assert type(coord) is list and type(anss) is list, 'format error'
                if string_match:
                    l['answer_coordinates'] = BasicDataset.locate_in_table(anss, l['table']['data'])
                    found_count += int(len(l['answer_coordinates']) > 0)
                    total_count += 1
                else:
                    l['answer_coordinates'] = [eval(c) for c in coord]
                l['answers'] = anss
                numans2count[len(anss)] += 1
                fout.write(json.dumps(l) + '\n')
            try:
                next(csv_reader)
            except StopIteration as e:
                matched = True
            if not matched:
                raise Exception(f'{prep_file} and {answer_file} have different #rows')
        print('#ans -> count: {}'.format(sorted(numans2count.items())))
        print(f'found {found_count} out of {total_count}')

    @staticmethod
    def is_a_word(sentence: str, substr: str):
        if len(substr) <= 0:
            return False
        start = sentence.find(substr)
        if start < 0:  # not found
            return False
        end = start + len(substr)
        if start != 0 and sentence[start - 1].isalnum():
            return False
        if end != len(sentence) and sentence[end].isalnum():
            return False
        return True

    @staticmethod
    def longest_substring(str1, str2):
        sm = SequenceMatcher(None, str1, str2)
        match = sm.find_longest_match(0, len(str1), 0, len(str2))
        if match.size != 0:
            return str1[match.a:match.a + match.size], match.a, match.b
        return '', -1, -1

    #@timeout_decorator.timeout(1, use_signals=False)
    @staticmethod
    def get_mention_locations(context: str, table: List[List[str]], highlighed_cells: Set[Tuple[int, int]] = None):
        locations: Set[Tuple[int, int]] = set()  # (inclusive, exclusive)
        location2cells: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        context = context.lower()
        annotated = True
        if highlighed_cells is None:  # assume all cells are highlighted
            annotated = False
            highlighed_cells = set((row_idx, col_idx) for row_idx, row in enumerate(table) for col_idx, _ in enumerate(row))
        for r, c in highlighed_cells:
            v = table[r][c].lower()
            if len(v) <= 0:
                continue
            # common = STree.STree([v, context]).lcs()  # slow
            common = BasicDataset.longest_substring(context, v)[0]
            common = common.strip()
            if annotated:
                find = len(common) == len(v) or \
                       (len(v) >= 5 and BasicDataset.is_a_word(v, common)) or \
                       len(common) > 10
            else:  # higher standard
                find = (common not in BasicDataset.get_stopwords()) and (re.search('[a-zA-Z0-9]', common) is not None)
                find &= (len(common) == len(v) and BasicDataset.is_a_word(context, common)) or \
                        (len(v) >= 5 and len(common) / len(v) >= 0.7 and BasicDataset.is_a_word(v, common) and BasicDataset.is_a_word(context, common)) or \
                        len(common) >= 20
            if len(common) > 0 and find:
                start = context.find(common)
                loc = (start, start + len(common))
                locations.add(loc)
                location2cells[loc].append((r, c))
        return sorted(list(locations)), location2cells
