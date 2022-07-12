import os
os.environ['USE_TRANSFORMER'] = 'True'  # use new version

from typing import List, Union, Dict
from pathlib import Path
from argparse import ArgumentParser
import json
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import re
from omnitab.dataset_utils import BasicDataset
from omnitab.config import TableBertConfig
from utils.wtq_evaluator import to_value, to_value_list, check_denotation
AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
COND_OPS = ['=', '>', '<', 'OP']
only_alphanumeric = re.compile('[\W_]+')


OP2WORDS: Dict[str, List[str]] = {
    'select': ['which', 'what', 'who'],
    'filter': ['within', 'in', 'after', 'before',
               'with', 'only', 'other than', 'out of',
               'besides', 'both', 'above', 'below'],
    'aggregate': ['how many', 'how long', 'total',
                  'number', 'time period'],
    'superlative': ['most', '1st', 'first', 'last',
                    'highest', 'top', 'least'],
    'comparative': ['or', 'more', 'less', 'than',
                    'larger', 'smaller'],
    'group': ['each'],
}


def compute_f1(preds: List[str], golds: List[str]):
    sames = set(preds) & set(golds)
    p = len(sames) / (len(preds) or 1)
    r = len(sames) / (len(golds) or 1)
    f1 = 2 * p * r / ((p + r) or 1)
    return f1


def source_contains(source: str, targets: Union[str, List[str]]):
    if type(targets) is str:  targets = [targets]
    source = only_alphanumeric.sub('', source.lower())  # remove non-alphanumeric characters
    for target in targets:
        if only_alphanumeric.sub('', target.lower()) not in source:
            return False
    return True


def tsv_unescape(x):
  return x.replace(r'\n', '\n').replace(r'\p', '|').replace('\\\\', '\\')


def tsv_unescape_list(x):
  return [tsv_unescape(y) for y in x.split('|')]


def load_tagged_file(tagged_file: Path) -> List[List[str]]:
  results = []
  with open(tagged_file, 'r') as fin:
    header = fin.readline().rstrip('\n').split('\t')
    for line in fin:
      stuff = dict(zip(header, line.rstrip('\n').split('\t')))
      ex_id = stuff['id']
      original_strings = tsv_unescape_list(stuff['targetValue'])
      results.append(original_strings)
  return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction', type=str, required=True, nargs='+')
    parser.add_argument('--gold', type=str, default=None)
    parser.add_argument('--multi_ans_sep', type=str, default=', ')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--data', type=str, default='wtq', choices=['wikisql', 'wtq', 'wikisql_sql', 'turl', 'totto'])
    parser.add_argument('--model_type', type=str, default='facebook/bart-base')
    parser.add_argument('--clean', action='store_true',
                        help='clean the output file and print (which can be followed by other evaluation scripts)')
    parser.add_argument('--compare_em', action='store_true')
    parser.add_argument('--not_split_single', action='store_true')
    args = parser.parse_args()
    if '_sql' in args.data:
        from rouge import Rouge
        rouge = Rouge()

    ind2tagged = load_tagged_file('/root/exp/WikiTableQuestions/tagged/data/pristine-unseen-tables.tagged')
    ind2tagged: Dict[int, List[List[str]]] = dict(zip(range(len(ind2tagged)), ind2tagged))

    cls_token, sep_token, pad_token = TableBertConfig.get_special_tokens(args.model_type)
    def clean_output_from_model(text: str):
        for rmt in [pad_token, cls_token, sep_token]:
            text = text.replace(rmt, '')
        return text.strip()

    def get_ops(source: str):
        ops: List[str] = []
        for op, words in OP2WORDS.items():
            for word in words:
                if word in source:
                    ops.append(op)
                    break
        return ops

    ems: List[bool] = []
    ems_all: List[List[bool]] = []

    op2ems: Dict[str, List[bool]] = defaultdict(list)

    ans_in_inputs = []
    tapas_ems = []  # follow the tapas filtering conditions
    num_cells = []
    agg2ems = defaultdict(list)
    agg2cases = defaultdict(lambda: ([], []))

    cond2ems = defaultdict(list)
    cond2cases = defaultdict(lambda: ([], []))

    numcond2ems = defaultdict(list)
    numcell2ems = defaultdict(list)

    firstword2ems = defaultdict(list)
    firstword2cases = defaultdict(lambda: ([], []))

    numcoord2ems = defaultdict(list)
    answertype2ems = defaultdict(list)

    pred_file1 = args.prediction[0]
    pred_file_others = args.prediction[1:]

    if args.gold is None:  args.gold = pred_file1
    prev_example = None
    with open(pred_file1, 'r', encoding='utf-8') as pfin, \
      open(args.gold, 'r', encoding='utf-8') as gfin:
        if args.gold.endswith('.tsv'):
          _ = gfin.readline()  # skip the header
        pfin_others = [open(f, 'r', encoding='utf-8') for f in pred_file_others]
        for i, p in enumerate(pfin):
            # read predictions
            p = p.rstrip('\n').split('\t')
            p_others = [f.readline().rstrip('\n').split('\t') for f in pfin_others]
            if len(p) == 2:
                pred, gold = p[0].strip(), p[1].strip()  # use the gold from the first pred file
                pred_others: List[str] = [po[0].strip() for po in p_others]
            elif len(p) >= 3:
                p = p[:3]
                pred, gold, source = p[0].strip(), p[1].strip(), p[2].strip()  # use the gold from the first pred file
                pred = clean_output_from_model(pred)
                gold = clean_output_from_model(gold)
                source = source.replace(pad_token, '')

                pred_others: List[str] = [po[0].strip() for po in p_others]
                source_others: List[str] = [po[2].strip() for po in p_others]
                pred_others = [clean_output_from_model(po) for po in pred_others]
                source_others = [so.replace(pad_token, '') for so in source_others]

                num_cell = len(source.split(sep_token)) - 1
                num_cells.append(num_cell)
                first_word = source.split()[1].lower()  # skip cls
            if args.clean:
                print(pred)
                continue
            # evaluate
            em_others = []
            if args.data == 'wikisql':  # exact match
                em = pred.lower() == gold.lower()
            elif args.data == 'wtq':  # official WTQ evaluation
                if args.multi_ans_sep:
                    sep = args.multi_ans_sep
                    if len(ind2tagged[i]) == 1 and args.not_split_single:
                      golds = [gold]
                      preds: List[str] = [pred]
                    else:
                      golds = gold.split(sep)
                      preds: List[str] = pred.split(sep)
                    em = check_denotation(to_value_list(golds), to_value_list(preds))
                    print(em, golds, preds)
                    aii = source_contains(source, golds)
                    ops = get_ops(source.lower())
                    for op in ops:
                        op2ems[op].append(em)

                    pred_others: List[List[str]] = [po.split(sep) for po in pred_others]
                    em_others = [check_denotation(to_value_list(golds), to_value_list(po)) for po in pred_others]
                    aii_others = [source_contains(so, golds) for so in source_others]
                else:
                    em = to_value(gold).match(to_value(pred))
                    aii = source_contains(source, gold)

                    ops = get_ops(source.lower())
                    for op in ops:
                        op2ems[op].append(em)

                    em_others = [to_value(gold).match(to_value(po)) for po in pred_others]
                    aii_others = [source_contains(so, golds) for so in source_others]
                ans_in_inputs.append(aii)
            elif args.data == 'wikisql_sql':
                em = rouge.get_scores([pred.lower()], [gold.lower()], avg=True)['rouge-l']['f']
            elif args.data == 'turl':
                pred = re.sub('\s+', '', pred)
                gold = re.sub('\s+', '', gold)
                preds = [i for i in pred.split('<|>') if i != '']
                golds = [i for i in gold.split('<|>') if i != '']
                em = compute_f1(preds, golds)
            elif args.data == 'totto':
                em = gold == pred
            else:
                raise NotImplementedError
            ems.append(em)
            ems_all.append([em] + em_others)
            anstype = 'number' if BasicDataset.is_number(gold.strip()) else 'text'
            answertype2ems[anstype].append(em)

            if args.gold.endswith('.tsv'):
              wtqid = gfin.readline().strip().split()[0]
              fout.write(json.dumps({'id': wtqid, 'pred': pred, 'gold': gold, 'em': em}) + '\n')

            else:
              example = prev_example = json.loads(gfin.readline())
              #example = next(csv_reader)
              if 'sql' in example and type(example['sql']) is dict and 'agg' in example['sql']:  # wikisql example
                  agg = example['sql']['agg']
                  num_cond = len(example['sql']['conds'])
                  agg2ems[AGG_OPS[agg]].append(em)
                  numcond2ems[num_cond].append(em)
                  agg2cases[AGG_OPS[agg]][int(em)].append((source, pred, gold))
                  for cond in example['sql']['conds']:
                      cond = cond[1]
                      cond2ems[COND_OPS[cond]].append(em)
                      cond2cases[COND_OPS[cond]][int(em)].append((source, pred, gold))
              elif 'answer_coordinates' in example:  # tabert format
                  num_coord = len(example['answer_coordinates'])
                  numcoord2ems[num_coord].append(em)
                  if anstype == 'number' or num_coord == 1:
                      tapas_ems.append(em)

              if len(p) == 3:
                  numcell2ems[num_cell].append(em)
                  firstword2ems[first_word].append(em)
                  firstword2cases[first_word][int(em)].append((source, pred, gold))

    if args.clean:
        exit(0)

    if args.compare_em:
        for emsa in ems_all:
            print('\t'.join(map(lambda x: str(int(x)), emsa)))
        exit()

    print(np.mean(ems))  # the fine line of output is used for automatic analysis
    print(f'Exact match #1: [Overall] {np.mean(ems)} [TAPAS] {np.mean(tapas_ems)}, avg #cell {np.mean(num_cells)}')
    print(f'Exact match #-1: [Overall] {[np.mean(emsa) for emsa in zip(*ems_all)]}')
    print(f'Answer in input: {np.mean(ans_in_inputs)}')
    exit()

    for op, _ems in op2ems.items():
        print(f'{op}\t{np.mean(_ems)}\t{len(_ems)}')

    firstword2ems = {k: v for k, v in firstword2ems.items() if len(v) >= 10}
    numcell2ems = {k: v for k, v in numcell2ems.items() if len(v) >= 100}
    for k2em, name in [(agg2ems, 'agg'), (cond2ems, 'cond'), (numcond2ems, '#cond'),
                       (numcell2ems, '#cell'), (firstword2ems, 'first word'),
                       (numcoord2ems, '#coord'), (answertype2ems, 'ans type')]:
        print(f'\n===== {name} ======')
        for key, ems in sorted(k2em.items(), key=lambda x: x[0] if type(x[0]) in {float, int} else -len(x[1])):
            print(f'{key}\t\t{np.mean(ems)}\t{len(ems)}')

    if args.output:
        with open(args.output, 'w') as fout:
            for k2em, name in [(agg2cases, 'agg'), (cond2cases, 'cond'), (firstword2cases, 'first word')]:
                fout.write(f'\n===== {name} cases ======\n')
                for key, (badcases, goldcases) in sorted(k2em.items(), key=lambda x: x[0] if type(x[0]) in {float, int} else -len(x[1])):
                    random.shuffle(badcases)
                    for source, pred, gold in badcases[:5]:
                        fout.write(f'{pred}\t{gold}\t{source}\n')
