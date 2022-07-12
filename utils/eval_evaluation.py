import os
os.environ['USE_TRANSFORMER'] = 'True'  # use new version

from typing import List, Tuple
from argparse import ArgumentParser
import json
import numpy as np
from collections import defaultdict
from omnitab.dataset import Example
from omnitab.config import TableBertConfig, MODEL2SEP, MODEL2TOKENIZER


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prediction', type=str, required=True)
    parser.add_argument('--prep_file', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='facebook/bart-base')
    parser.add_argument('--ana', action='store_true', help='use preprocessed files for analysis')
    args = parser.parse_args()
    mt = TableBertConfig.check_model_type(args.model_type)

    tokenizer = MODEL2TOKENIZER[mt].from_pretrained(args.model_type)
    sep_token = MODEL2SEP[mt]

    is_sames = []
    cellstatus2name = {0: 'name', 1: 'type', 2: 'value'}
    type2issames = defaultdict(list)
    with open(args.prediction, 'r') as fin, open(args.prep_file, 'r') as pfin:
        for line_id, l in enumerate(fin):
            example = json.loads(pfin.readline())
            example = Example.from_dict(example, tokenizer, suffix=None)
            l = json.loads(l)
            tgt = l['tgt']
            pred = l['pred']
            gold = l['gold']
            assert len(tgt) == len(pred) == len(gold), 'inconsistent number of tokens'

            is_table = False
            cell_status = 0  # 0 -> name, 1 -> type, 2 -> value (assume template is "column | type | value")
            context: List[str] = []
            cell: Tuple[List, List, List] = ([], [], [])
            cell_idx = 0
            for idx, (tgt_t, pred_t, gold_t) in enumerate(zip(tgt, pred, gold)):
                if tgt_t == sep_token and not is_table:  # context table split point
                    context = tokenizer.convert_tokens_to_string(context)
                    is_table = True
                    continue
                raw_t = gold_t or tgt_t

                if is_table:
                    if tgt_t == sep_token:
                        cell = ([], [], [])
                        cell_status = 0
                        cell_idx += 1
                        continue
                    elif raw_t == '|':  # might lead to errors if some tokens are accidentally "|"
                        cell_status += 1
                        cell_status = min(cell_status, 2)
                        continue
                    else:
                        cell[cell_status].append(raw_t)
                else:
                    context.append(raw_t)

                if gold_t is None:
                    continue

                is_same = pred_t == gold_t

                if is_table:
                    type2issames[cellstatus2name[cell_status]].append(is_same)
                    if args.ana and ((cell_status in {0, 1} and example.header[cell_idx].used) or
                                     (cell_status == 2 and example.header[cell_idx].value_used)):
                        type2issames[cellstatus2name[cell_status] + '-used'].append(is_same)
                else:
                    type2issames['context'].append(is_same)
                is_sames.append(is_same)
            #input()

    print(f'average accuracy {np.mean(is_sames)} with total count {len(is_sames)}')
    for key, iss in type2issames.items():
        print(f'{key}\t{np.mean(iss)}\t{len(iss)}')
