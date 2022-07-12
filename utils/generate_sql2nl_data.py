from argparse import ArgumentParser
from pathlib import Path
import os
import json
from omnitab.spider import Spider
from omnitab.squall import Squall
from omnitab.wikitablequestions import WikiTQ


def main():
  parser = ArgumentParser()
  parser.add_argument('--task', type=str, required=True, choices=['spider', 'squall', 'tapex', 'combine_nl'])
  parser.add_argument('--inp', type=Path, required=True, nargs='+')
  parser.add_argument('--out', type=Path, required=True)
  parser.add_argument('--split', type=str, default='dev')
  args = parser.parse_args()

  if args.task == 'spider':
    spider_dir = args.inp[0]
    spider = Spider(spider_dir)
    os.makedirs(args.out / args.split, exist_ok=True)
    spider.gen_sql2nl_data(args.split, args.out / args.split / 'sqlnl.json')
    #spider.convert_to_tabert_format(args.split, args.out / args.split / 'db_tabert.json')
    #spider.sample_negative(args.split, args.out / args.split / 'samples.tsv', neg_num=9)

  elif args.task == 'squall':
    squall_file, wtq_dir = args.inp
    output_path = args.out
    fews = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    squall = Squall(squall_file, wikitq=None)
    #print(sorted(squall.keyword2count.items(), key=lambda x: -x[1]))
    for few in fews:
      print(f'load the sql for few-shot {few}')
      squall.get_subset(wtq_dir / f'train.src.{few}', output_path / f'train.src.{few}')
    squall.get_subset(wtq_dir / f'train.src', output_path / f'train.src')
    squall.get_subset(wtq_dir / f'valid.src', output_path / f'valid.src')
    squall.get_subset(wtq_dir / f'test.src', output_path / f'test.src')

  elif args.task == 'tapex':
    prep_file = args.inp[0]
    out = args.out
    with open(prep_file, 'r') as fin, open(out, 'w') as fout:
      for l in fin:
        l = json.loads(l)
        sql = l['context_before'][0]
        nl = 'dummy'
        td = {
          'uuid': l['uuid'],
          'metadata': {
            'sql': sql,
            'nl': nl,
          },
          'table': l['table'],
          'context_before': [nl],
          'context_after': []
        }
        fout.write(json.dumps(td) + '\n')


if __name__ == '__main__':
  main()
