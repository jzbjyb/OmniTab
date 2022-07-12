#!/usr/bin/env bash
set -e

model_root=/mnt/root/TaBERT/data/runs

task=wtqqa
pred=$1

host=$(hostname)
if [[ "$host" == "GPU02" ]]; then
  prefix=$HOME
else
  prefix=""
fi

if [[ "$task" == "wtqqa" ]]; then
  gold=${prefix}/mnt/root/TaBERT/data/wikitablequestions/test/preprocessed_with_ans.jsonl
elif [[ "$task" == "wtqqa_dev" ]]; then
  gold=data/wikitablequestions/WikiTableQuestions/data/random-split-1-dev.tsv  # case study purpose
elif [[ "$task" == "wikisqlqa" ]]; then
  gold=${prefix}/mnt/root/TaBERT/data/wikisql/tapex/test.src
elif [[ "$task" == "totto_official" ]]; then
  gold=$(pwd)/data/totto_data/totto_dev_data.jsonl
elif [[ "$task" == "totto" ]]; then
  gold=$(pwd)/data/totto_data/dev/preprocessed_mention_cell.jsonl.raw
else
  exit
fi

eval_totto() {
  pred=$1
  temp_file=$(mktemp)
  python -m utils.eval --prediction ${pred} --clean 2> /dev/null 1> ${temp_file}
  pushd ~/exp/language
  bash language/totto/totto_eval.sh --prediction_path ${temp_file} --target_path ${gold}
  popd
}

if [[ -d $pred ]]; then
  for i in ${pred}/*.tsv; do
    if [[ "$task" == totto* ]]; then
      result=$(eval_totto ${i} | grep 'BLEU+' | tr -s " " | cut -f3 -d" ")
    else
      result=$(python -W ignore -m utils.eval --prediction ${i} --gold ${gold} --multi_ans_sep ", " 2> /dev/null | head -n 1)
    fi
    echo $(basename $i) ${result}
  done
elif [[ -f $pred ]]; then
  if [[ "$task" == totto* ]]; then
    # clean the prediction file
    eval_totto ${pred}
  else
    python -m utils.eval \
      --prediction ${pred} \
      --gold ${gold} \
      --multi_ans_sep ", "
  fi
else
  exit
fi
