#!/usr/bin/env bash

model_root=/mnt/root/TaBERT/data/runs

# --- arguments ---
pipeline=skip
# all: (default) run all
# last_epoch: run finetune and only evaluate the last epoch
# only_last_epoch: only evaluate the last epoch
# single: only perform evaluation for a single checkpoint
# multi: only perform evaluation for all checkpoints
# skip: skip evaluation
task=$1  # "wtqqa"
test_task=""  # "wtqqa_topic"
model_size=$2  # "base", "large"
scale=$3  # "full" (8 gpus), "half" (4 gpus)

# hyperparameters
if [[ "$model_size" == "base" ]]; then
  if [[ "$scale" == "full" ]]; then
    ngpu=8
    batch_size=12
    grad_acc=1
  elif [[ "$scale" == "half" ]]; then
    ngpu=4
    batch_size=12
    grad_acc=2
  else
    exit 1
  fi
  base_model_name=facebook/bart-base
elif [[ "$model_size" == "large" ]]; then
  if [[ "$scale" == "full" ]]; then
    ngpu=8
    batch_size=6
    grad_acc=2
  elif [[ "$scale" == "half" ]]; then
    ngpu=4
    batch_size=6
    grad_acc=4
  else
    exit 1
  fi
  base_model_name=facebook/bart-large
else
  exit 1
fi

epoch=$4
model_ckpt=$5  # use for initialization if it follows the pattern "*.bin"; otherwise use it as output directory
model_ckpt=${model_root}/${model_ckpt}
args="${@:6}"  # additional args

# predefined collections of tasks
if [[ "$task" == "default" ]]; then
  task="wtqqa_strict_128:wtqqa_strict"
  task+=":wtqqa_strict_16:wtqqa_strict_32:wtqqa_strict_64:wtqqa_strict_256:wtqqa_strict_512:wtqqa_strict_1024"
elif [[ "$task" == "cate_efp_all" ]]; then
  task="wtqqa_strict_efp_Q1457982_128:wtqqa_strict_efp_Q4047087_128:wtqqa_strict_efp_Q54070_128"
  task+=":wtqqa_strict_efp_Q1457673_128:wtqqa_strict_efp_Q8255_128:wtqqa_strict_efp_Q1458390_128"
  task+=":wtqqa_strict_efp_Q6337045_128:wtqqa_strict_efp_Q7214908_128:wtqqa_strict_efp_Q2944929_128"
  task+=":wtqqa_strict_efp_Q1457595_128"
  task+=":wtqqa_strict_efp_Q1457982:wtqqa_strict_efp_Q4047087:wtqqa_strict_efp_Q54070:wtqqa_strict_efp_Q1457673"
  task+=":wtqqa_strict_efp_Q8255:wtqqa_strict_efp_Q1458390:wtqqa_strict_efp_Q6337045:wtqqa_strict_efp_Q7214908"
  task+=":wtqqa_strict_efp_Q2944929:wtqqa_strict_efp_Q1457595"
elif [[ "$task" == "cate_efp_16" ]]; then
  task="wtqqa_strict_efp_Q1457982_16:wtqqa_strict_efp_Q4047087_16:wtqqa_strict_efp_Q54070_16"
  task+=":wtqqa_strict_efp_Q1457673_16:wtqqa_strict_efp_Q8255_16:wtqqa_strict_efp_Q1458390_16"
  task+=":wtqqa_strict_efp_Q6337045_16:wtqqa_strict_efp_Q7214908_16:wtqqa_strict_efp_Q2944929_16"
  task+=":wtqqa_strict_efp_Q1457595_16:wtqqa_strict_efp_Q7386634_16:wtqqa_strict_efp_Q1457756_16"
  task+=":wtqqa_strict_efp_Q4103183_16:wtqqa_strict_efp_Q5613113_16:wtqqa_strict_efp_Q5850187_16"
  task+=":wtqqa_strict_efp_Q8413436_16:wtqqa_strict_efp_Q4049293_16:wtqqa_strict_efp_Q4103249_16"
  task+=":wtqqa_strict_efp_Q9715089_16:wtqqa_strict_efp_Q6528585_16:wtqqa_strict_efp_Q6353120_16"
  task+=":wtqqa_strict_efp_Q6576895_16:wtqqa_strict_efp_Q5645580_16"
elif [[ "$task" == "topic_all" ]]; then
  task="wtqqa_strict_sports_128:wtqqa_strict_politics_128:wtqqa_strict_people_128"
  task+=":wtqqa_strict_culture_128:wtqqa_strict_misc_128"
  task+=":wtqqa_strict_sports:wtqqa_strict_politics:wtqqa_strict_people"
  task+=":wtqqa_strict_culture:wtqqa_strict_misc"
elif [[ "$task" == "topic_fp_all_fewshot" ]]; then
  task="wtqqa_strict_fp_sports_128:wtqqa_strict_fp_politics_128:wtqqa_strict_fp_people_128"
  task+=":wtqqa_strict_fp_culture_128:wtqqa_strict_fp_misc_128"
elif [[ "$task" == "topic_fp_all" ]]; then
  task="wtqqa_strict_fp_sports_128:wtqqa_strict_fp_politics_128:wtqqa_strict_fp_people_128"
  task+=":wtqqa_strict_fp_culture_128:wtqqa_strict_fp_misc_128"
  task+=":wtqqa_strict_fp_sports:wtqqa_strict_fp_politics:wtqqa_strict_fp_people"
  task+=":wtqqa_strict_fp_culture:wtqqa_strict_fp_misc"
elif [[ "$task" == "wikisql_fewshot" ]]; then
  task="wikisqlqa_strict_16:wikisqlqa_strict_128:wikisqlqa_strict_1024"
elif [[ "$task" == "wikisql_full" ]]; then
  task="wikisqlqa_strict"
fi

# --- finetune & predict ---
IFS=':' read -ra tasks <<< "$task"
IFS=':' read -ra epochs <<< "$epoch"

for task in "${tasks[@]}"; do
  if [[ "$task" == "wtqqa" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_1024_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_32" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_num32
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_64" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_num64
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_256" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_num256
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_512" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_num512
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_culture" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_culture
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_culture_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_culture_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_culture_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_culture_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_culture_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_culture_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_culture_exclude" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_culture_exclude
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_culture_exclude_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_culture_exclude_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_culture_exclude_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_culture_exclude_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_culture_exclude_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_culture_exclude_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_misc" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_misc
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_misc_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_misc_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_misc_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_misc_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_misc_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_misc_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_misc_exclude" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_misc_exclude
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_misc_exclude_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_misc_exclude_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_misc_exclude_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_misc_exclude_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_misc_exclude_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_misc_exclude_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_people" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_people
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_people_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_people_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_people_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_people_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_people_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_people_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_people_exclude" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_people_exclude
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_people_exclude_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_people_exclude_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_people_exclude_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_people_exclude_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_people_exclude_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_people_exclude_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_politics" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_politics
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_politics_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_politics_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_politics_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_politics_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_politics_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_politics_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_politics_exclude" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_politics_exclude
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_politics_exclude_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_politics_exclude_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_politics_exclude_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_politics_exclude_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_politics_exclude_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_politics_exclude_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_sports" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_sports
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_sports_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_sports_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_sports_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_sports_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_sports_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_sports_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_sports_exclude" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_sports_exclude
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_sports_exclude_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_sports_exclude_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_sports_exclude_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_sports_exclude_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_sports_exclude_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_sports_exclude_num1024
    mode=generate-test

  elif [[ "$task" == "wtqqa_strict_fp_culture" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_culture
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_culture_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_culture_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_culture_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_culture_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_culture_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_culture_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_misc" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_misc
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_misc_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_misc_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_misc_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_misc_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_misc_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_misc_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_people" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_people
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_people_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_people_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_people_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_people_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_people_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_people_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_politics" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_politics
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_politics_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_politics_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_politics_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_politics_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_politics_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_politics_num1024
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_sports" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_sports
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_sports_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_sports_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_sports_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_sports_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_fp_sports_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_sports_num1024
    mode=generate-test

  elif [[ "$task" == "wtqqa_strict_efp_Q1457982" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1457982
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1457982_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1457982_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1457982_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1457982_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q4047087" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q4047087
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q4047087_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q4047087_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q4047087_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q4047087_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q54070" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q54070
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q54070_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q54070_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q54070_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q54070_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1457673" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1457673
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1457673_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1457673_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1457673_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1457673_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q8255" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q8255
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q8255_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q8255_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q8255_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q8255_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1458390" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1458390
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1458390_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1458390_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1458390_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1458390_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q6337045" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q6337045
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q6337045_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q6337045_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q6337045_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q6337045_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q7214908" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q7214908
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q7214908_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q7214908_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q7214908_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q7214908_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q2944929" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q2944929
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q2944929_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q2944929_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q2944929_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q2944929_num128
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1457595" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1457595
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1457595_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1457595_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1457595_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1457595_num128
    mode=generate-test

  elif [[ "$task" == "wtqqa_strict_efp_Q7386634_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q7386634_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q1457756_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q1457756_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q4103183_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q4103183_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q5613113_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q5613113_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q5850187_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q5850187_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q8413436_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q8413436_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q4049293_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q4049293_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q4103249_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q4103249_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q9715089_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q9715089_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q6528585_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q6528585_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q6353120_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q6353120_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q6576895_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q6576895_num16
    mode=generate-test
  elif [[ "$task" == "wtqqa_strict_efp_Q5645580_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_cate_exclusive_followpaper_Q5645580_num16
    mode=generate-test

  elif [[ "$task" == "wtqqa_deprecate" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wtq_qa_firstansrow_add30
    mode=generate-test
  elif [[ "$task" == "totto_official" ]]; then
    data=/mnt/root/TaBERT/data/train_data/totto_data2text_official_bart
    mode=generate-dev
  elif [[ "$task" == "totto_official_wholetable" ]]; then
    data=/mnt/root/TaBERT/data/train_data/totto_data2text_official_wholetable_bart
    mode=generate-dev
  elif [[ "$task" == "totto" ]]; then
    data=/mnt/root/TaBERT/data/train_data/totto_data2text_bart
    mode=generate-dev
  elif [[ "$task" == "totto_1_10" ]]; then
    data=/mnt/root/TaBERT/data/train_data/totto_data2text_bart_1_10
    mode=generate-dev
  elif [[ "$task" == "totto_1_20" ]]; then
    data=/mnt/root/TaBERT/data/train_data/totto_data2text_bart_1_20
    mode=generate-dev
  elif [[ "$task" == "totto_1_100" ]]; then
    data=/mnt/root/TaBERT/data/train_data/totto_data2text_bart_1_100
    mode=generate-dev

  elif [[ "$task" == "wikisqlqa_strict" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wikisql_qa_tapex_strict_1024
    mode=generate-test
  elif [[ "$task" == "wikisqlqa_strict_16" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wikisql_qa_tapex_strict_1024_num16
    mode=generate-test
  elif [[ "$task" == "wikisqlqa_strict_128" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wikisql_qa_tapex_strict_1024_num128
    mode=generate-test
  elif [[ "$task" == "wikisqlqa_strict_1024" ]]; then
    data=/mnt/root/TaBERT/data/train_data/wikisql_qa_tapex_strict_1024_num1024
    mode=generate-test

  else
    echo 'unsupported tasks' ${task}
    exit 1
  fi

  if [[ "$pipeline" == "single" ]]; then
    output="$(dirname "${model_ckpt}")"
    prediction_file=${model_ckpt}.${task}.tsv
    ./run_vanilla.sh \
      1 ${data} ${output} seq2seq ${batch_size} 1 ${model_ckpt} \
      --base-model-name ${base_model_name} \
      --only_test --mode ${mode} --output_file ${prediction_file}
    exit
  fi

  for epoch in "${epochs[@]}"; do
    if [[ "$pipeline" == "multi" || "$pipeline" == "only_last_epoch" ]]; then
      if [[ ${model_ckpt} == *.bin ]]; then   # a real checkpoint
        output="$(dirname "${model_ckpt}")"_${task}_ep${epoch}
      else
        output=${model_ckpt}_${task}_ep${epoch}
      fi
    else
      # finetune
      if [[ ${model_ckpt} == *.bin ]]; then   # a real checkpoint
        output="$(dirname "${model_ckpt}")"_${task}_ep${epoch}
        ./run_vanilla.sh ${ngpu} ${data} ${output} seq2seq ${batch_size} ${epoch} ${model_ckpt} \
          --gradient-accumulation-steps ${grad_acc} --base-model-name ${base_model_name} \
          --mode ${mode} ${args}
      else
        output=${model_ckpt}_${task}_ep${epoch}
        ./run_vanilla.sh ${ngpu} ${data} ${output} seq2seq ${batch_size} ${epoch} null \
          --gradient-accumulation-steps ${grad_acc} --base-model-name ${base_model_name} \
          --mode ${mode} ${args}
      fi
    fi

    if [[ "$pipeline" != "skip" ]]; then
      # evaluate every checkpoint
      remain=${ngpu}
      isfirst=true
      if [[ "$test_task" == "" ]]; then
        max_epoch=$(expr $epoch - 2)  # skip the last epoch because it's already evaluated
      else
        max_epoch=$(expr $epoch - 1)  # evalute all epochs
      fi
      for (( i=$max_epoch; i>=0; --i )); do
        iwz=$(printf "%02d" $i)  # add a preceding zero
        inter_ckpt=${output}/pytorch_model_epoch${iwz}.bin
        prediction_file=ep${iwz}.tsv

        if [[ "$test_task" == "wtqqa_topic" ]]; then  # use another test data directory
          data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_all_topic_test
          mode=generate-test_all
        elif [[ "$test_task" == "wtqqa_topic_followpaper" ]]; then
          data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_followpaper_all_topic_test
          mode=generate-test_all
        elif [[ "$test_task" == "wtqqa_cate_exclusive_followpaper_10" ]]; then
          data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_exclusive_followpaper_all_cate_test_10
          mode=generate-test_all
        elif [[ "$test_task" == "wtqqa_cate_exclusive_followpaper_23" ]]; then
          data=/mnt/root/TaBERT/data/train_data/wtq_qa_tapex_strict_1024_exclusive_followpaper_all_cate_test_23
          mode=generate-test_all
        fi

        CUDA_VISIBLE_DEVICES="$((remain - 1))" ./run_vanilla.sh \
          1 ${data} ${output} seq2seq ${batch_size} 1 ${inter_ckpt} \
          --base-model-name ${base_model_name} \
          --only_test --mode ${mode} --output_file ${prediction_file} &
        if [[ "$isfirst" == "true" ]]; then
          # run the first exclusively to download necessary files
          wait
          isfirst=false
          if [[ "$pipeline" == "last_epoch" || "$pipeline" == "only_last_epoch" ]]; then
            break
          fi
        fi
        remain="$((remain - 1))"
        if (( ${remain} == 0 )); then
          wait
          remain=${ngpu}
        fi
      done
      wait
    fi
  done
done
