#!/usr/bin/env bash

# these should be consistent with the setting in YAML config files
data_root=/mnt/root/TaBERT/data/train_data
model_root=/mnt/root/TaBERT/data/runs

# setting
num_gpu=$1
model_size=$2  # base, large

# input/output
input_dir=$3
output_dir=$4
load=$5

# hyperparameters
batchsize=$6
grad_acc=$7
epochs=$8

# extra
args="${@:9}"

function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

# config based on model_size
if [[ "$model_size" == "base" ]]; then
  base_model_name=${model_root}/bart_base  # mask token embedding bug
elif [[ "$model_size" == "large" ]]; then
  base_model_name=facebook/bart-large
else
  exit 1
fi

# config initial checkpoint
if [[ "$load" != "null" ]]; then
  load=${model_root}/${load}
fi

# config input
if [[ $input_dir == *":"* ]]; then
  echo "multitasking"
  IFS=':' read -ra input_dir <<< "$input_dir"
  for i in "${!input_dir[@]}"; do
    input_dir[i]=${data_root}/${input_dir[i]}
  done
  input_dir=$(join_by : ${input_dir[@]})
else
  input_dir=${data_root}/${input_dir}
fi

./run_vanilla.sh \
  ${num_gpu} \
  ${input_dir} \
  ${model_root}/${output_dir} \
  seq2seq ${batchsize} ${epochs} \
  ${load} \
  --gradient-accumulation-steps ${grad_acc} \
  --base-model-name ${base_model_name} \
  ${args}
