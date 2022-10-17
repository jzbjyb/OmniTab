#!/usr/bin/env bash

source env_initialize.sh

MAX_NUM_GPU_PER_NODE=8
num_gpu=$1
input_dir=$2  # data/train_data/vanilla_tabert
output_dir=$3  # data/runs/vanilla_tabert
mkdir -p ${output_dir}
loss=$4
batchsize=$5
epochs=$6
name="$(basename -- $output_dir)"
echo '==========' ${name} '=========='
load=$7
args="${@:8}"

# (1) single node w/o deepspeed
# export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
# (2) single node w/ deepspeed
# export NGPU=8; deepspeed train.py
# (3) single node w/ deepspeed and limited GPUs
# export NGPU=1; deepspeed --num_gpus 1 train.py
# (4) multi node w/ deepspeed
# export NGPU=8; deepspeed train.py

# use different lanucher for single/multi-node
if (( ${num_gpu} == 1 )); then
  echo 'single-GPU'
  prefix=""
elif (( ${num_gpu} <= ${MAX_NUM_GPU_PER_NODE} )); then
  echo 'single-node'
  export NGPU=${num_gpu}
  prefix="-m torch.distributed.launch --nproc_per_node=${num_gpu}"
else
  echo 'multi-node'
  prefix=""
fi

if [[ "$load" != "null" ]]; then  # add quote for json-based config
  load='"'${load}'"'
fi

python ${prefix} train.py \
  --task vanilla \
  --data-dir ${input_dir} \
  --output-dir ${output_dir} \
  --table-bert-extra-config '{"objective_function": "'${loss}'", "load_model_from": '${load}'}' \
  --train-batch-size ${batchsize} \
  --learning-rate 2e-5 \
  --max-epoch ${epochs} \
  --adam-eps 1e-08 \
  --weight-decay 0.0 \
  --fp16 \
  --clip-norm 1.0 \
  --empty-cache-freq 128 \
  --name ${name} \
  ${args}
