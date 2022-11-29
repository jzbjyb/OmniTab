#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%x.%j.out
#SBATCH -e slurm/%x.%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=256GB
#SBATCH --job-name=omnitab

# Notes
# 1. The effective batch size is 8 (#gpu) * 36 (batch size) * 2 (accumulation) = 576.
# 2. The pretraining dataset has ~2m examples, which is ~17k update steps for 5 epochs.
# 3. The pretraining dataset takes 1~2 hours to process using 10 workers.
#    To accelerate, you could process it using all cores, cache it (activated by default), then run multi-GPU training.

# activate env here (change based on your cluster)
module purge
module load anaconda3
. /usr/share/modules/init/sh
eval "$(conda shell.bash hook)"
conda activate omnitab

# output directory
data=$1  # omnitab_download/pretrain_data
model=$2  # microsoft/tapex-large
output=$3  # output/omnitab-large

python -m torch.distributed.launch --nproc_per_node=8 run.py \
  --do_train \
  --do_eval \
  --pretraindata_dir ${data} \
  --dataset_name wikitablequestions \
  --remove_unused_columns false \
  --model_name_or_path ${model} \
  --tokenizer_name neulab/omnitab-large \
  --overwrite_output_dir \
  --output_dir ${output} \
  --max_source_length 1024 \
  --max_target_length 1024 \
  --val_max_target_length 128 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 36 \
  --per_device_eval_batch_size 6 \
  --num_train_epochs 5.0 \
  --warmup_ratio 0.1 \
  --learning_rate 2e-5 \
  --fp16 \
  --preprocessing_num_workers 80 \
  --logging_steps 10 \
  --eval_steps 1000 \
  --save_steps 5000 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --generation_max_length 128
