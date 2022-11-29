#!/usr/bin/env bash
#SBATCH --time=8:00:00
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%x.%j.out
#SBATCH -e slurm/%x.%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=256GB
#SBATCH --job-name=wtq

# Notes
# 1. The effective batch size is 8 (#gpu) * 6 (batch size) * 2 (accumulation) = 96.
# 2. The WTQ dataset has ~11k examples, which is ~6k update steps for 50 epochs.

# activate env here (change based on your cluster)
module purge
module load anaconda3
. /usr/share/modules/init/sh
eval "$(conda shell.bash hook)"
conda activate omnitab

# arguments
model=$1  # neulab/omnitab-large
output=$2  # output/omnitab-large-finetuned-wtq

# few-shot or full setting
if [ $# == 3 ]; then
    fewshot="--train_ids_file omnitab_download/wtq_fewshot/${3}.txt"
else
    fewshot=""
fi

python -m torch.distributed.launch --nproc_per_node=8 run.py \
  --do_train \
  --do_eval \
  --dataset_name wikitablequestions \
  --model_name_or_path ${model} \
  --output_dir ${output} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --overwrite_output_dir \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 6 \
  --num_train_epochs 50.0 \
  --warmup_ratio 0.1 \
  --learning_rate 2e-5 \
  --fp16 \
  --logging_steps 10 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --generation_max_length 128 \
  ${fewshot}
