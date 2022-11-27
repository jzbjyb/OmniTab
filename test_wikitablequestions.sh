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

model=neulab/omnitab-large-finetuned-wtq

python run_wikitablequestions.py \
  --do_predict \
  --model_name_or_path ${model} \
  --output_dir test \
  --per_device_eval_batch_size 6 \
  --predict_with_generate \
  --num_beams 5
