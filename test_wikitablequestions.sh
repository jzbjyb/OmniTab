#!/usr/bin/env bash
#SBATCH --time=1:00:00
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%x.%j.out
#SBATCH -e slurm/%x.%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=256GB
#SBATCH --job-name=test

# activate env here (change based on your cluster)
module purge
module load anaconda3
. /usr/share/modules/init/sh
eval "$(conda shell.bash hook)"
conda activate omnitab

split=$1  # test or validation
model=$2  # neulab/omnitab-large-finetuned-wtq
output=$3  # output dir

python -m torch.distributed.launch --nproc_per_node=8 run.py \
  --do_predict \
  --do_predict_on ${split} \
  --model_name_or_path ${model} \
  --output_dir ${output} \
  --per_device_eval_batch_size 6 \
  --predict_with_generate \
  --num_beams 5
