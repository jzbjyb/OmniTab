#!/usr/bin/env bash

# activate env if needed
if [[ "$PATH" == *"tabert"* ]]; then
  echo "tabert env activated"
else
  echo "tabert env not activated"
  conda_base=$(conda info --base)
  source ${conda_base}/etc/profile.d/conda.sh
  conda activate tabert
fi

# wandb
export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a
