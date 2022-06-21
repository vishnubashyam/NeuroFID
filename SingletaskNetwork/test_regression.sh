#! /bin/bash

cd /cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/SingletaskNetwork/

#Load Virtual ENV
# source /cbica/home/bashyamv/ENV/dl_env/dlenv/bin/activate
source /cbica/home/bashyamv/ENV/torch_lightning_env/torch_lightning_env/bin/activate
module load cudnn/8.2.1
module load cuda/11.2

# CUDA_VISIBLE_DEVICES=0,1 python3 main.py
python3 main.py \
  --experiment test_run_regression \
  --experiment_type Regression \
  --experiment_tag Test \
  --data_csv /cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/Lists/Training_Lists/Digit_Span_Forward_Interpolated_REG.csv \
  --data_dir /cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/BrainAligned \
  --dataloader_num_processes 4 \
  --model_size 18 \
  --pretrained_weights \
  --batch_size 8 \
  --max_epochs 10 \
  --optimizer Adam \
  --accumulate_grad_batches 2 \
  --mixed_precision

