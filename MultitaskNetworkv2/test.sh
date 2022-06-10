#! /bin/bash

cd /cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/MultitaskNetworkv2/

#Load Virtual ENV
# source /cbica/home/bashyamv/ENV/dl_env/dlenv/bin/activate
source /cbica/home/bashyamv/ENV/torch_lightning_env/torch_lightning_env/bin/activate
module load cudnn/8.2.1
module load cuda/11.2

# CUDA_VISIBLE_DEVICES=0,1 python3 main.py
python3 main.py \
  --experiment Test_Multitask \
  --data_csv /cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/SingletaskNetwork/FID_Data_SingleTask_Prep_Only2Class.csv \
  --model_size 10 \
  --batch_size 8
