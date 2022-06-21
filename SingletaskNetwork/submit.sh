#! /bin/bash
# MultiTask Network Submission Script for CBICA Cluster - Vishnu Bashyam

cd /cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/SingletaskNetwork/

#Load Virtual ENV
# source /cbica/home/bashyamv/ENV/dl_env/dlenv/bin/activate
source /cbica/home/bashyamv/ENV/torch_lightning_env/torch_lightning_env/bin/activate
module load cudnn/8.2.1
module load cuda/11.2

# Get GPU IDs
# CUDA_VISIBLE_DEVICES=$(get_CUDA_VISIBLE_DEVICES) || exit
# export CUDA_VISIBLE_DEVICES
# echo $OMP_NUM_THREADS
# echo $MKL_NUM_THREADS

# Stop GPU Logging on Keyboard Interrupt
trap ctrl_c INT

function ctrl_c() {
        pkill timeout
}

# GPU Logging
timeout 2700 nvidia-smi --query-gpu=timestamp,name,pci.bus_id,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > gpu-usage_single.csv &

# Main Script
# CUDA_VISIBLE_DEVICES=0,1 python3 main.py
python3 main.py  \
  --experiment $1 \
  --experiment_type $6 \
  --experiment_tag $7 \
  --prediction_endpoint $5 \
  --data_csv  $2 \
  --data_dir /cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/BrainAligned \
  --dataloader_num_processes 4 \
  --model_size $3 \
  --pretrained_weights \
  --batch_size $4 \
  --max_epochs 10 \
  --optimizer Adam \
  --accumulate_grad_batches 2 \
  --mixed_precision


# Stop GPU Logging
pkill timeout