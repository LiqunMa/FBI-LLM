#!/bin/bash
#SBATCH --job-name=FBI-LLM_analysis
#SBATCH --partition=your_partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

srun python analysis.py --tag tinyllama_pretrain --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --n_nodes 1 --n_devices_per_node 4 --per_device_batch_size 6 --accumulate_grad_batches 32  --train_data_dir Amber_data_path --run_wandb