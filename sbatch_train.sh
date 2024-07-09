#!/bin/bash
#SBATCH --job-name=FBI-LLM
#SBATCH --partition=your_partition
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

# srun python fully_qat.py --tag FBI-LLM_130M --model_size 130M --train_data_dir Amber_data_path --use_kd 1 --n_nodes 1 --n_devices_per_node 4 --per_device_batch_size 32 --accumulate_grad_batches 8  --run_wandb

# srun python fully_qat.py --tag FBI-LLM_1.3B --model_size 1.3B --train_data_dir Amber_data_path --use_kd 1 --skip_chunk --n_nodes 4 --n_devices_per_node 4 --per_device_batch_size 18 --accumulate_grad_batches 4 --run_wandb

srun python fully_qat.py --tag FBI-LLM_7B --model_size 7B --train_data_dir Amber_data_path --use_kd 1 --n_nodes 8 --n_devices_per_node 4 --per_device_batch_size 6 --accumulate_grad_batches 10 --run_wandb