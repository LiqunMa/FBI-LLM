#!/bin/bash
#SBATCH --job-name=eval_FBI-LLM
#SBATCH --partition=your_partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

python eval.py --path fully_qat_record/FBI-LLM_7B_1_7B_amber --exist_extra_para --ckpt_ids 0 --batch_size 16 --model_size 7B
# --path: the directory where you save checkpoints
# --exist_extra_para: if you evalute FBI-LLM, specify this parameter.
# --ckpt_ids: the id of the checkpoint to evaluate
# --model_size: the model size you evaluate (130M, 1.3B, 7B)