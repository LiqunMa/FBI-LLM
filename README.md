# FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation

*[FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation](https://arxiv.org/abs/2407.07093)*

[Liqun Ma](https://scholar.google.com/citations?user=zVXXXGIAAAAJ&hl=zh-CN), [Mingjie Sun](https://eric-mingjie.github.io/), [Zhiqiang Shen](http://zhiqiangshen.com/)

Mohamed bin Zayed University of Artificial Intelligence.

Carnegie Mellon University.

## Abstract
This work presents a **F**ully **BI**narized **L**arge **L**anguage **M**odel (FBI-LLM), demonstrating for the first time how to train a large-scale binary language model (not the ternary LLM like BitNet b1.58 from scratch to match the performance of its full-precision counterparts (e.g., FP16 or BF16) in transformer-based LLMs. It achieves this by employing an autoregressive distillation (AD) loss with maintaining equivalent model dimensions (130M, 1.3B, 7B) and training data volume as regular LLM pretraining, while delivering competitive results in terms of perplexity and task-specific effectiveness. Intriguingly, by analyzing the training trajectory, we find that the pretrained weight is not necessary for training binarized LLMs from scratch. This research encourages a new computational framework and may facilitate the future design of specialized hardware tailored for fully 1-bit LLMs. We make all models, code, and training dataset fully accessible and transparent to support further research.

![image](https://github.com/LiqunMa/FBI-LLM/blob/main/figures/structure_and_training_procedure.png)

## News
- [2024/7/1] The pretrained FBI-LLMs in the paper are released on HuggingFace ([https://huggingface.co/LiqunMa/](https://huggingface.co/LiqunMa/)).
- [2024/7/1] FBI-LLM code is open.

## Requirements
```
torch==2.1.1
transformers==4.40.0
tokenizers==0.19.1
datasets==2.19.0
lightning==2.1.3
flash-attn==2.5.0
fastchat==0.1.0
lm-eval==0.4.2
pytz
wandb
fire
```

## Train FBI-LLM
- Please download the [AmberDatasets](https://huggingface.co/datasets/LLM360/AmberDatasets) firstly.
- You can use `sbatch` to submit slurm job to train FBI-LLM:
  ```
  sbatch sbatch_train.sh
  ```
  
  Please modify the corresponding parameters in `sbatch_train.sh` according to the configuration of your cluster.

## Pretrained Models
Pretrained moodels are released on HuggingFace ([https://huggingface.co/LiqunMa/](https://huggingface.co/LiqunMa/)): `FBI-LLM-130M`, `FBI-LLM-1.3B`, `FBI-LLM-7B`.

The structural parameters of these models are as follows:

|      | FBI-LLM 130M | FBI-LLM 1.3B | FBI-LLM 7B |
| ----------- | ----------- |----------- |----------- |
| # layers      | 12          |  24      | 32         |
| hidden size   | 768        | 2048 | 4096 |
| intermediate size | 2048   | 5632 | 11008 |
| # attention heads | 12     |32    |32     |

## Evaluate FBI-LLM
We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate FBI-LLMs. 

- Please install `lm-evaluation-harness` (`pip install lm-eval==0.4.2`)
- NOTE: Although the parameters of our model have been binarized, using 1-bit representation for the corresponding parameters and performing inference requires further low-level deployment design. In our current code, inference is still performed using high precision calculations. As a result, the inference speed cannot be improved compared to the full-precision model with the current code implementation.
- Since the llama structure needs to be replaced, please use `load_ckpts()` from `eval.py` to load the model. If you want to directly evaluate our open-sourced model, please use the following command:
 ```
 sbatch sbatch_eval.sh
 ```

## Results
![image](https://github.com/LiqunMa/FBI-LLM/blob/main/figures/main_result.jpg)

## Citation
If you find our work useful and helpful to your research, please consider citing this paper:
```
@article{ma2024fbi,
  title={Fbi-llm: Scaling up fully binarized llms from scratch via autoregressive distillation},
  author={Ma, Liqun and Sun, Mingjie and Shen, Zhiqiang},
  journal={arXiv preprint arXiv:2407.07093},
  year={2024}
}
```
