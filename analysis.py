from datetime import datetime
from pytz import timezone
import time
from functools import partial
import wandb
import fire
import tqdm
import torch
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from numpy import linalg as LA

from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from lightning.fabric.strategies import FSDPStrategy

from model_utils.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from qat.replace_module import replace_with_learnable_binarylinear
import json
from pathlib import Path
import numpy as np

from main_utils import (
    load_jsonl_examples,
    get_cosine_lr_decay_fn,
    get_grad_norm,
    save_checkpoint,
    get_last_ckpt_idx)


PROJECT_NAME = 'FBI-LLM_analysis'
TIMEZONE = timezone('EST')
DATE = str(datetime.now(tz=TIMEZONE)).split()[0]
LEARNING_RATE = 3e-4
LR_SCHEDULE_TYPE = 'cosine'
END_LEARNING_RATE = 3e-5
WARMUP_GRAD_STEPS = 2000
GRAD_NORM_CLIP = 1.
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
ACCELERATOR = 'cuda'
PRECISION = 'bf16-mixed'
RANDOM_SEED = 11111
TRAIN_EXAMPLES_PER_CHUNK = 1706976
N_CHUNKS = 360
SKIP_CHUNK_ID = [7, 12, 20, 24, 26]

def cal_flip_rate(A, B):
    bA = np.sign(A).astype(np.int8)
    bB = np.sign(B).astype(np.int8)
    bA[bA==0] = -1
    bB[bB==0] = -1
    
    flip_num = (bA != bB).astype(np.float32).sum()
    flip_rate = flip_num/bA.size

    return int(flip_num), float(flip_rate)


def collate_fn(examples, device):
    token_ids = torch.tensor(
        [example['token_ids'] for example in examples], device=device)
    return {'input_ids': token_ids[:, :-1], 'labels': token_ids[:, 1:]}


def train_chunk(fabric,
                tokenizer,
                model,
                pre_params, 
                param_info,
                param_info_p,
                teacher,
                optimizer,
                lr_schedule_fn,
                examples,
                per_device_batch_size,
                accumulate_grad_batches,
                chunk_idx,
                chunk_name,
                run_wandb,
                WORKDIR):
    step = chunk_idx * (len(examples) // per_device_batch_size)
    example_batch_idxes = tqdm.trange(
        0, len(examples), per_device_batch_size,
        desc=f'chunk {chunk_name}({chunk_idx}) (global_micro_batch_size='
             f'{per_device_batch_size * fabric.world_size}, '
             f'acc_grad_batches={accumulate_grad_batches})')
    for i in example_batch_idxes:
        t0 = time.time()

        lr = lr_schedule_fn(step)
        step += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        is_accumulating = (step % accumulate_grad_batches != 0)

        batch = collate_fn(
            examples=examples[i:i+per_device_batch_size], device=fabric.device)
        input_ids, labels = batch['input_ids'], batch['labels']
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            student_logits = model(input_ids).logits
            with torch.no_grad():
                teacher_logits = teacher(input_ids).logits
            teacher_prob = F.softmax(teacher_logits, dim=2).clone().detach()
            loss = torch.nn.functional.cross_entropy(
                student_logits.reshape((-1, student_logits.size(-1))), teacher_prob.reshape((-1, teacher_prob.size(-1))))
                
            fabric.backward(loss / accumulate_grad_batches)

        if not is_accumulating:
            grad_norm = get_grad_norm(model=model)
            fabric.clip_gradients(model, optimizer, max_norm=GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            
            fabric.barrier()
            policy = FullStateDictConfig(offload_to_cpu=(fabric.world_size > 1), rank0_only=True)
            
            with FSDP.state_dict_type(
                model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=policy):
            
                state_dict = model._forward_module.state_dict()
                state_dict = {k: v.to('cpu').numpy() for k, v in state_dict.items()}
                # fabric.broadcast(state_dict)
            
            if fabric.global_rank == 0:
                cur_flip_num, all_linear_size = 0, 0
                for k, v in state_dict.items():
                    if k in pre_params:
                        flip_num, flip_rate = cal_flip_rate(v, pre_params[k])
                        pre_params[k] = v
                        param_info[k]['flip_num'].append(flip_num)
                        param_info[k]['flip_rate'].append(flip_rate)
                        cur_flip_num += flip_num
                        all_linear_size += v.size
                ave_flip_rate = cur_flip_num / all_linear_size

                param_info['ave_flip_rate'].append(ave_flip_rate)
                with open(param_info_p, 'w') as w_f:
                    json.dump(param_info, w_f, indent=4)
        
        if run_wandb and not is_accumulating and fabric.global_rank == 0:
            log = {
            'loss': loss.item(),
            'learning_rate': lr,
            'step': step,
            'acc_step': step//accumulate_grad_batches,
            'grad_norm': grad_norm,
            'ave_flip_rate': ave_flip_rate,
            'speed(#tok/s/gpu)': int(input_ids.numel() / (time.time() - t0)),
            }
            example_batch_idxes.set_postfix(log)
            wandb.log(log)


    save_checkpoint(
        fabric=fabric,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        save_dir=f'{WORKDIR}/ckpt-{chunk_name}')


def main(tag,
         model_name,
         n_nodes,
         n_devices_per_node,
         per_device_batch_size,
         accumulate_grad_batches,
         train_data_dir = 'Amber_data_path',
         skip_chunk = False,
         from_scratch = False,
         run_wandb = False
         ):
    
    WORKDIR = f'fully_qat_record/{tag}_{"-".join(model_name.split("/"))}'
    RUN_NAME = f'{WORKDIR}_{DATE}'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    fabric = L.Fabric(
        accelerator=ACCELERATOR,
        num_nodes=n_nodes,
        devices=n_devices_per_node,
        precision=PRECISION,
        strategy=FSDPStrategy(
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer}),
            activation_checkpointing_policy={LlamaDecoderLayer},
            cpu_offload=True,
            limit_all_gathers=True))
    fabric.launch()

    if fabric.global_rank == 0:
        Path(WORKDIR).mkdir(exist_ok=True, parents=True)
        if run_wandb:
            wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    last_ckpt_name = get_last_ckpt_idx(workdir=WORKDIR)
    del_list = []
    if skip_chunk:
        del_list = SKIP_CHUNK_ID
    cur_skip_num = sum([1 for i in del_list if i < last_ckpt_name])
    last_ckpt_idx = last_ckpt_name - cur_skip_num
    fabric.seed_everything(RANDOM_SEED + last_ckpt_idx + 1)

    

    if from_scratch:
        config = AutoConfig.from_pretrained(model_name)
        config.max_position_embeddings = 2048
        model =AutoModelForCausalLM.from_config(config=config)
        param_info_p = Path(f"analyse_result/{'-'.join(model_name.split('/'))}_scatch.json")
        print(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
        param_info_p = Path(f"analyse_result/{'-'.join(model_name.split('/'))}_pretrain.json")

    model = replace_with_learnable_binarylinear(model, 'column', ['lm_head'])    
    
    pre_params, param_info = {}, {}
    zero_num = 0
    if not param_info_p.exists():
        for name, param in model.named_parameters():
            if 'weight' in name and ('self_attn' in name or 'mlp' in name):
                zero_num += (param == 0).sum().item()
                param_info[name] = {
                    'shape': param.size(),
                    'flip_num': [],
                    'flip_rate': []
                }
        param_info['ave_flip_rate'] = []
        with param_info_p.open('w') as w_f:
            json.dump(param_info, w_f, indent=4)
    else:
        with param_info_p.open('r') as r_f:
            param_info = json.load(r_f) 


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2),
        foreach=False)

    model, optimizer = fabric.setup(model, optimizer)
    if last_ckpt_name != -1:
        fabric.load(
            path=f'{WORKDIR}/ckpt-{last_ckpt_name}/fabric_ckpt',
            state={'model': model, 'optimizer': optimizer})
        
    policy = FullStateDictConfig(
        offload_to_cpu=(fabric.world_size > 1), rank0_only=True)
    with FSDP.state_dict_type(
            model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=policy):
        state_dict = model._forward_module.state_dict()
        state_dict = {k: v.to('cpu').numpy() for k, v in state_dict.items()}
        for k, v in state_dict.items():
            if 'weight' in k and ('self_attn' in k or 'mlp' in k):
                pre_params[k] = v
    
    teacher = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.config.use_cache = False
    teacher = fabric.setup(teacher)
    
    torch.cuda.empty_cache()

    global_micro_batch_size = per_device_batch_size * fabric.world_size
    total_steps = TRAIN_EXAMPLES_PER_CHUNK // global_micro_batch_size * N_CHUNKS
    lr_schedule_fn = get_cosine_lr_decay_fn(
        total_steps=total_steps,
        warmup_steps=WARMUP_GRAD_STEPS * accumulate_grad_batches,
        learning_rate=LEARNING_RATE,
        end_learning_rate=END_LEARNING_RATE)
    
    chunk_list = [i for i in range(last_ckpt_name + 1, N_CHUNKS) if i not in del_list]

    for chunk_idx, chunk_name in enumerate(chunk_list, start=last_ckpt_idx+1):
        examples = load_jsonl_examples(
            filename=f'{train_data_dir}/train_{chunk_name:03}.jsonl',
            n_examples=TRAIN_EXAMPLES_PER_CHUNK,
            shuffle=True,
            global_micro_batch_size=global_micro_batch_size,
            global_rank=fabric.global_rank,
            world_size=fabric.world_size)

        train_chunk(
            fabric=fabric,
            tokenizer=tokenizer,
            model=model,
            pre_params = pre_params, 
            param_info = param_info,
            param_info_p =param_info_p,
            teacher=teacher,
            optimizer=optimizer,
            lr_schedule_fn=lr_schedule_fn,
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            chunk_idx=chunk_idx,
            chunk_name=chunk_name,
            run_wandb=run_wandb,
            WORKDIR=WORKDIR)


if __name__ == '__main__':
    fire.Fire(main)