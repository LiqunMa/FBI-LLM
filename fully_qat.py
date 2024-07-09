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
from transformers import AutoTokenizer,LlamaConfig

from model_utils.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from qat.replace_module import (
    replace_with_learnable_binarylinear, 
    check_para_state
)
import json
from pathlib import Path

from main_utils import (
    load_jsonl_examples,
    get_cosine_lr_decay_fn,
    get_grad_norm,
    save_checkpoint,
    get_last_ckpt_idx)


PROJECT_NAME = 'FBI-LLM'
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


def collate_fn(examples, device):
    token_ids = torch.tensor(
        [example['token_ids'] for example in examples], device=device)
    return {'input_ids': token_ids[:, :-1], 'labels': token_ids[:, 1:]}


def train_chunk(fabric,
                tokenizer,
                model,
                teacher,
                use_kd,
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
        desc=f'Training chunk {chunk_name}({chunk_idx}) (global_micro_batch_size='
             f'{per_device_batch_size * fabric.world_size}, '
             f'accumulate_grad_batches={accumulate_grad_batches})')
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
            if use_kd == 1:
                student_logits = model(input_ids).logits
                with torch.no_grad():
                    teacher_logits = teacher(input_ids).logits
                teacher_prob = F.softmax(teacher_logits, dim=2).clone().detach()
                loss = torch.nn.functional.cross_entropy(
                    student_logits.reshape((-1, student_logits.size(-1))), teacher_prob.reshape((-1, teacher_prob.size(-1))))
            elif use_kd == 2:
                student_logits = model(input_ids).logits
                with torch.no_grad():
                    teacher_logits = teacher(input_ids).logits
                teacher_prob = F.softmax(teacher_logits, dim=2).clone().detach()
                kd_loss = torch.nn.functional.cross_entropy(
                    student_logits.reshape((-1, student_logits.size(-1))), teacher_prob.reshape((-1, teacher_prob.size(-1))))
                ar_loss = torch.nn.functional.cross_entropy(
                    student_logits.reshape((-1, student_logits.size(-1))), labels.reshape(-1))
                loss = 0.5*ar_loss + 0.5*kd_loss
            else:
                logits = model(input_ids).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape((-1, logits.size(-1))), labels.reshape(-1))
                
            fabric.backward(loss / accumulate_grad_batches)

        if not is_accumulating:
            grad_norm = get_grad_norm(model=model)
            fabric.clip_gradients(model, optimizer, max_norm=GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        log = {
            'loss': loss.item(),
            'learning_rate': lr,
            'step': step,
            'speed(#tok/s/gpu)': int(input_ids.numel() / (time.time() - t0)),
        }
        if use_kd == 2:
            log['ar_loss'] = ar_loss.item()
            log['kd_loss'] = kd_loss.item()
        
        if not is_accumulating:
            log['grad_norm'] = grad_norm

        example_batch_idxes.set_postfix(log)
        if run_wandb and fabric.global_rank == 0:
            wandb.log(log)

    save_checkpoint(
        fabric=fabric,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        save_dir=f'{WORKDIR}/ckpt-{chunk_name}')


def main(tag='FBI-LLM-7B',
         model_size='7B',
         n_nodes=8,
         n_devices_per_node=4,
         per_device_batch_size=50,
         accumulate_grad_batches=40,
         train_data_dir = 'Amber_data_path',
         skip_chunk = False,
         use_kd=1,
         run_wandb=False
         ):

    WORKDIR = f'fully_qat_record/{tag}_{use_kd}_{model_size}_amber'
    RUN_NAME = f'{WORKDIR}_{DATE}'
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

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

    

    
    with Path(f'FBI-LLM_llama2_{model_size}.json').open('r') as r_f:
        _config = json.load(r_f)
    config = LlamaConfig(**_config)
    model = LlamaForCausalLM(config=config)
            
    model = replace_with_learnable_binarylinear(model, 'column', ['lm_head'])

    if fabric.global_rank == 0:
        print(config)
        check_para_state(model)

    optimizer = torch.optim.AdamW( 
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2),
        foreach=False)
 
    last_ckpt_name = get_last_ckpt_idx(workdir=WORKDIR)
    del_list = []
    if skip_chunk:
        del_list = SKIP_CHUNK_ID
    cur_skip_num = sum([1 for i in del_list if i < last_ckpt_name])
    last_ckpt_idx = last_ckpt_name - cur_skip_num
    fabric.seed_everything(RANDOM_SEED + last_ckpt_idx + 1)

    model, optimizer = fabric.setup(model, optimizer)

    if last_ckpt_name != -1:
        fabric.load(
            path=f'{WORKDIR}/ckpt-{last_ckpt_name}/fabric_ckpt',
            state={'model': model, 'optimizer': optimizer})
        
    if use_kd > 0:
        teacher = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.config.use_cache = False
        teacher = fabric.setup(teacher)
    else:
        teacher = None
        
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
            teacher=teacher,
            use_kd=use_kd,
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