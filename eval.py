import argparse
import torch
from tqdm import tqdm

from lm_eval.models.huggingface import HFLM
import lm_eval
import torch.nn as nn
from pathlib import Path
import json
from datautils import get_loaders
from transformers import AutoTokenizer,LlamaConfig,LlamaForCausalLM, AutoModelForCausalLM
from safetensors import safe_open

from utils import load_json, save_json
from qat.replace_module import replace_with_learnable_binarylinear


def _parse_eval_task(task_str):
    optional_tasks = ['ppl', 'boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa', 'mmlu', 'storycloze_2016', 'storycloze', 'storycloze_2018']
    tasks = [s.strip() for s in task_str.split(',')]
    parsed_tasks = []
    for task in tasks:
        if task in optional_tasks:
            parsed_tasks.append(task)
        else:
            print(f'Wrong task name: {task} in your input: {task_str}. The optional tasks are: {", ".join(optional_tasks)}')
    return parsed_tasks


def load_open_src(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=False)
    return model, tokenizer


def load_ckpts(model_size, ckpt_dir, ckpt_type, exist_extra_para):
    assert model_size in ["130M", "1.3B", "7B"]
    assert ckpt_type in ['torch', 'hf_st', 'lightning']

    ckpt_dir = Path(ckpt_dir)
    with Path(f'FBI-LLM_configs/FBI-LLM_llama2_{model_size}.json').open('r') as r_f:
        config = json.load(r_f)
    llama_config = LlamaConfig(**config)
    model = LlamaForCausalLM(llama_config).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b', padding_side="right", use_fast=False)

    if exist_extra_para:
        model = replace_with_learnable_binarylinear(model, scaling_pattern = "column", keep_parts = ["lm_head"])

    weight_dict = {}
    if ckpt_type == 'torch':
        ckpt_plist = [p for p in ckpt_dir.iterdir() if p.suffix == '.bin']
        for p in ckpt_plist:
            _weight_dict = torch.load(p)
            for k,v in _weight_dict.items():
                if 'self_attn.rotary_emb.inv_freq' not in k:
                    weight_dict[k] = v

    elif ckpt_type == 'lightning':
        ckpt_plist = [p for p in (ckpt_dir/'fabric_ckpt').iterdir() if p.suffix == '.distcp']
        pass # TODO: add lightning ckpt load

    elif ckpt_type == 'hf_st':
        ckpt_plist = [p for p in ckpt_dir.iterdir() if p.suffix == '.safetensors']
        for p in ckpt_plist:
            with safe_open(p, framework="pt", device="cpu") as f:
                weight_dict.update({key: f.get_tensor(key) for key in f.keys()})
    

    model.load_state_dict(weight_dict)
    for param in model.parameters():
        param.data = param.data.to(torch.float16)
        
    return model, tokenizer


@torch.no_grad()
def evaluate_ckpt_task(model, tokenizer, tasks, num_fewshot, batch_size, max_length):
    task_manager = lm_eval.tasks.TaskManager()
    eval_lm = HFLM(model, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length)
    result = lm_eval.simple_evaluate(eval_lm, tasks = tasks, num_fewshot = num_fewshot, task_manager=task_manager)
    result = result["results"]
    print(result)
    return result

    
@torch.no_grad()
def evaluate_ckpt_ppl(model, tokenizer, ppl_datasets, max_length, limit = -1):
    results = {}
    for dataset in ppl_datasets:
        _, testloader = get_loaders(dataset, tokenizer)
        testenc = testloader.input_ids
        nsamples = testenc.numel() // max_length
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []

        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * max_length) : ((i + 1) * max_length)].to(model.device)
            outputs = model.model(batch)
            hidden_states = outputs[0]  # .to(model.lm_head.weight.device)
            logits = model.lm_head(hidden_states)  # .contiguous()
            shift_logits = logits[:, :-1, :]  # .contiguous()
            shift_labels = testenc[:, (i * max_length) : ((i + 1) * max_length)][:, 1:].to(model.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * max_length
            nlls.append(neg_log_likelihood)
            if i == limit:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * max_length))
        print(dataset, ppl.item())
        model.config.use_cache = use_cache
        results[dataset] = ppl.item()

    return results


def eval_ckpt(ckpt_dir, args):
    tasks = _parse_eval_task(args.task)
    print('tasks', tasks)
    
    eval_ppl = 'ppl' in tasks
    down_stream_tasks = [t for t in tasks if t != 'ppl']

    ckpt_dir = Path(ckpt_dir)
    model, tokenizer = load_ckpts(
        model_size = args.model_size, 
        ckpt_dir = ckpt_dir, 
        ckpt_type = args.ckpt_type, 
        exist_extra_para = args.exist_extra_para
        )
    # model = to_regular_linear(model)

    res = {}
    if eval_ppl:
        ppl_res = evaluate_ckpt_ppl(
            model = model, 
            tokenizer = tokenizer, 
            ppl_datasets = ['wikitext2', 'ptb', 'c4'],
            max_length = 2048
            )
        res.update(ppl_res)
    if len(down_stream_tasks) > 0:
        task_res = evaluate_ckpt_task(
            model = model, 
            tokenizer = tokenizer, 
            tasks = down_stream_tasks, 
            num_fewshot = 0, 
            batch_size = args.batch_size, 
            max_length = 2048
            )
        res.update(task_res)

    return res


def evaluate_qat(args):
    save_dir = Path('eval_result')
    save_dir.mkdir(exist_ok=True, parents=True)
    
    src_dir = Path(args.path)
    
    ckpt_ids = [i.strip() for i in args.ckpt_ids.split(',')]
    ckpt_ids = sorted(ckpt_ids, key=lambda x: int(x))
    save_p = save_dir / f"{src_dir.name}_{'-'.join(ckpt_ids)}.json"
    if save_p.exists():
        result = load_json(save_p)
    else:
        result = {}
    
    for cid in ckpt_ids:
        ckpt_name = f'ckpt-{cid}'
        print(ckpt_name)
        ckpt_dir = src_dir / ckpt_name

        res = eval_ckpt(ckpt_dir, args)

        if ckpt_name not in result:
            result[ckpt_name] = res
        else:
            result[ckpt_name].update(res)
    save_json(result, save_p)

def evaluate_open_src(args):
    save_dir = Path('eval_result')
    save_dir.mkdir(exist_ok=True, parents=True)
    tasks = _parse_eval_task(args.task)
    print('tasks', tasks)
    
    eval_ppl = 'ppl' in tasks
    down_stream_tasks = [t for t in tasks if t != 'ppl']
    
    save_p = save_dir / f"{'_'.join(args.path.split('/'))}.json"
    if save_p.exists():
        result = load_json(save_p)
    else:
        result = {}
    model, tokenizer = load_open_src(args.path)

    res = {}
    if eval_ppl:
        ppl_res = evaluate_ckpt_ppl(
            model = model, 
            tokenizer = tokenizer, 
            ppl_datasets = ['wikitext2', 'ptb', 'c4'],
            max_length = 2048
            )
        res.update(ppl_res)
    if len(down_stream_tasks) > 0:
        task_res = evaluate_ckpt_task(
            model = model, 
            tokenizer = tokenizer, 
            tasks = down_stream_tasks, 
            num_fewshot = 0, 
            batch_size = args.batch_size, 
            max_length = 2048
            )
        res.update(task_res)

    if '_'.join(args.path.split('/')) not in result:
        result['_'.join(args.path.split('/'))] = res
    else:
        result['_'.join(args.path.split('/'))].update(res)
    save_json(result, save_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--path",
        type=str,
        help="Saved model path",
    )
    parser.add_argument(
        "--eval_open_src",
        action="store_true",
        help="If evaluating open source LLMs outside of FBI-LLMs, please specify this argument."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ppl,boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
        help="evaluate tasks",
    )
    parser.add_argument(
        "--ckpt_ids",
        type=str,
        help='The checkpoints to evaluate'
    )
    parser.add_argument(
        "--model_size",
        type=str,
        help="model size",
    )
    parser.add_argument(
        "--ckpt_type",
        type=str,
        default='torch',
        help="The saving type of checkpoints"
    )
    parser.add_argument(
        "--exist_extra_para",
        action="store_true",
        help="Are there any additional parameters for the model to be evaluated. If evaluating FBI-LLM, please specify this argument. If evaluating other open source LLMs, do not specify this argument"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="batch_size for evaluation"
    )
    
    args = parser.parse_args()
    
    if args.eval_open_src:
        evaluate_open_src(args)
    else:
        evaluate_qat(args)

