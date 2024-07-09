# --------------------------------------------------------------------------------
# Code References and Acknowledgements:
# This file contains functions and code segments adapted or inspired by the following 
# projects and authors:
#
# 1. Function "get_tokenizer()" adapted from tatsu-lab/stanford_alpaca (https://github.com/tatsu-lab/stanford_alpaca). 
#    Source code is available at https://github.com/tatsu-lab/stanford_alpaca/blob/761dc5bfbdeeffa89b8bff5d038781a4055f796a/train.py#L198. 
#    Licensed under Apache-2.0 license (http://www.apache.org/licenses/).
#
# 2. Function "get_wikitext2()", "get_ptb()", and  "get_c4()" are derived from IST-DASLab/sparsegpt (https://github.com/IST-DASLab/sparsegpt).
#    Source code is available at https://github.com/IST-DASLab/sparsegpt/blob/c3bbf613a1822229767f4d8870b933049b8bef15/datautils.py. 
#    Licensed under Apache-2.0 license (http://www.apache.org/licenses/).
#
# We acknowledge and are grateful to these developers for their contributions to open 
# source. Please note that all aforementioned projects, source code and their licenses 
# are owned by their respective authors and are not affiliated with this project.
# --------------------------------------------------------------------------------

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import transformers
from pathlib import Path
import json
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from torch.utils.data import Dataset
import random
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from typing import Dict
from utils import load_json
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

TXT_KEY = {
    "togethercomputer/RedPajama-Data-1T-Sample": 'text'
}

class ARDataset(Dataset):
    """Dataset for autoregressive training."""
    def __init__(self, data_p):
        super(ARDataset, self).__init__()
        print(f'{data_p.name} exists, loading ...')
        self.dataset = get_jsonl(data_p)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return {
            'input_ids': self.dataset[i]['input_ids']
        }


class FTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_p):
        super(FTDataset, self).__init__()

        data_dict = load_json(data_p)

        self.input_ids = torch.LongTensor(data_dict["input_ids"])
        self.labels = torch.LongTensor(data_dict["labels"])
        self.attention_mask = torch.BoolTensor(data_dict["attention_mask"])
        print(self.input_ids.shape)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


def get_tokenizer(model_name = 'huggyllama/llama-7b'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", padding_side="right", use_fast=False)
    special_tokens_dict = dict()

    if tokenizer.pad_token is None:
        print('no tokenizer.pad_token')
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        print('no tokenizer.eos_token')
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        print('no tokenizer.bos_token')
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        print('no tokenizer.unk_token')
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer, special_tokens_dict


def get_jsonl(data_path):
    data_path = Path(data_path)
    data = []
    with data_path.open('r') as r_f:
        for line in r_f:
            data.append(json.loads(line.strip()))
    return data


def _tokenize(ids_range, txt_key, txt_data, max_length, tokenizer):
    tk_ids = []
    for i in tqdm(ids_range):
        txt = txt_data[i][txt_key]
        res = tokenizer(txt, truncation=True, padding='max_length', max_length=max_length)
        tk_ids.append(json.dumps(
            {
                'input_ids': res['input_ids'],
                'attention_mask': res['attention_mask']
            }
        ))
    return tk_ids


def preprocess_redpajama(dataset, dataset_name, tokenizer, max_length, pool_size, save_path):
    txt_key = TXT_KEY[dataset_name]
    
    step = len(dataset) // (pool_size - 1)
    id_list = list(range(len(dataset)))
    paras = [id_list[i*step: (i+1)*step] for i in range(pool_size)]
    func = partial(_tokenize, txt_key = txt_key, txt_data = dataset, max_length = max_length, tokenizer = tokenizer)
    
    with Pool(pool_size) as pool:
        data_tkids = []
        results = pool.map(func, paras)
        for res in results:
            data_tkids += res
        with save_path.open('w') as w_f:
            w_f.write('\n'.join(data_tkids))



def preprocess_sharegpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    print(conv.roles)
    data_specific_extra_roles = {
        'user': conv.roles[0],
        'system': conv.roles[1],
        'chatgpt': conv.roles[1],
        'bing': conv.roles[1],
        'bard': conv.roles[1],
        }
    roles.update(data_specific_extra_roles)

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if len(source) > 0 and roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv.roles[j % 2]:
                break
            # assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                # rank0_print(
                #     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                #     f" #turn = {len(turns) - 1}. (ignored)"
                # )
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def get_autoregressive_dataset(dataset_name, tokenizer, max_length, percent, part_id):
    data_p =  Path(f"finetuning_data/{dataset_name.split('/')[-1]}_{int(percent*100)}_{part_id}.jsonl")
    if dataset_name == "togethercomputer/RedPajama-Data-1T-Sample":
        try:
            tkids_dataset = ARDataset(data_p)
        except:
            print(f'{data_p.name} does not exist, processing  ...')

            sample_size = int(850000 * percent)
            if part_id * sample_size >= 850000:
                raise ValueError("\"part_id\" exceeds the dataset limit")
    
            split = f"train[{part_id * sample_size}:{(part_id + 1) * sample_size}]"
            sample_dataset = load_dataset(dataset_name, split=split)

            preprocess_redpajama(sample_dataset, dataset_name, tokenizer, max_length, pool_size=32, save_path = data_p)

            tkids_dataset = ARDataset(data_p)

    elif dataset_name == "Abirate/english_quotes":
        # TO DO
        pass

    return tkids_dataset

def get_finetuning_dataset(dataset_name, tokenizer):
    if dataset_name == 'ShareGPT_V3_unfiltered_cleaned_split_no_imsorry':
        data_p =  Path(f"finetuning_data/tokenized_{dataset_name}.json")
        
        try:
            tkids_dataset = FTDataset(data_p)
        except:
            print(f'{data_p.name} does not exist, processing  ...')

            raw_data = load_json(f'finetuning_data/{dataset_name}.json')
            sources = [example["conversations"] for example in raw_data]
            
            res = preprocess_sharegpt(sources, tokenizer)
            res = {k:v.to(torch.int32).tolist() for k, v in res.items()}
            with data_p.open('w') as w_f:
                json.dump(res, w_f, ensure_ascii=False)

            tkids_dataset = FTDataset(data_p)

    return tkids_dataset


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
    

def get_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    if  name == 'wikitext2':
        loaders = get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if name == 'ptb':
        loaders = get_ptb(nsamples, seed, seqlen, tokenizer)
    if name == 'c4':
        loaders = get_c4(nsamples, seed, seqlen, tokenizer)

    return loaders


if __name__ == '__main__':
    tokenizer, _ = get_tokenizer(model_name = 'huggyllama/llama-7b')

    # max_length = tokenizer.model_max_length
    # for part_id in range(10):
    #     get_autoregressive_dataset("togethercomputer/RedPajama-Data-1T-Sample", tokenizer, max_length, 0.1, part_id)

    # raw_data = load_json('finetuning_data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json')
    # sources = [example["conversations"] for example in raw_data]
    
    # res = preprocess_sharegpt(sources, tokenizer)
    # res = {k:v.to(torch.int32).tolist() for k, v in res.items()}
    # with Path('finetuning_data/tokenized_ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json').open('w') as w_f:
    #     json.dump(res, w_f, ensure_ascii=False)

    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')