import os
import torch, warnings
import math
from torch.utils.data import DataLoader, Dataset
from textbox import CLM_MODELS
from typing import List
from torch.nn.utils.rnn import pad_sequence
from textbox.data.misc import load_data, _pad_sequence
import random

MAX_TOKENIZE_NUM = 1000000


class TranslateCollate_BI:

    def __init__(self, config, tokenizer, set):
        self.config = config
        self.tokenizer = tokenizer
        self.set = set
        self.is_casual_model = bool(config["model_name"] in CLM_MODELS)
        self.paired_text = bool(
            set == "train" or (self.set == 'valid' and self.config['metrics_for_best_model'] == ['loss'])
        )

        self.mask_ratio = config['mask_ratio'] or 0.15
        self.poisson_lambda = config['poisson_lambda'] or 3.5
        self.poisson_distribution = torch.distributions.Poisson(rate=self.poisson_lambda)

    @classmethod
    def get_type(cls) -> str:
        return 'pretrain translate with BI-decoder'
        
    def __call__(self, samples):
        batch = {}
        source_text = [sample["source_text"] for sample in samples]
        source_ids = self.tokenizer(
            source_text,
            max_length=self.config['src_len'],
            truncation=True,
            padding=True,
            return_attention_mask=False,
            return_tensors='pt'
        )['input_ids']

        target_text = [sample["target_text"] for sample in samples]
        target_ids = self.tokenizer(
            target_text,
            max_length=self.config['tgt_len'],
            truncation=True,
            padding=True,
            return_attention_mask=False,
            return_tensors='pt'
        )['input_ids']

        if self.mask_ratio > 0.0:
            mask_ratio=random.uniform(0.1, 0.2)
            source_ids, labels_enc = my_mask(self.tokenizer, source_ids, mask_ratio)
            mask_ratio=random.uniform(0.2, 0.5)
            target_ids, labels_dec = my_mask(self.tokenizer, target_ids, mask_ratio)

        batch["source_ids"] = source_ids
        batch["source_mask"] = source_ids.ne(self.tokenizer.pad_token_id)
        batch["enc_labels"]=labels_enc
        
        batch["target_ids"] = target_ids
        batch["target_mask"] = target_ids.ne(self.tokenizer.pad_token_id)
        batch["dec_labels"]=labels_dec

        return batch

def my_mask(tokenizer, inputs,mask_ratio):
    bsz, seq_len = inputs.size()
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    num_to_mask = math.ceil((~special_tokens_mask).sum() * mask_ratio)

    token_indices = (~special_tokens_mask).nonzero()
    rand_index = torch.randperm(token_indices.size(0))
    rand_index  = rand_index[:num_to_mask] #生成了要mask的位置

    #生成labels，和src_ids相同尺寸的-100，torch.full_like，把其中rand_index位置变成src_id
    labels=torch.full_like(inputs, -100)
    labels[tuple(token_indices[rand_index].t())]=inputs[tuple(token_indices[rand_index].t())]

    #811:
    mask_index  = rand_index[:int(0.8*num_to_mask)]
    replace_index=rand_index[int(0.8*num_to_mask):int(0.9*num_to_mask)]

    #让inputs对应位置变成mask_id
    if len(mask_index)!=0:
        inputs[tuple(token_indices[mask_index].t())]=tokenizer.mask_token_id

    #replace：torch.multinomial，生成replace_index长度的，每个词为词表随机生成的内容
    if len(replace_index)!=0:
        vocab=torch.tensor(list(dict(tokenizer.vocab).values()), dtype=float)
        inputs[tuple(token_indices[replace_index].t())]=torch.multinomial(vocab, len(replace_index), replacement=True)

    inputs = _pad_sequence(inputs, tokenizer.pad_token_id)
    return inputs, labels
