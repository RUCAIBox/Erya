import math
import torch
import random
from typing import List, Dict
from nltk import sent_tokenize
from textbox.data.misc import _pad_sequence


class DenoisingCollate_BI:

    def __init__(self, config, tokenizer, set):
        self.config = config
        self.tokenizer = tokenizer
        self.set = set
        self.mask_ratio = config['mask_ratio'] or 0.15
        self.poisson_lambda = config['poisson_lambda'] or 3.5
        self.poisson_distribution = torch.distributions.Poisson(rate=self.poisson_lambda)

    @classmethod
    def get_type(cls) -> str:
        return 'denoising with bi-decoder'

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError

    def __call__(self, samples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
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

        target_ids = source_ids.clone()

        if self.mask_ratio > 0.0:
            source_ids, labels_enc = my_mask(self.tokenizer, source_ids)

        batch["source_ids"] = source_ids
        batch["source_mask"] = source_ids.ne(self.tokenizer.pad_token_id)
        batch["enc_labels"]=labels_enc
        
        target_ids = source_ids.clone()
        labels_dec=labels_enc.clone()
        batch["target_ids"] = target_ids
        batch["target_mask"] = target_ids.ne(self.tokenizer.pad_token_id)
        batch["dec_labels"]=labels_dec

        return batch

def my_mask(tokenizer, inputs):
    bsz, seq_len = inputs.size()
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    mask_ratio=random.uniform(0.3, 0.4)
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
