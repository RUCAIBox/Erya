import math
import torch
import random
from typing import List, Dict
from nltk import sent_tokenize
from textbox.data.misc import _pad_sequence
import string
import re

class MonoCollate:
    def __init__(self, config, tokenizer, set):
        self.config = config
        self.tokenizer = tokenizer
        self.set = set
        self.mask_ratio = config['mask_ratio'] or 0.3
        self.poisson_lambda = config['poisson_lambda'] or 3.5
        self.permutate_sentence_ratio = config['permutate_sentence_ratio'] or 1.0
        self.poisson_distribution = torch.distributions.Poisson(rate=self.poisson_lambda)

    @classmethod
    def get_type(cls) -> str:
        return 'denoising (UNI & BI dec)'

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError

    def __call__(self, samples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch1 = {}
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
        target_ids[torch.eq(target_ids, self.tokenizer.pad_token_id)] = -100
        batch1["target_ids"] = target_ids

        punctuation_string = string.punctuation+'＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
        if self.permutate_sentence_ratio:
            new_source_text = []
            for text in source_text:
                texts = sent_tokenize(text)
                random.shuffle(texts)
                new_source_text.append(' '.join(texts))
            nnew_source_text = []
            for text in new_source_text:
                text = re.sub(r'[{}]+'.format(punctuation_string),'',text)
                nnew_source_text.append(text)
            source_ids = self.tokenizer(
                nnew_source_text,
                max_length=self.config['src_len'],
                truncation=True,
                padding=True,
                return_attention_mask=False,
                return_tensors='pt'
            )['input_ids']

        if self.mask_ratio > 0.0:
            source_ids = self.add_whole_word_mask(source_ids)

        batch1["source_ids"] = source_ids
        batch1["source_mask"] = source_ids.ne(self.tokenizer.pad_token_id)
        
        batch2 = {}
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
            new_mask_ratio=random.uniform(0.3, 0.4)
            source_ids, labels_enc = my_mask(self.tokenizer, source_ids, new_mask_ratio)

        batch2["source_ids"] = source_ids
        batch2["source_mask"] = source_ids.ne(self.tokenizer.pad_token_id)
        batch2["enc_labels"]=labels_enc
        
        target_ids = source_ids.clone()
        labels_dec=labels_enc.clone()
        batch2["target_ids"] = target_ids
        batch2["target_mask"] = target_ids.ne(self.tokenizer.pad_token_id)
        batch2["dec_labels"]=labels_dec

        batch['uni']=batch1
        batch['bi']=batch2
        batch['is_stage1']=True

        return batch

    def add_whole_word_mask(self, inputs):
        bsz, seq_len = inputs.size()
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # determine how many tokens we need to mask in total
        num_to_mask = math.ceil((~special_tokens_mask).sum() * self.mask_ratio)

        # generate a sufficient number of span lengths
        lengths = self.poisson_distribution.sample(sample_shape=(num_to_mask,))
        cum_length = torch.cumsum(lengths, 0)
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat([lengths, self.poisson_distribution.sample(sample_shape=(num_to_mask,))])
            cum_length = torch.cumsum(lengths, 0)

        # trim to about num_to_mask tokens
        idx = ((cum_length - num_to_mask) >= 0).nonzero()[0][0]
        lengths[idx] = num_to_mask - (0 if idx == 0 else cum_length[idx - 1])
        num_span = idx + 1
        lengths = lengths[:num_span]

        # handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_span - lengths.size(0)
        num_span -= num_inserts

        # select span start indices
        token_indices = (~special_tokens_mask).nonzero()
        rand_span = torch.randperm(token_indices.size(0))
        span_starts = rand_span[:num_span]

        # prepare mask and mask span start indices
        masked_indices = token_indices[span_starts]
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        mask[tuple(masked_indices.t())] = True
        lengths -= 1

        # fill up spans
        remaining = (lengths > 0) & (masked_indices[:, 1] < seq_len - 1)
        while torch.any(remaining):
            masked_indices[remaining, 1] += 1
            mask[tuple(masked_indices.t())] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < seq_len - 1)

        # place the mask tokens
        mask[special_tokens_mask] = False
        inputs[mask] = self.tokenizer.mask_token_id

        # remove mask tokens that are not starts of spans
        to_remove = mask & mask.roll(1, 1) | inputs.eq(self.tokenizer.pad_token_id)
        # calculate the number of inserted mask token per row
        inserts_num = torch.bincount(token_indices[rand_span[:num_inserts]][:, 0], minlength=bsz)
        new_inputs = []
        for i, example in enumerate(inputs):
            new_example = example[~to_remove[i]]
            n = inserts_num[i]
            if n:
                new_num = n + new_example.size(0)
                noise_mask = torch.zeros(new_num, dtype=torch.bool)
                mask_indices = torch.randperm(new_num - 2)[:n] + 1
                noise_mask[mask_indices] = 1
                result = torch.LongTensor(new_num.item())
                result[mask_indices] = self.tokenizer.mask_token_id
                result[~noise_mask] = new_example
                new_example = result
            new_inputs.append(new_example)
        new_inputs = _pad_sequence(new_inputs, self.tokenizer.pad_token_id)
        return new_inputs

def my_mask(tokenizer, inputs, mask_ratio):
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
