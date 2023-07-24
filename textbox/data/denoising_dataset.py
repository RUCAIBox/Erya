import math
import torch
import random
from typing import List, Dict
from nltk import sent_tokenize
from textbox.data.misc import _pad_sequence
import string
import re

class DenoisingCollate:
    def __init__(self, config, tokenizer, set):
        self.config = config
        self.tokenizer = tokenizer
        self.set = set
        self.mask_ratio = config['mask_ratio'] or 0.3
        self.poisson_lambda = config['poisson_lambda'] or 3.5
        self.permutate_sentence_ratio = config['permutate_sentence_ratio'] or 1.0
        self.poisson_distribution = torch.distributions.Poisson(rate=self.poisson_lambda)
        self.punctuation_ratio=config['punctuation_ratio'] or 1.0 #mask掉标点和断句
        self.switch_prob=config['switch_prob'] or 1.0 #双语数据当中进行替换的可能性

    @classmethod
    def get_type(cls) -> str:
        return 'denoising (BART)'

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError

    def __call__(self, samples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """Batching, adding whole word mask and permutate sentences
        Args:
            samples (dict): list of samples each samples contains input_ids field
        """
        # Handle dict or lists with proper padding and conversion to tensor.
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
        target_ids[torch.eq(target_ids, self.tokenizer.pad_token_id)] = -100
        batch["target_ids"] = target_ids

        #改成我们的加噪方式：（1）去掉全部标点和断句；（2）mask掉15%的词语；（3）打乱词语
        punctuation_string = string.punctuation+'＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
        #打乱句子+（if去掉标点断句）
        if self.permutate_sentence_ratio:
            new_source_text = []
            for text in source_text:
                texts = sent_tokenize(text)
                random.shuffle(texts)
                new_source_text.append(' '.join(texts))
            #同时打乱句子和删除断句标点
            if self.punctuation_ratio:
                # print("here")
                #nnew_source_text = ''
                nnew_source_text = []
                for text in new_source_text:
                    text = re.sub(r'[{}]+'.format(punctuation_string),'',text)
                    #nnew_source_text=nnew_source_text+text
                    nnew_source_text.append(text)
                # print("##############")
                # print(source_text)
                # print(nnew_source_text)
                source_ids = self.tokenizer(
                    nnew_source_text,
                    max_length=self.config['src_len'],
                    truncation=True,
                    padding=True,
                    return_attention_mask=False,
                    return_tensors='pt'
                )['input_ids']
            else:
                source_ids = self.tokenizer(
                    new_source_text,
                    max_length=self.config['src_len'],
                    truncation=True,
                    padding=True,
                    return_attention_mask=False,
                    return_tensors='pt'
                )['input_ids']
        #仅去掉标点断句
        elif self.punctuation_ratio:
            punctuation_string = string.punctuation+punctuation_str
            new_source_text = []
            for text in source_text:
                text = re.sub(r'[{}]+'.format(punctuation_string),'',text)
                new_source_text.append(text)
            source_ids = self.tokenizer(
                new_source_text,
                max_length=self.config['src_len'],
                truncation=True,
                padding=True,
                return_attention_mask=False,
                return_tensors='pt'
            )['input_ids']

        if self.mask_ratio > 0.0:
            source_ids = self.add_whole_word_mask(source_ids)

        batch["source_ids"] = source_ids
        batch["source_mask"] = source_ids.ne(self.tokenizer.pad_token_id)

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
