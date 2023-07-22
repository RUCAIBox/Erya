# Erya
This repository is the official implementation of our paper: **Towards Effective Ancient Chinese Translation: Dataset, Model, and Evaluation**, based on our text generation library TextBox 2.0.

## Introduction
- We collect, clean, and classify ancient Chinese materials from various sources, forming the most extensive ancient Chinese resource Erya to date.
- We devise Erya training method oriented towards ancient Chinese, composing two jointly-working tasks: disyllabic aligned substitution (DAS) and dual masked language model (DMLM).
- We build a benchmark to judge ancient Chinese translation quality in different scenarios and evaluate the ancient Chinese translation capacities of various existing models.


## Datasets

You can download Erya datasets in: https://huggingface.co/RUCAIBox. You should create a folder dataset and download dataset such as xint in it.

To be specific, the datasets and their corresponding title are:

- hans: Book of han
- mings: Ming History
- shij: Shi Ji
- taip: Taiping Guangji
- xint: New Tang History
- xux: Xu Xiake's Travels


## Fine-tuning, Inference and Evaluation
After setting up the environment, you are able to conduct training, inference, and evaluation using our code in a pipeline.

### Training
```
python run_textbox.py --model=CPT --dataset=trans --pretrain_task=para --model_path=fnlp/cpt --epochs=[epoch_nums] --uni_weight=0.7 --bi_weight=0.3
```
`uni_weight` can be the weight of DAS loss, and `bi_weight` can be the weight of DMLM loss.


### Inference
We have released Erya model in HuggingFace, which you can use to generate translation as the following example.

```
from transformers import BertTokenizer, CPTForConditionalGeneration

tokenizer = BertTokenizer.from_pretrained("RUCAIBox/Erya")
model = CPTForConditionalGeneration.from_pretrained("RUCAIBox/Erya")

input_ids = tokenizer("安世字子孺，少以父任为郎。", return_tensors='pt')
input_ids.pop("token_type_ids")

pred_ids = model.generate(max_new_tokens=256, **input_ids)
print(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
```
