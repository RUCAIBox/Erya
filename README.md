# Erya
This repository is the official implementation of NLPCC 2023 paper: **Towards Effective Ancient Chinese Translation: Dataset, Model, and Evaluation**. 

The implementation is based on the text generation library [TextBox 2.0](https://github.com/RUCAIBox/TextBox).

## Introduction
- We collect, clean, and classify ancient Chinese materials from various sources, forming the most extensive ancient Chinese resource Erya to date.
- We devise Erya training method oriented towards ancient Chinese, composing two jointly-working tasks: disyllabic aligned substitution (DAS) and dual masked language model (DMLM).
- We build a benchmark to judge ancient Chinese translation quality in different scenarios and evaluate the ancient Chinese translation capacities of various existing models.

## Installation
You should clone the TextBox repository and follow its instructions.
```
git clone https://github.com/RUCAIBox/TextBox.git && cd TextBox
bash install.sh
```

## Datasets

You can download Erya datasets in: https://huggingface.co/datasets/RUCAIBox/Erya-dataset. You should download datasets such as xint in it and place them in the `dataset` folder.

To be specific, the datasets and their corresponding title are:

- hans: Book of han
- mings: Ming History
- shij: Shi Ji
- taip: Taiping Guangji
- xint: New Tang History
- xux: Xu Xiake's Travels


## Fine-tuning and Inference
After setting up the environment, you are able to use Erya model directly in the zero-shot scenario, or further tune the Erya4FT model for a better translation .

### Inference
We have released Erya model in: [https://huggingface.co/RUCAIBox/Erya](https://huggingface.co/RUCAIBox/Erya-zero), which you can use directly to generate translation as the following example.

```
from transformers import BertTokenizer, CPTForConditionalGeneration

tokenizer = BertTokenizer.from_pretrained("RUCAIBox/Erya-zero")
model = CPTForConditionalGeneration.from_pretrained("RUCAIBox/Erya-zero")

input_ids = tokenizer("安世字子孺，少以父任为郎。", return_tensors='pt')
input_ids.pop("token_type_ids")

pred_ids = model.generate(max_new_tokens=256, **input_ids)
print(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
```

### Training
You are able to further tune Erya4FT model on the translation dataset (like xint).
```
python run_textbox.py --model=CPT --dataset=[dataset] --model_path=RUCAIBox/Erya --epochs=[epoch_nums]
```

