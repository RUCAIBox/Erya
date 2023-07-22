# Erya
This repository is the official implementation of our paper: *Towards Effective Ancient Chinese Translation: Dataset, Model, and Evaluation*.

## Datasets

You can download our Erya dataset and benchmark in: https://drive.google.com/drive/folders/1YjKjvjhzwE3QeCHn39yksCIyNMUVGP6M?usp=sharing.

## Model Weights

The pre-trained weights of Erya can be downloaded here: https://drive.google.com/drive/folders/1VV7DFABVGtRCDwJxWK1hzLWSy7Fn_E_X?usp=share_link .


## Fine-tuning and Evaluation

After downloading the dataset and the Erya model, you can use the Erya to generate translation as the following example.

```
from transformers import BertTokenizer, CPTForConditionalGeneration

tokenizer = BertTokenizer.from_pretrained("RUCAIBox/Erya")
model = CPTForConditionalGeneration.from_pretrained("RUCAIBox/Erya")

input_ids = tokenizer("安世字子孺，少以父任为郎。", return_tensors='pt')
input_ids.pop("token_type_ids")

pred_ids = model.generate(max_new_tokens=256, **input_ids)
print(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
```
