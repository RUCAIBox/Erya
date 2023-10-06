# Erya
This repository is the official implementation of NLPCC 2023 paper: **Towards Effective Ancient Chinese Translation: Dataset, Model, and Evaluation**. 

The implementation is based on the text generation library [TextBox 2.0](https://github.com/RUCAIBox/TextBox).

## Installation
You should clone the TextBox repository and follow its instructions.
```
git clone https://github.com/RUCAIBox/TextBox.git && cd TextBox
bash install.sh
```

## Datasets

You can download Erya datasets in: https://huggingface.co/datasets/RUCAIBox/Erya-dataset. You should download datasets such as xint in it and place them in the `dataset` folder.

To be specific, the datasets in Erya benchmark and their corresponding title are:

- hans: Book of han
- mings: Ming History
- shij: Shi Ji
- taip: Taiping Guangji
- xint: New Tang History
- xux: Xu Xiake's Travels


## Fine-tuning and Inference
After setting up the environment, you can either use Erya model in the zero-shot scenario, or further tune Erya4FT model for a better translation performance.

### Inference
We have released Erya model in: [https://huggingface.co/RUCAIBox/Erya](https://huggingface.co/RUCAIBox/Erya), which you can use directly to generate translation as below.

```
from transformers import BertTokenizer, CPTForConditionalGeneration

tokenizer = BertTokenizer.from_pretrained("RUCAIBox/Erya")
model = CPTForConditionalGeneration.from_pretrained("RUCAIBox/Erya")

input_ids = tokenizer("安世字子孺，少以父任为郎。", return_tensors='pt')
input_ids.pop("token_type_ids")

pred_ids = model.generate(max_new_tokens=256, **input_ids)
print(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
```

### Training
We also released Erya4FT model in: [https://huggingface.co/RUCAIBox/Erya4FT](https://huggingface.co/RUCAIBox/Erya4FT), which you can further tune on the translation dataset.

```
python run_textbox.py --model=CPT --dataset=[dataset] --model_path=RUCAIBox/Erya4FT --epochs=[epoch_nums]
```

