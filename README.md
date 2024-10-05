Table Language Model
====================
This project provides code to pretrain a GPT based table language model to model the language of the scientific table cell contents at character level. It is based on [nanoGPT](https://github.com/karpathy/nanoGPT) codebase.

## Prerequisites
* Linux
* Python 3.9+

# Python Setup
Create a Python virtual environment

```bash
python3 -m venv ~/table_lm_venv
```
Install the requirements to your newly created environment

```bash
source ~/table_lm__env/bin/activate
cd $HOME/table_lm
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## Data Preparation for Pretraining

The raw data for this model is assumed in the following JSON format (one file per table)

```json
{
    "headers": [
       {"columns" : [{"content" : "<content>", "colspan" : 1},
                 {"content" : "<content>", "colspan" : 1}]
       },
       {"columns" : [{"content" : "<content>", "colspan" : 1},
                 {"content" : "<content>", "colspan" : 1}]
       }
    ],
    "rows" : [
              { 
               {"content": "<cell-content>", "colspan" : 1},
               {"content": "<cell-content>", "colspan" : 1}
              },
              { 
               {"content": "<cell-content>", "colspan" : 1},
               {"content": "<cell-content>", "colspan" : 1}
              }
             ]
}
```


The data preparation is done in two stages. First, training and validation intermediate files (one cell content per line) is generated via the script `pretrain_data_prep.py`

```bash
python pretrain_data_prep.py -i <JSON-table-files-directory> -o <training/validation-output-directory>

```
The second step maps characters to int, prepares the vocabulary metadata and creates encoded training and validation files.

First create the vocabulary

```
python data/table_llm_char/prepare.py -c vocab -i <input-text-corpus-file> -od <output-dir>
```
Then encode the training and validation text files generated in the first step. The script assumes the validation text corpus file is in the same directory as the training text corpus file.

```
python data/table_llm_char/prepare.py -c encode -i <train-text-corpus-file> -od <output-dir>
```

## Pretrain Table Language Model

Make sure you have a GPU with at least 16GB RAM. The script assumes that the `train.bin`, `val.bin` and `meta.pkl` files generated in the second step of data preparation are stored under `data/table_llm_char/` directory.

```
nohup python train.py config/train_table_char_llm.py &> nohup_pretrain.log &
```
The pretraining takes about 8 hours on a RTX 4090 24GB GPU.


## Row Merge Classifier Data Preparation

Data preparation for synthetic training data generation for the row merge classifier is done via the script `merge_model_data_prep.py`.

```bash
python merge_model_data_prep.py -c prep -i <table-json-files-dir> -o <output-dir>

```

Here the tables for training are represented in the same JSON format as the Table LM pretraining. 

After that, you can use the `merge_model_train.py` script to train/validate the
row merge classifier via transfer learning from the pretrained Table LM.

```bash
python merge_model_train.py -p <table-lm-checkpoint-file> -v <vocabulary-meta-file> -d <data-dir> -o <model-output-dir>

```
Here `<data-dir>` is the same as the `<output-dir>` as in the previous data
preparation step.









