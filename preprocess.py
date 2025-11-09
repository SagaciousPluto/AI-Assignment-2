import pandas as pd
from data_loader import *
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast

train_data = train_data[["code", "label", "language"]][:50000]
train_data = train_data.dropna()
train_data = train_data.drop_duplicates()

val_data = val_data[["code", "label", "language"]][:10000]
val_data = val_data.dropna()
val_data = val_data.drop_duplicates()

tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

def tokenize(examples):
    return tokenizer(
        examples["code"].tolist(),
        padding = "max_length",
        truncation = True,
        max_length = 300,
        return_tensors="pt"
    )

train_tokenized = tokenize(train_data)
val_tokenized = tokenize(val_data)

train_data_labels_list = train_data["label"].tolist()
val_data_labels_list = val_data["label"].tolist()