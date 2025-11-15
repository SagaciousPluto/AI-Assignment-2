import re
from tqdm.auto import tqdm
from transformers import RobertaTokenizerFast, AutoTokenizer
import torch
from torch.utils.data import TensorDataset

tqdm.pandas()

# ------------------------------
# Common cleaning
# ------------------------------
def clean_code(text):
    text = str(text).lower()
    text = re.sub(r'["\'`]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_A(train_df, val_df, test_df):
    train_df['clean_code'] = train_df['code'].progress_apply(clean_code)
    val_df['clean_code'] = val_df['code'].progress_apply(clean_code)
    test_df['clean_code'] = test_df['code'].progress_apply(clean_code)
    return train_df, val_df, test_df

# ------------------------------
# Model B: CodeBERT tokenizer
# ------------------------------
tokenizer_B = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

def preprocess_B(train_df, val_df, test_df, max_length=256):
    def tokenize(df):
        tokens = tokenizer_B(list(df["code"]), padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        labels = torch.tensor(df["label"].tolist())
        dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], labels)
        return dataset
    train_dataset = tokenize(train_df)
    val_dataset = tokenize(val_df)
    test_dataset = tokenize(test_df)
    return train_dataset, val_dataset, test_dataset, tokenizer_B

# ------------------------------
# Model C: DistilBERT tokenizer
# ------------------------------
tokenizer_C = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_C(train_df, val_df, test_df, max_length=256):
    def tokenize(df):
        tokens = tokenizer_C(list(df["code"]), padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        labels = torch.tensor(df["label"].tolist())
        dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], labels)
        return dataset
    train_dataset = tokenize(train_df)
    val_dataset = tokenize(val_df)
    test_dataset = tokenize(test_df)
    return train_dataset, val_dataset, test_dataset, tokenizer_C
