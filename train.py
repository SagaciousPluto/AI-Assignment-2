import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from preprocess import *
from model_baseline_bert import CodeBERTBaseline

def train_model(data_path, epochs=3, batch_size=8, lr=2e-5):
    # Load preprocessed data
    train_tokenized, val_tokenized, train_data_labels_list, val_data_labels_list = load_and_preprocess_data(data_path)
    
    # Prepare datasets
    train_dataset = TensorDataset(train_tokenized['input_ids'], train_tokenized['attention_mask'], torch.tensor(train_data_labels_list))
    test_dataset = TensorDataset(val_tokenized['input_ids'], val_tokenized['attention_mask'], torch.tensor(val_data_labels_list))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodeBERTBaseline(num_labels=2).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    
