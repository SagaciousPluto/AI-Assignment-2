import torch
from torch import nn
from transformers import AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_baseline_C(model_name="distilbert-base-uncased", num_labels=2, dropout=0.3):
    class CodeClassifier(nn.Module):
        def __init__(self, model_name, num_labels, dropout):
            super(CodeClassifier, self).__init__()
            self.transformer = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)
        def forward(self, input_ids, attention_mask):
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            mean_pooled = outputs.last_hidden_state.mean(dim=1)
            mean_pooled = self.dropout(mean_pooled)
            logits = self.classifier(mean_pooled)
            return logits
    return CodeClassifier(model_name, num_labels, dropout).to(DEVICE)
