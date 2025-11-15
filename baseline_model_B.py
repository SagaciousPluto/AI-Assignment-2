from torch import nn
import torch
from transformers import RobertaModel

# Hyperparameters
LR = 2e-5
BATCH_SIZE = 8
EPOCHS: 3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0
MAX_LENGTH = 256
DROPOUT = 0.3
RANDOM_SEED = 42
MODEL_NAME = "microsoft/codebert-base"
DEVICE = "cuda"

def get_baseline_B(num_labels=2):
    class CodeBERTBaseline(nn.Module):
        def _init_(self, num_labels):
            super(CodeBERTBaseline, self)._init_()
            self.codebert = RobertaModel.from_pretrained(MODEL_NAME)
            self.dropout = nn.Dropout(DROPOUT)
            self.classifier = nn.Linear(self.codebert.config.hidden_size, num_labels)
        def forward(self, input_ids, attention_mask):
            outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits
    return CodeBERTBaseline(num_labels).to(DEVICE)