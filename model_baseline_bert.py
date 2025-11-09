import torch
from torch import nn
from transformers import RobertaModel

class CodeBERTBaseline(nn.Module):
    def __init__(self, label_count=2):
        super(CodeBERTBaseline, self).__init__()
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.codebert.config.hidden_size, label_count)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
