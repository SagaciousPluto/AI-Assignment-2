import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from transformers import RobertaModel
from preprocess_A import preprocess_A
from preprocess_B import preprocess_B
from preprocess_C import preprocess_C
from models.model_A import ModelA
from models.model_B import ModelB
from models.model_C import ModelC
from datasets import CodeDataset

def load_preprocess(name):
    if name == "A":
        return preprocess_A
    if name == "B":
        return preprocess_B
    return preprocess_C

def load_model(name, num_labels):
    if name == "A":
        return ModelA(num_labels)
    if name == "B":
        return ModelB(num_labels)
    return ModelC(num_labels)

def evaluate(model, dataloader, device):
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(ids, mask)
            p = torch.argmax(outputs, dim=1)
            preds.extend(p.cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(trues, preds)
    report = classification_report(trues, preds)
    return acc, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    df = torch.load(args.data)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    num_labels = len(set(labels))

    preprocess_fn = load_preprocess(args.model_type)
    tokenizer, enc = preprocess_fn()

    tok = tokenizer(texts, truncation=True, padding=True, max_length=256)
    dataset = CodeDataset(tok, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_type, num_labels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    acc, report = evaluate(model, loader, device)
    print("Accuracy:", acc)
    print(report)

if __name__ == "__main__":
    main()
