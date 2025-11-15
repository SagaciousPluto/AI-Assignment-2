import os
import torch
import joblib
import argparse
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from data_loader import load_data
from preprocess import preprocess_A, preprocess_B, preprocess_C
from model_baseline_A import get_baseline_A
from model_baseline_B import get_baseline_B
from model_baseline_C import get_baseline_C
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="A", choices=["A","B","C"])
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

train_dir = "/kaggle/input/semeval-2026-task13/SemEval-2026-Task13/task_a/task_a_training_set_1.parquet"
val_dir   = "/kaggle/input/semeval-2026-task13/SemEval-2026-Task13/task_a/task_a_validation_set.parquet"
test_dir  = "/kaggle/input/semeval-2026-task13/SemEval-2026-Task13/task_a/task_a_test_set_sample.parquet"

train_df, val_df, test_df = load_data(train_dir, val_dir, test_dir)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# -------------------------------
# MODEL A
# -------------------------------
if args.model == "A":
    train_df, val_df, test_df = preprocess_A(train_df, val_df, test_df)
    y_train, y_val = train_df["label"], val_df["label"]

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), max_features=30000, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["clean_code"]).astype("float32")
    X_val   = vectorizer.transform(val_df["clean_code"]).astype("float32")

    model = get_baseline_A()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)

    val_preds = model.predict(X_val)
    print("Validation Accuracy:", (val_preds == y_val).mean())

    joblib.dump(model, "results/xgb_baseline_A.pkl")
    joblib.dump(vectorizer, "results/tfidf_vectorizer.pkl")
    print("✅ Model A saved.")

# -------------------------------
# MODEL B
# -------------------------------
elif args.model == "B":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset, val_dataset, test_dataset, tokenizer = preprocess_B(train_df, val_df, test_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = get_baseline_B()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for ids, attn, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            ids, attn, labels = ids.to(DEVICE), attn.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ids, attn)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for ids, attn, labels in val_loader:
                ids, attn, labels = ids.to(DEVICE), attn.to(DEVICE), labels.to(DEVICE)
                logits = model(ids, attn)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total * 100
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(val_loss / len(val_loader))
        metrics["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "results/codebert_baseline_B.pt")

# -------------------------------
# MODEL C
# -------------------------------
elif args.model == "C":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset, val_dataset, test_dataset, tokenizer = preprocess_C(train_df, val_df, test_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = get_baseline_C()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for ids, attn, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            ids, attn, labels = ids.to(DEVICE), attn.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ids, attn)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for ids, attn, labels in val_loader:
                ids, attn, labels = ids.to(DEVICE), attn.to(DEVICE), labels.to(DEVICE)
                logits = model(ids, attn)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total * 100
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(val_loss / len(val_loader))
        metrics["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "results/codebert_baseline_C.pt")
