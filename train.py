import os
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from preprocess import *
from model_baseline_bert import *

def train_model(train_tokenized, val_tokenized, test_tokenized, 
                train_data_labels_list, val_data_labels_list, test_data_labels_list, 
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):

    # --- Directories ---
    os.makedirs("/results", exist_ok=True)
    os.makedirs("/plots", exist_ok=True)

    # --- Dataset & Dataloaders ---
    train_dataset = TensorDataset(train_tokenized['input_ids'], train_tokenized['attention_mask'], torch.tensor(train_data_labels_list))
    val_dataset = TensorDataset(val_tokenized['input_ids'], val_tokenized['attention_mask'], torch.tensor(val_data_labels_list))
    test_dataset = TensorDataset(test_tokenized['input_ids'], test_tokenized['attention_mask'], torch.tensor(test_data_labels_list))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # --- Model, Optimizer, Scheduler, Loss ---
    model = CodeBERTBaseline(num_labels=2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    # --- Metrics storage ---
    metrics = {"train_loss": [], "val_loss": [], "val_acc": []}

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch]
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total * 100

        # --- Store epoch metrics ---
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(avg_val_loss)
        metrics["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

        # --- Save checkpoint after each epoch ---
        ckpt_path = f"results/codebert_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"checkpoint: {ckpt_path}")

    # --- Save final model ---
    final_path = "results/codebert_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n model saved to {final_path}")

    # --- Final Testing ---
    model.eval()
    test_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch]
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            test_loss += loss.item()
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_test_loss = test_loss / len(test_loader)
    test_acc = correct / total * 100

    print(f"\nTest Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    # metrics are saved to a json file
    metrics["test_loss"] = avg_test_loss
    metrics["test_acc"] = test_acc

    metrics_path = "results/training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

    # --- Plot and save metrics ---
    plt.figure(figsize=(8,6))
    plt.plot(metrics["train_loss"], label="Train Loss", marker="o")
    plt.plot(metrics["val_loss"], label="Validation Loss", marker="o")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("/plots/loss_plot.pdf")
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(metrics["val_acc"], label="Validation Accuracy", color="green", marker="o")
    plt.title("Validation Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/accuracy_plot.pdf")
    plt.close()

    print("All processes completed.")

# Example call:
train_model(train_tokenized, val_tokenized, test_tokenized,
            train_data_labels_list, val_data_labels_list, test_data_labels_list)
