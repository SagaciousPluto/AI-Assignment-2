Code Classification Baselines

This repository contains implementations of three baseline models for code classification tasks, including XGBoost and transformer-based models (CodeBERT and DistilBERT). It also provides data loading, preprocessing, training, and evaluation scripts.

---

Table of Contents

* Project Structure
* Installation
* Data
* Preprocessing
* Training
* Evaluation
* Model Details
* Results

---

Project Structure

```
.
├── data_loader.py           # Load training, validation, and test datasets
├── preprocess.py            # Data cleaning and tokenization functions for all baselines
├── model_baseline_A.py      # XGBoost classifier
├── model_baseline_B.py      # CodeBERT-based classifier
├── model_baseline_C.py      # DistilBERT-based classifier
├── train.py                 # Training script for all models
├── evaluate.py              # Evaluation script for saved models
└── results/                 # Folder where trained models are saved
```

---

Installation

Create a Python environment and install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ensure `torch` is installed with GPU support if using GPU.

---

Data

The code expects **Parquet files** for training, validation, and test sets with the following columns:

* `code` – the source code snippet
* `label` – the target label

Example:

```python
train_df, val_df, test_df = load_data(train_dir, val_dir, test_dir)
```

---

Preprocessing

* **Baseline A**: Cleans code (lowercase, remove quotes/backticks, normalize spaces).
* **Baseline B**: Tokenizes code using `CodeBERT` tokenizer.
* **Baseline C**: Tokenizes code using `DistilBERT` tokenizer.

```python
from preprocess import preprocess_A, preprocess_B, preprocess_C
```

---

Training

Run `train.py` with the desired baseline model:

```bash
python train.py --model A 
python train.py --model B --epochs 3 --batch_size 8
python train.py --model C --epochs 3 --batch_size 8
```

* `--model`: Choose baseline model (`A`, `B`, or `C`)
* `--epochs`: Number of training epochs
* `--batch_size`: Batch size for training

**Example for training Baseline A:**

```bash
python train.py --model A
```

Trained models are saved in `results/`:

* `xgb_baseline_A.pkl`
* `tfidf_vectorizer.pkl`
* `codebert_baseline_B.pt`
* `codebert_baseline_C.pt`

---

Evaluation

Use `evaluate.py` to evaluate a saved model:

```bash
python evaluate.py --model_type B --model_path results/codebert_baseline_B.pt --data path_to_data.pt --batch_size 16
```

Outputs accuracy and detailed classification report.

---

Model Details

**Baseline A – XGBoost**

* Tree-based model with GPU support
* TF-IDF character n-grams (3-4) as input
* Early stopping enabled

**Baseline B – CodeBERT**

* Pretrained `microsoft/codebert-base`
* Dropout: 0.3
* Linear classifier on top of pooled output
* Optimizer: AdamW, learning rate: 2e-5

**Baseline C – DistilBERT**

* Pretrained `distilbert-base-uncased`
* Dropout: 0.3
* Linear classifier on mean-pooled hidden states
* Optimizer: AdamW, learning rate: 2e-5

---

Results

After training, validation accuracy and loss are printed per epoch. Models are saved for later inference or evaluation.
