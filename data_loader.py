import pandas as pd

def load_data(train_dir, val_dir, test_dir):
    train_df = pd.read_parquet(train_dir)
    val_df = pd.read_parquet(val_dir)
    test_df = pd.read_parquet(test_dir)
    print(f"Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")
    return train_df, val_df, test_df
