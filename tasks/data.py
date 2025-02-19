"""
Download the dataset from hugging face for sequence classification
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

def download_dataset() ->  tuple[str, str, str]:

    # Load IMDB dataset
    dataset = load_dataset("imdb") # Load the IMDB dataset from Hugging Face
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Split training set into train and validation sets
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=42
    )

    data_dir = Path("datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

     # Save datasets as CSV files
    train_path = data_dir / "imdb_train.csv"
    val_path = data_dir / "imdb_val.csv"
    test_path = data_dir / "imdb_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return (
        train_path,
        val_path,
        test_path,
    )

if __name__ == "__main__":
    download_dataset()