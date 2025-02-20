"""
Download the dataset from hugging face for sequence classification
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import base64
from tasks.utils import image_to_base64


# --------------------------------
# Download the dataset
# --------------------------------
def download_dataset(dataset_name: str = "imdb") -> dict:

    # Load Hugging Face Dataset
    dataset = load_dataset(dataset_name)
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Split training set into train and validation sets
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=42
    )

    data_dir = Path(f"datasets/{dataset_name}")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets as CSV files
    train_path = data_dir / "train_data.csv"
    val_path = data_dir / "val_data.csv"
    test_path = data_dir / "test_data.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    data_paths = {"train": train_path, "val": val_path, "test": test_path}

    return data_paths


# --------------------------------
# Visualize the data
# --------------------------------
def visualize_data(data_paths: dict, 
                    label_column: str = "label",
                     text_column: str = "text") -> str:

    sample_reviews = {}

    for dataset_name, file_path in data_paths.items():
        df = pd.read_csv(file_path)
        print(f"Dataset: {dataset_name}")
        print(df.head())

        # Extract a positive and negative review if they exist
        positive_review = (
            df[df[label_column] == 1].iloc[0][text_column]
            if not df[df[label_column] == 1].empty
            else "No positive review found"
        )
        negative_review = (
            df[df[label_column] == 0].iloc[0][text_column]
            if not df[df[label_column] == 0].empty
            else "No negative review found"
        )

        sample_reviews[dataset_name] = {
            "positive": positive_review,
            "negative": negative_review,
        }

    return sample_reviews

    

if __name__ == "__main__":
    download_dataset()
