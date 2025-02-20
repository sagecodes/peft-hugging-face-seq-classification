"""
Download the dataset from hugging face for sequence classification
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import base64

# --------------------------------
# Download the dataset
# --------------------------------
def download_dataset(dataset_name: str = "imdb") ->  tuple[str, str, str]:

    # Load IMDB dataset
    dataset = load_dataset(dataset_name) # Load the IMDB dataset from Hugging Face
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

    return (
        train_path,
        val_path,
        test_path,
    )

# --------------------------------
# Visualize the data
# --------------------------------

def visualize_data():
    train_df = pd.read_csv("datasets/imdb_train.csv")
    print(train_df.head())

    val_df = pd.read_csv("datasets/imdb_val.csv")
    print(val_df.head())

    test_df = pd.read_csv("datasets/imdb_test.csv")
    print(test_df.head())

    # Sample reviews from the datasets
    train_positive_review = train_df[train_df["label"] == 1].iloc[0]["text"]
    train_negative_review = train_df[train_df["label"] == 0].iloc[0]["text"]
    val_positive_review = val_df[val_df["label"] == 1].iloc[0]["text"]
    val_negative_review = val_df[val_df["label"] == 0].iloc[0]["text"]
    test_positive_review = test_df[test_df["label"] == 1].iloc[0]["text"]
    test_negative_review = test_df[test_df["label"] == 0].iloc[0]["text"]

     # Convert images to base64 for embedding
    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    
if __name__ == "__main__":
    download_dataset()