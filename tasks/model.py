"""
This module downloads a pretrained sequence classification model from Hugging Face,
fine-tunes it on a dataset, evaluates its performance on a test set, and saves the trained model.

Dependencies:
    Install required packages by running:
        pip install -r requirements.txt
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path


# --------------------------------
# Download the model from hugging face for sequence classification
# --------------------------------
def download_sequence_class_model(model_name: str = "distilbert-base-uncased") -> str:

    model_dir = Path(f"pre_trained_models/{model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save model and tokenizer
    model.save_pretrained(
        model_dir,
    )
    tokenizer.save_pretrained(model_dir)

    return model_dir


# --------------------------------
# Fine-tune the model LoRA
# --------------------------------
def fine_tune_model():
    pass
