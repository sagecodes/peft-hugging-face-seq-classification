"""
This module downloads a pretrained sequence classification model from Hugging Face,
fine-tunes it on a dataset, evaluates its performance on a test set, and saves the trained model.

Dependencies:
    Install required packages by running:
        pip install -r requirements.txt
"""

from pathlib import Path

import bitsandbytes as bnb
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments, EarlyStoppingCallback)

from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json


# --------------------------------
# Download the model from hugging face for sequence classification
# --------------------------------
def download_sequence_class_model(model_name: str = "distilbert-base-uncased") -> str:
    """
    Download a pre-trained sequence classification model from Hugging Face.

    Args:
        model_name (str): The name of the pre-trained model to download from HF.

    Returns:
        str: The path to the downloaded model.
    """

    # Create a directory to save the model
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
# Fine-tune the full model
# --------------------------------
def fine_tune_model(model_dir: str, data_paths: dict, epochs: int = 2) -> str:
    """
    Fine-tune a pre-trained sequence classification model on a dataset.

    Args:
        model_dir (str): The path to the pre-trained model.
        data_paths (dict): A dictionary containing the paths to the train, validation, and test datasets.
        epochs (int): The number of epochs for training.

    Returns:
        str: The path to the trained model.
    """
    
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load a sample of the data
    train_data = pd.read_csv(data_paths["train"]).sample(n=5000, random_state=42)
    val_data = pd.read_csv(data_paths["val"]).sample(n=1000, random_state=42)

    # Convert DataFrames to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # Tokenize the data
    def tokenizer_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenizer_function)
    tokenized_val_dataset = val_dataset.map(tokenizer_function)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results", num_train_epochs=epochs, evaluation_strategy="epoch"
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
    )

    trainer.train()

    # Save the trained model
    output_dir = Path("trained_model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # return the output directory
    return output_dir


# --------------------------------
# Fine-tune LoRA
# --------------------------------
def fine_tune_model_lora(
    model_dir: str,
    data_paths: dict,
    epochs: int = 2,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> str:
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Apply LoRA adaptation
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "k_lin", "v_lin"],  # adjust depending on model
    )

    model = get_peft_model(model, lora_config)

    # Load a sample of the data
    train_data = pd.read_csv(data_paths["train"]).sample(n=5000, random_state=42)
    val_data = pd.read_csv(data_paths["val"]).sample(n=1000, random_state=42)

    # Convert DataFrames to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # Tokenize the data
    def tokenizer_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenizer_function)
    tokenized_val_dataset = val_dataset.map(tokenizer_function)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    # Save the trained model
    output_dir = Path("model_lora")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return str(output_dir)


# --------------------------------
# Fine-tune QLoRA
# --------------------------------
def fine_tune_model_qlora(
    model_dir: str,
    data_paths: dict,
    epochs: int = 2,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> str:

    # configure the Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["classifier", "pre_classifier"],
    )

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Apply LoRA adaptation
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "k_lin", "v_lin"],  # adjust depending on model
    )

    model = get_peft_model(model, lora_config)

    # Load a sample of the data
    train_data = pd.read_csv(data_paths["train"]).sample(n=5000, random_state=42)
    val_data = pd.read_csv(data_paths["val"]).sample(n=1000, random_state=42)

    # Convert DataFrames to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # Tokenize the data
    def tokenizer_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenizer_function)
    tokenized_val_dataset = val_dataset.map(tokenizer_function)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        save_total_limit=1,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
    )

    trainer.train()

    # Save the trained model
    output_dir = Path("model_qlora")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return str(output_dir)

# --------------------------------
# Evaluate the model
# --------------------------------

def evaluate_model(model_dir: str, data_paths: dict) -> dict:
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the test data
    # test_data = pd.read_csv(data_path)
    test_data = pd.read_csv(data_paths["test"]).sample(n=1000, random_state=42)


    # Convert DataFrame to Hugging Face dataset
    test_dataset = Dataset.from_pandas(test_data)

    # Tokenize the data
    def tokenizer_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_test_dataset = test_dataset.map(tokenizer_function)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results", evaluation_strategy="epoch"
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test_dataset,
    )

    # Evaluate the model
    eval_results = trainer.evaluate()
    predictions = trainer.predict(tokenized_test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    labels = test_data["label"].values

    # Generate Classification Report
    report = classification_report(labels, preds, output_dict=True)
    report_path = f"{model_dir}/classification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(labels, preds)
    cm_path = f"{model_dir}/confusion_matrix.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(labels)), yticklabels=sorted(set(labels)))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()
    
    # Generate ROC Curve
    if len(set(labels)) == 2:  # Only for binary classification
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_path = f"{model_dir}/roc_curve.png"
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(roc_path)
        plt.close()
    else:
        roc_path = None
    
    return {
        "eval_results": eval_results,
        "classification_report": report_path,
        "confusion_matrix": cm_path,
        "roc_curve": roc_path
    }