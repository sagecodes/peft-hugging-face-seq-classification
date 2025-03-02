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
from datasets import Dataset
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import bitsandbytes as bnb



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
# Fine-tune the full model
# --------------------------------
def fine_tune_model(model_dir: str, data_paths: dict, epochs: int = 2) -> str:

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load a sample of the data
    train_data = pd.read_csv(data_paths["train"]).sample(n=500, random_state=42)
    val_data = pd.read_csv(data_paths["val"]).sample(n=100, random_state=42)

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
    train_data = pd.read_csv(data_paths["train"]).sample(n=500, random_state=42)
    val_data = pd.read_csv(data_paths["val"]).sample(n=100, random_state=42)

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
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype= torch.bfloat16,
            bnb_4bit_use_double_quant= True,
            llm_int8_skip_modules=["classifier", "pre_classifier"]
        )

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, 
                                                               quantization_config=bnb_config,
                                                                torch_dtype=torch.bfloat16,)
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
    train_data = pd.read_csv(data_paths["train"]).sample(n=500, random_state=42)
    val_data = pd.read_csv(data_paths["val"]).sample(n=100, random_state=42)

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