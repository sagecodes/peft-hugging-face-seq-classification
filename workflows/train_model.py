# from tasks.data import download_data
from tasks.model import download_sequence_class_model, fine_tune_model, fine_tune_model_lora, fine_tune_model_qlora, evaluate_model
import pandas as pd
import os

DATASET_NAME = "imdb" 
MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "pre_trained_models\distilbert-base-uncased"
data_paths = {
    "train": "datasets/imdb/train_data.csv",
    "val": "datasets/imdb/val_data.csv",
    "test": "datasets/imdb/test_data.csv",
}


def train_model_pipeline():
    # check if the model is already downloaded
    if not os.path.exists(MODEL_DIR):
        download_sequence_class_model(MODEL_NAME)
    
    # check if the data is available
    for path in data_paths.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file {path} not found")

    # Fine-tune the model
    fine_tune_model(MODEL_DIR, data_paths)

    # Fine-tune the model with LoRA
    fine_tune_model_lora(
        MODEL_DIR,
        data_paths,
        epochs=2,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )

# download_sequence_class_model(MODEL_NAME)
# fine_tune_model(MODEL_DIR, data_paths)

#--------------------------------
# Fine-tune the model with LoRa 
# #--------------------------------
fine_tune_model_lora(
    MODEL_DIR,
    data_paths,
    epochs=2,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)
# # python -m workflows.train_model fine_tune_model_lora

#--------------------------------
# Fine-tune the model with QLoRa 
# #--------------------------------
# fine_tune_model_qlora(
#     MODEL_DIR,
#     data_paths,
#     epochs=2,
#     lora_r= 8,
#     lora_alpha = 16,
#     lora_dropout= 0.1,
# )

# python -m workflows.train_model fine_tune_model_qlora

model_lora_dir = "model_lora"
# model_qlora_dir = "model_qlora"

evaluate_model(model_lora_dir, data_paths)
# evaluate_model(model_qlora_dir, data_paths)

# python -m workflows.train_model evaluate_model


# Run the script from the command line:

# python -m workflows.train_model