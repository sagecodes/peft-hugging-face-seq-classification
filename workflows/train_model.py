from tasks.model import download_sequence_class_model, fine_tune_model, fine_tune_model_lora
import pandas as pd

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "pre_trained_models\distilbert-base-uncased"
data_paths = {
    "train": "datasets/imdb/train_data.csv",
    "val": "datasets/imdb/val_data.csv",
    "test": "datasets/imdb/test_data.csv",
}



# download_sequence_class_model(MODEL_NAME)
# fine_tune_model(MODEL_DIR, data_paths)

# Fine-tune the model with LoR
fine_tune_model_lora(
    MODEL_DIR,
    data_paths,
    epochs=2,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)


# Run the script from the command line:
# python -m workflows.train_model