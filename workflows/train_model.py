from tasks.model import download_sequence_class_model, fine_tune_model
import pandas as pd

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "pre_trained_models\distilbert-base-uncased"
data_paths = {
    "train": "datasets/imdb/train_data.csv",
    "val": "datasets/imdb/val_data.csv",
    "test": "datasets/imdb/test_data.csv",
}



# download_sequence_class_model(MODEL_NAME)
fine_tune_model(MODEL_DIR, data_paths)



# Run the script from the command line:
# python -m workflows.train_model