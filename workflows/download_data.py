from tasks.data import download_dataset, visualize_data

# --------------------------------
# Define the dataset name to download
# --------------------------------

DATASET_NAME = "imdb"

# --------------------------------
# Download and vizualize the dataset
# --------------------------------
data_paths = download_dataset(DATASET_NAME)
visualize_data(data_paths)

# Run the script from the command line:
# python -m workflows.download_data
