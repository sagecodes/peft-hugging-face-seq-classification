from tasks.data import download_dataset, visualize_data

DATASET_NAME = "imdb"

data_paths = download_dataset(DATASET_NAME)
visualize_data(data_paths)

# Run the script from the command line:
# python -m workflows.download_data