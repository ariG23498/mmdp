from random import choice
from datasets import load_dataset
from mmdp import config
from mmdp.utils import visualize_sample


if __name__ == "__main__":
    # Load the dataset from Hugging Face
    train_ds = load_dataset(
        path=config.train_dataset_path, name=config.train_dataset_name, split="train"
    )

    # Samples one data point from the dataset at random and visualizes it
    sample = choice(train_ds)
    visualize_sample(sample)
