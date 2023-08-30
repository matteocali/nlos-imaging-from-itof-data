import glob
import random
from pathlib import Path


def dts_splitter(out_path: Path, data_path: Path, shuffle: bool):
    """
    Function used to split the dataset in train, validation and test
        param:
            - out_path: path where to save the csv files
            - data_path: path where the dataset is located
            - shuffle: flag to shuffle the dataset
        return:
            - None
    """

    # Set the random seed
    random.seed(2097710)

    # Load the dataset folder
    elements = glob.glob1(str(data_path), "*.h5")
    n_of_elements = len(elements)

    # Shuffle the dataset if the flag is set
    if shuffle:
        random.shuffle(elements)

    # Split the dataset in train, validation and test (60, 20, 20)
    train = elements[: int(round(n_of_elements * 0.6, 0))]
    validation = elements[
        int(round(n_of_elements * 0.6, 0)) : int(round(n_of_elements * 0.8, 0))
    ]
    test = elements[int(round(n_of_elements * 0.8, 0)) :]

    # Create the csv files
    with open(out_path / (str(out_path.name) + "_train.csv"), "w") as f:
        for element in train:
            f.write(element + "\n")
    with open(out_path / (str(out_path.name) + "_validation.csv"), "w") as f:
        for element in validation:
            f.write(element + "\n")
    with open(out_path / (str(out_path.name) + "_test.csv"), "w") as f:
        for element in test:
            f.write(element + "\n")
