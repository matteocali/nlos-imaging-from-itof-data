import h5py
import numpy as np
from pathlib import Path


def load(path: Path) -> dict:
    """
    Load the data from the h5 file.\n
    Param:
        - path (Path): The path of the h5 file\n
    Return:
        - data (dict): The data loaded from the h5 file
    """

    data = dict()

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        for key in keys:
            elm = f[key]
            data[key] = np.array(elm)
    
    return data


def save(data: dict, path: Path) -> None:
    """
    Save the data in the h5 file.\n
    Param:
        - data (dict): The data to save in the h5 file
        - path (Path): The path of the h5 file
    """

    with h5py.File(path, "w") as f:
        for key in data.keys():
            f[key] = data[key]
