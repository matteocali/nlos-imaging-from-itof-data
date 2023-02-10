import torch
import numpy as np
import h5py as h5
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
from utils.utils import phi_func


class NlosTransientDataset(Dataset):
    """
    NLOS Transient Dataset class
        param:
            - dts_folder: path to the dataset folder
            - csv_file: path to the csv file containing the list of the images
            - transform: transformation to apply to the data
    """

    def __init__(self, dts_folder: Path, csv_file: Path, transform=None):
        """
        Constructor of the NlosTransientDataset class
            param:
                - dts_folder: path to the dataset folder
                - csv_file: path to the csv file containing the list of the images
                - transform: transformation to apply to the data
        """

        # Set the transform attribute
        self.transform = transform

        frequencies = np.array((20e06, 50e06, 60e06), dtype=np.float32)  # Define the frequencies used by the considered iToF sensor
        phi = phi_func(frequencies)                                      # Compute the phi matrix (iToF data)
        nf = phi.shape[0]                                                # Extract the number of frequencies

        # Load the csv files
        images = []
        with open(csv_file, "r") as f:
            images += f.readlines()
        images = [dts_folder / (image[:-1]) for image in images]

        # Load the first image to get the size of the images
        num_images = len(images)
        with h5.File(images[0], "r") as h:
            temp_data = h["data"][:]            # type: ignore
        [dim_x, dim_y, dim_t] = temp_data.shape # type: ignore

        itof_data = np.zeros((num_images, dim_x, dim_y, nf), dtype=np.float32)  # Create the iToF data tensor
        gt_alpha = np.zeros((num_images, dim_x, dim_y), dtype=np.float32)       # Create the ground truth alpha map tensor
        gt_depth = np.zeros((num_images, dim_x, dim_y), dtype=np.float32)       # Create the ground truth depth map tensor

        names = []
        count = 0
        for image in tqdm(images, desc="Loading images", total=num_images):
            file_name = str(image).split("/")[-1][:-3]
            names.append(file_name)

            # Load the transient data
            with h5.File(image, "r") as h:
                temp_data = h["data"][:]                       # Load the transient data  # type: ignore
                temp_gt_depth = h["depth_map"][:]              # Load the ground truth depth data  # type: ignore
                temp_gt_alpha = h["alpha_map"][:].astype(int)  # Load the ground truth alpha map data  # type: ignore

            temp_lin = np.reshape(temp_data, (dim_x * dim_y, dim_t))  # Reshape the transient data  # type: ignore

            # Computation with the direct component
            v = np.matmul(temp_lin, np.transpose(phi))
            v = np.reshape(v, (dim_x, dim_y, phi.shape[0]))
            itof_data[count, ...] = v

            # Add the gt depth and alpha map
            gt_depth[count, ...] = temp_gt_depth
            gt_alpha[count, ...] = temp_gt_alpha

            # Increment the counter
            count += 1

        # Move the axis to have the channels as the second dimension instead of being the last one
        itof_data = np.moveaxis(itof_data, 3, 1)

        # Transform the data to torch tensors
        self.itof_data = torch.from_numpy(itof_data)
        self.gt_depth = torch.from_numpy(gt_depth)
        self.gt_mask = torch.from_numpy(gt_alpha)


    def __getitem__(self, index: int):
        """
        Get the item at the given index
            param:
                - index: index of the item to get
            return:
                - the item at the given index
        """

        # Create the sample
        sample = {"itof_data": self.itof_data[index, ...], "gt_depth": self.gt_depth[index, ...], "gt_mask": self.gt_mask[index, ...]}

        # Apply the transformation to the data
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        """
        Get the length of the dataset
            return:
                - the length of the dataset
        """
        
        return self.itof_data.shape[0]