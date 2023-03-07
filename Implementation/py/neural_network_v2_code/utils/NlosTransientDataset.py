import torch
import numpy as np
import h5py as h5
import torchvision.transforms as T
import utils.CustomTransforms as CT
from tqdm import tqdm
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
from utils.utils import phi_func, hfov2focal, depth_cartesian2radial


class NlosTransientDataset(Dataset):
    """
    NLOS Transient Dataset class
    """

    def __init__(self, dts_folder: Path, csv_file: Path, transform=None):
        """
        Args:
            dts_folder (Path): path to the dataset folder
            csv_file (Path): path to the csv file containing the list of the images
            transform (callable, optional): Optional transform to be applied on a sample
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
            temp_data = h["data"][:]             # type: ignore
        [dim_x, dim_y, dim_t] = temp_data.shape  # type: ignore
        focal = hfov2focal(hdim=dim_x, hfov=60)  # Compute the focal length

        itof_data = np.zeros((num_images, dim_x, dim_y, nf), dtype=np.float32)       # Create the iToF data tensor
        gt_alpha = np.zeros((num_images, dim_x, dim_y), dtype=np.float32)            # Create the ground truth alpha map tensor
        gt_depth = np.zeros((num_images, dim_x, dim_y), dtype=np.float32)            # Create the ground truth depth map tensor in radial coordinates
        gt_depth_cartesian = np.zeros((num_images, dim_x, dim_y), dtype=np.float32)  # Create the ground truth depth map tensor in cartesian coordinates

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
            gt_depth_cartesian[count, ...] = depth_cartesian2radial(temp_gt_depth, focal)  # type: ignore
            gt_alpha[count, ...] = temp_gt_alpha

            # Increment the counter
            count += 1

        # Move the axis to have the channels as the second dimension instead of being the last one
        itof_data = np.moveaxis(itof_data, 3, 1)

        # Transform the data to torch tensors
        self.itof_data = torch.from_numpy(itof_data)
        self.gt_depth_cartesian = torch.from_numpy(gt_depth)
        self.gt_depth = torch.from_numpy(gt_depth_cartesian)
        self.gt_mask = torch.from_numpy(gt_alpha)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): index of the item to get
        Returns:
            sample (dict): the sample at the given index
        """

        # Create the sample
        sample = {"itof_data": self.itof_data[index, ...], "gt_depth": self.gt_depth[index, ...], "gt_depth_cartesian": self.gt_depth_cartesian, "gt_mask": self.gt_mask[index, ...]}

        # Apply the transformation to the data
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


    def __len__(self) -> int:
        """
        Returns:
            length (int): the length of the dataset
        """
        
        return self.itof_data.shape[0]
    

    def get_bg_obj_ratio(self) -> float:
        """
        Returns:
            bg_obj_ratio (dict): the ratio of the number of background pixels over the number of object pixels
        """

        # Compute the number of background and object pixels
        bg_pixels = np.sum(self.gt_mask.numpy() == 0)
        obj_pixels = np.sum(self.gt_mask.numpy() == 1)

        # Compute the ratio
        bg_obj_ratio = bg_pixels / obj_pixels

        return bg_obj_ratio

    
    def augment_dts(self, batch_size: int) -> None:
        """
        Augment the dataset by applying: 
            - random rotations
            - random translations
            - horizzontal flip
            - vertical_flip
            - random noise
        to a random group of elements (of size batch size) sampled at random.

        params:
            batch_size: the size of the batch to sample
        returns:
            the augmented dataset
        """

        # Define the transforms to apply to the dataset
        transforms = {
            "random rotate": CT.RandomRotation(degrees=180, interpolation=T.InterpolationMode.NEAREST, fill=float("inf")),
            "random translate": CT.RandomAffine(degrees=0, translate=(0.2, 0.2), interpolation=T.InterpolationMode.NEAREST, fill=float("inf")),
            "random hflip": T.RandomHorizontalFlip(p=1.0),
            "random vflip": T.RandomVerticalFlip(p=1.0),
            "random noise": CT.AddGaussianNoise(mean=0.0, std=1.0)
        }
        
        # Sample a random batch of elements for each transform
        indices = [np.random.choice(self.itof_data.shape[0], batch_size, replace=False) for _ in range(len(transforms.keys()))]

        # Apply the transforms to the dataset
        for i, (key, transform) in tqdm(enumerate(transforms.items()), desc="Applying transforms", total=len(transforms.keys())):
            for index in tqdm(indices[i], desc=f"Applying {key}", total=batch_size, leave=False):
                # Extract the sample
                itof_data = self.itof_data[index, ...].unsqueeze(0)
                gt_depth_cartesian = self.gt_depth_cartesian[index, ...].unsqueeze(0)
                gt_depth = self.gt_depth[index, ...].unsqueeze(0)
                gt_mask = self.gt_mask[index, ...].unsqueeze(0)

                # Concatenate all the data before applying the transform in order to apply the exact same traasformation to all the data
                data = torch.cat((itof_data, gt_depth_cartesian.unsqueeze(0), gt_depth.unsqueeze(0), gt_mask.unsqueeze(0)), dim=1)

                # Apply the transform
                data = transform(data)

                # Extract the data from the transformed tensor
                itof_data = data[0, 0:6, ...].unsqueeze(0)
                if key != "random noise":  # The noise transform does not change the ground truth
                    gt_depth_cartesian = data[0, 6, ...].unsqueeze(0)
                    gt_depth = data[0, 7, ...].unsqueeze(0)
                    gt_mask = data[0, 8, ...].unsqueeze(0)

                # Update the dataset
                self.itof_data = torch.cat((self.itof_data, itof_data), dim=0)
                self.gt_depth_cartesian = torch.cat((self.gt_depth_cartesian, gt_depth_cartesian), dim=0)
                self.gt_depth = torch.cat((self.gt_depth, gt_depth), dim=0)
                self.gt_mask = torch.cat((self.gt_mask, gt_mask), dim=0)
