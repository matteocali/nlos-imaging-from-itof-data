import torch
import numpy as np
import h5py as h5
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
from . import CustomTransforms as CT
from ..utils import phi_func, hfov2focal, depth_cartesian2radial, depth2itof


class NlosTransientDatasetItofGt(Dataset):
    """
    NLOS Transient Dataset class
    Each element contains:
        - the raw iToF data (nf, dim_x, dim_y) (nf = number of frequencies) (dim_x, dim_y = image size)
        - the ground truth iToF data (dim_x, dim_y)
        - the ground truth depth map in radial coordinates (dim_x, dim_y)
    """

    def __init__(
        self,
        dts_folder: Path,
        csv_file: Path,
        frequencies: np.ndarray = np.array((20e06, 50e06, 60e06), dtype=np.float32),
        transform=None,
    ):
        """
        Args:
            dts_folder (Path): path to the dataset folder
            csv_file (Path): path to the csv file containing the list of the images
            transform (callable, optional): Optional transform to be applied on a sample
        """

        # Set the transform attribute
        self.transform = transform

        phi = phi_func(frequencies)  # Compute the phi matrix (iToF data)
        nf = phi.shape[0]  # Extract the number of frequencies

        # Load the csv files
        images = []
        with open(csv_file, "r") as f:
            images += f.readlines()
        images = [dts_folder / (image[:-1]) for image in images]

        # Load the first image to get the size of the images
        num_images = len(images)
        with h5.File(images[0], "r") as h:
            temp_data = h["data"][:]  # type: ignore
        [dim_x, dim_y, dim_t] = temp_data.shape  # type: ignore
        focal = hfov2focal(hdim=dim_x, hfov=60)  # Compute the focal length

        itof_data = np.zeros(
            (num_images, dim_x, dim_y, nf), dtype=np.float32
        )  # Create the iToF data tensor
        gt_itof = np.zeros(
            (num_images, 2, dim_x, dim_y), dtype=np.float32
        )  # Create the ground truth iToF data tensor
        gt_depth = np.zeros(
            (num_images, dim_x, dim_y), dtype=np.float32
        )  # Create the ground truth depth map tensor in radial coordinates

        names = []
        count = 0
        for image in tqdm(images, desc="Loading images", total=num_images):
            file_name = str(image).split("/")[-1][:-3]
            names.append(file_name)

            # Load the transient data
            with h5.File(image, "r") as h:
                temp_data = h["data"][:]  # Load the transient data  # type: ignore
                temp_gt_depth = h["depth_map"][
                    :
                ]  # Load the ground truth depth data  # type: ignore
                temp_gt_mask = h["alpha_map"][
                    :
                ]  # Load the ground truth mask data  # type: ignore

            temp_lin = np.reshape(
                temp_data, (dim_x * dim_y, dim_t)
            )  # Reshape the transient data  # type: ignore

            # Computation with the direct component
            v = np.matmul(temp_lin, np.transpose(phi))
            v = np.reshape(v, (dim_x, dim_y, phi.shape[0]))
            itof_data[count, ...] = v

            # Add the gt depth and alpha map
            gt_itof[count, ...] = depth2itof(temp_gt_depth, 20e06, ampl=temp_gt_mask)  # type: ignore
            gt_depth[count, ...] = depth_cartesian2radial(temp_gt_depth, focal)  # type: ignore

            # Increment the counter
            count += 1

        # Move the axis to have the channels as the second dimension instead of being the last one
        itof_data = np.moveaxis(itof_data, 3, 1)

        # Transform the data to torch tensors
        self.itof_data = torch.from_numpy(itof_data)
        self.gt_itof = torch.from_numpy(gt_itof)
        self.gt_depth = torch.from_numpy(gt_depth)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): index of the item to get
        Returns:
            sample (dict): the sample at the given index
        """

        # Create the sample
        sample = {
            "itof_data": self.itof_data[index, ...],
            "gt_itof": self.gt_itof[index, ...],
            "gt_depth": self.gt_depth[index, ...],
        }

        # Apply the transformation to the data
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        """
        Returns:
            length (int): the length of the datasetuando vuoi
        """

        return self.itof_data.shape[0]

    def get_bg_obj_ratio(self) -> float:
        """
        Returns:
            bg_obj_ratio (dict): the ratio of the number of background pixels over the number of object pixels
        """

        # Compute the number of background and object pixels
        bg_pixels = np.sum(self.gt_depth.numpy() == 0)
        obj_pixels = np.sum(self.gt_depth.numpy() != 0)

        # Compute the ratio
        bg_obj_ratio = bg_pixels / obj_pixels

        return bg_obj_ratio

    def augment_dts(self, batch_size: int, gaussian: bool = True) -> None:
        """
        Augment the dataset by applying:
            - random rotations
            - random translations
            - horizzontal flip
            - vertical flip
            - gaussian random noise
        to a random group of elements (of size batch size) sampled at random.\n\n

        Params:
            batch_size (int): the size of the batch to sample
            gaussian (bool): if True, gaussian noise will be one off the augmentation, otherwise it will be ignored
        """

        # Define the transforms to apply to the dataset
        transforms = {
            "random rotate": CT.RandomRotation(
                degrees=180,
                interpolation=T.InterpolationMode.NEAREST,
                fill=float("inf"),
            ),
            "random translate": CT.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                interpolation=T.InterpolationMode.NEAREST,
                fill=float("inf"),
            ),
            "random translate rotate": CT.RandomAffine(
                degrees=180,
                translate=(0.1, 0.1),
                interpolation=T.InterpolationMode.NEAREST,
                fill=float("inf"),
            ),  # "random translate and rotate
            "random hflip": T.RandomHorizontalFlip(p=1.0),
            "random vflip": T.RandomVerticalFlip(p=1.0),
            "random hflip vflip": T.Compose(
                [T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0)]
            ),  # "random hflip and vflip
            "random noise": CT.AddGaussianNoise(mean=0.0, std=1.0),
        }
        if not gaussian:
            transforms.pop("random noise")

        # Sample a random batch of elements for each transform
        indices = [
            np.random.choice(self.itof_data.shape[0], batch_size, replace=False)
            for _ in range(len(transforms.keys()))
        ]

        # Define the tmp tensors that will contains the transformed data
        tmp_itof_data = torch.zeros(
            (
                batch_size * len(transforms.keys()),
                self.itof_data.shape[1],
                self.itof_data.shape[2],
                self.itof_data.shape[3],
            ),
            dtype=torch.float32,
        )
        tmp_gt_itof = torch.zeros(
            (
                batch_size * len(transforms.keys()),
                self.gt_itof.shape[1],
                self.gt_itof.shape[2],
                self.gt_itof.shape[3],
            ),
            dtype=torch.float32,
        )
        tmp_gt_depth = torch.zeros(
            (
                batch_size * len(transforms.keys()),
                self.gt_depth.shape[1],
                self.gt_depth.shape[2],
            ),
            dtype=torch.float32,
        )

        # Apply the transforms to the dataset
        for i, (key, transform) in tqdm(
            enumerate(transforms.items()),
            desc="Applying transforms",
            total=len(transforms.keys()),
        ):
            for index in tqdm(
                indices[i], desc=f"Applying {key}", total=batch_size, leave=False
            ):
                # Extract the sample
                itof_data = self.itof_data[index, ...].unsqueeze(0)
                gt_itof = self.gt_itof[index, ...].unsqueeze(0)
                gt_depth = self.gt_depth[index, ...].unsqueeze(0)

                # Compute the total number of pixels composing the object
                gt_obj_pixels = torch.sum(gt_depth != 0)

                # Check the number of frequencies used
                n_freqs = itof_data.shape[1]

                # Concatenate all the data before applying the transform in order to apply the exact same traasformation to all the data
                data = torch.cat((itof_data, gt_itof, gt_depth.unsqueeze(0)), dim=1)

                # Apply the transform
                data = transform(data)

                # Extract the data from the transformed tensor
                itof_data = data[0, :n_freqs, ...].unsqueeze(0)
                if (
                    key != "random noise"
                ):  # The noise transform does not change the ground truth
                    gt_itof = data[0, n_freqs : n_freqs + 2, ...].unsqueeze(0)
                    gt_depth = data[0, n_freqs + 2, ...].unsqueeze(0)

                # Check if after the transformation the object is still fully in frame and update the dataset
                obj_pixels = torch.sum(
                    gt_depth != 0
                )  # Compute the number of pixels composing the object
                if (
                    abs(gt_obj_pixels - obj_pixels) < 2
                ):  # If the object is fully in frame, save the sample
                    # Update the tmp dataset
                    pos = i * batch_size + index
                    tmp_itof_data[pos, ...] = itof_data
                    tmp_gt_itof[pos, ...] = gt_itof
                    tmp_gt_depth[pos, ...] = gt_depth

        # Update the dataset
        self.itof_data = torch.cat((self.itof_data, tmp_itof_data), dim=0)
        self.gt_itof = torch.cat((self.gt_itof, tmp_gt_itof), dim=0)
        self.gt_depth = torch.cat((self.gt_depth, tmp_gt_depth), dim=0)

    def apply_noise(self, mean: float, std: float) -> None:
        """
        This function will take the whole dataset and apply gaussian noise to it.\n
        Params:
            mean (float): the mean of the gaussian distribution
            std (float): the standard deviation of the gaussian distribution
        """

        # Define the transformation function
        transform = CT.AddGaussianNoise(mean=mean, std=std)

        # Define the tmp tensors that will contains the transformed data
        itof_data = torch.empty(0, dtype=torch.float32)

        # Apply the transforms to the dataset
        for index in tqdm(
            range(self.itof_data.shape[0]), desc=f"Applying noise", leave=True
        ):
            # Extract the sample
            itof_data = self.itof_data[index, ...].unsqueeze(0)

            # Apply the transform
            itof_data = transform(itof_data).unsqueeze(0)

            # Update the dataset
            self.itof_data[index, ...] = itof_data


class NlosTransientDatasetItofReal(Dataset):
    """
    NLOS Transient Dataset class for real data
    Each element contains:
        - the raw iToF data (nf, dim_x, dim_y) (nf = number of frequencies) (dim_x, dim_y = image size)
        - the ground truth iToF data (dim_x, dim_y)
        - the ground truth depth map in radial coordinates (dim_x, dim_y)
    """

    def __init__(
        self,
        dts_folder: Path,
        frequencies: np.ndarray = np.array((20e06, 50e06, 60e06), dtype=np.float32),
        transform=None,
    ):
        """
        Args:
            dts_folder (Path): path to the dataset folder
            transform (callable, optional): Optional transform to be applied on a sample
        """

        # Set the transform attribute
        self.transform = transform

        phi = phi_func(frequencies)  # Compute the phi matrix (iToF data)
        nf = phi.shape[0]  # Extract the number of frequencies

        # Load the csv files
        images = sorted(list(dts_folder.glob("*.h5")))

        # Load the first image to get the size of the images
        num_images = len(images)
        with h5.File(images[0], "r") as h:
            temp_data = h["itof_data"][:]  # type: ignore
        [_, dim_x, dim_y] = temp_data.shape  # type: ignore

        itof_data = np.zeros(
            (num_images, nf, dim_x, dim_y), dtype=np.float32
        )  # Create the iToF data tensor
        gt_itof = np.zeros(
            (num_images, 2, dim_x, dim_y), dtype=np.float32
        )  # Create the ground truth iToF data tensor
        gt_depth = np.zeros(
            (num_images, dim_x, dim_y), dtype=np.float32
        )  # Create the ground truth depth map tensor in radial coordinates

        count = 0
        for image in tqdm(images, desc="Loading images", total=num_images):
            # Load the transient data
            with h5.File(image, "r") as h:
                temp_itof_data = h["itof_data"][:]  # Load the itof data  # type: ignore
                temp_gt_depth = h["depth_gt"][
                    :
                ]  # Load the ground truth depth data  # type: ignore
                temp_gt_itof = h["itof_gt"][
                    :
                ]  # Load the ground truth mask data  # type: ignore

            # Add the itof data
            temp_itof_data[:, :, 239] = temp_itof_data[
                :, :, 238
            ]  # Fill the last row copying the second to last one # type: ignore
            temp_itof_data = np.flip(
                temp_itof_data, axis=0
            )  # Rotate the data 180 degrees
            temp_itof_data = np.flip(
                temp_itof_data, axis=1
            )  # using two consecutive flips
            itof_data[count, ...] = temp_itof_data

            # Add the gt depth and alpha map
            gt_itof[count, ...] = temp_gt_itof
            gt_depth[count, ...] = temp_gt_depth

            # Increment the counter
            count += 1

        # Transform the data to torch tensors
        self.itof_data = torch.from_numpy(itof_data)
        self.gt_itof = torch.from_numpy(gt_itof)
        self.gt_depth = torch.from_numpy(gt_depth)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): index of the item to get
        Returns:
            sample (dict): the sample at the given index
        """

        # Create the sample
        sample = {
            "itof_data": self.itof_data[index, ...],
            "gt_itof": self.gt_itof[index, ...],
            "gt_depth": self.gt_depth[index, ...],
        }

        # Apply the transformation to the data
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        """
        Returns:
            length (int): the length of the datasetuando vuoi
        """

        return self.itof_data.shape[0]

    def get_bg_obj_ratio(self) -> float:
        """
        Returns:
            bg_obj_ratio (dict): the ratio of the number of background pixels over the number of object pixels
        """

        # Compute the number of background and object pixels
        bg_pixels = np.sum(self.gt_depth.numpy() == 0)
        obj_pixels = np.sum(self.gt_depth.numpy() != 0)

        # Compute the ratio
        bg_obj_ratio = bg_pixels / obj_pixels

        return bg_obj_ratio
