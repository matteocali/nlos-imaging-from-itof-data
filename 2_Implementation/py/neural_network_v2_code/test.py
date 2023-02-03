import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
from utils.utils import phi_func
import h5py as h5
from pathlib import Path
from tqdm import tqdm
import time


dts_folder = Path("/media/matteocali/shared_data/Matteo/NLoS imaging using iToF/mirror_dts/fixed_camera_diffuse_wall")
csv_file = Path("neural_network_v2_code/test_dts_csv_split/test_dts_test.csv")


st = time.time()                                                 # Start the timer
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

fi = time.time()
minutes, seconds = divmod(fi - st, 60)
hours, minutes = divmod(minutes, 60)
print("The overall computation time for the dataset is %d:%02d:%02d" % (hours, minutes, seconds))

self.data = data
self.frequencies = frequencies
