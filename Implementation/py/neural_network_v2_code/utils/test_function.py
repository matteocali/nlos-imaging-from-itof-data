import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from utils.utils import save_test_plots_itof, depth_radial2cartesian, hfov2focal, itof2depth


def test(net: nn.Module, data_loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device, out_path: Path, bg: int = 0) -> None:
    """
    Function to test the network
    param:
        - net: network to test
        - data_loader: data loader
        - loss_fn: loss function
        - device: device to use
        - out_path: path where to save the outputs
        - bg: background value
    """

    epoch_loss = []                                        # Initialize the loss
    out = np.empty((1, 3, 320, 240), dtype=np.float32)     # Initialize the output depth maps

    # Set the network in evaluation mode
    net.eval()

    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="Testing", total=len(data_loader)):
            # Get the input and the target
            itof_data = sample["itof_data"].to(device)  # Extract the input itof data
            gt_itof = sample["gt_itof"].to(device)      # Extract the ground truth itof data
            gt_depth = sample["gt_depth"].to(device)    # Extract the ground truth depth

            # Forward pass
            itof = net(itof_data)

            # Change the background value if needed
            if bg != 0:
                itof = torch.where(itof == bg, 0, itof)
                gt_itof = torch.where(gt_itof == bg, 0, gt_itof)

            # Compute the predicted radial depth
            depth = itof2depth(itof, 20e06).unsqueeze(0)  # type: ignore
            
            # Compute the losses
            itof_loss_real = loss_fn(itof.squeeze(0)[0, ...], gt_itof.squeeze(0)[0, ...])  # Compute the loss over the itof data (real)
            itof_loss_imag = loss_fn(itof.squeeze(0)[1, ...], gt_itof.squeeze(0)[1, ...])  # Compute the loss over the itof data (imaginary)
            depth_loss = loss_fn(depth, gt_depth)  # Compute the loss over the depth

            # Remove the batch dimension and convert the data to numpy
            n_itof = itof.to("cpu").squeeze(0).numpy()
            n_gt_itof = gt_itof.to("cpu").squeeze(0).numpy()
            n_depth = depth.to("cpu").squeeze(0).numpy()
            n_gt_depth = gt_depth.to("cpu").squeeze(0).numpy()

            # Convert the depth from radial to cartesian
            n_depth = depth_radial2cartesian(n_depth, hfov2focal(320, 60))
            n_gt_depth = depth_radial2cartesian(n_gt_depth, hfov2focal(320, 60))

            # Create the folder to save the plots
            plots_dir = Path(out_path / "plots")
            Path.mkdir(plots_dir, exist_ok=True)

            # Save the plots
            save_test_plots_itof(
                (n_gt_depth, n_depth),  # type: ignore
                (n_gt_itof, n_itof), 
                (depth_loss.item(), itof_loss_real.item(), itof_loss_imag.item()), 
                i,
                plots_dir)

            tmp_out = torch.cat((depth.unsqueeze(0), itof), dim=1)  # Concatenate the depth and the mask

            # Append the output
            out = np.concatenate((out, tmp_out.to("cpu").numpy()), axis=0)
    
    out = np.delete(out, 0, axis=0)  # Delete the first empty array

    # Save the output npy
    np.save(out_path / "results.npy", out)
    