import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from utils.utils import save_test_plots, depth_radial2cartesian, hfov2focal


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
    out = np.empty((1, 2, 320, 240), dtype=np.float32)     # Initialize the output depth maps

    # Set the network in evaluation mode
    net.eval()

    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="Testing", total=len(data_loader)):
            # Get the input and the target
            itof_data = sample["itof_data"].to(device)                    # Extract the input itof data
            gt_depth_cartesian = sample["gt_depth_cartesian"].to(device)  # Extract the ground truth depth
            gt_depth = sample["gt_depth"].to(device)                      # Extract the ground truth depth
            gt_mask = sample["gt_mask"].to(device)                        # Extract the ground truth mask

            # Forward pass
            depth, mask = net(itof_data)

            # Force the mask to assume only value 0 or 1
            mask = torch.where(torch.sigmoid(mask) > 0.5, 1, 0)
            # Compute the masked depth (create correlation bertween the depth and the mask)
            depth = depth * mask

            # Change the background value if needed
            if bg != 0:
                depth = torch.where(depth == bg, 0, depth)
                gt_depth = torch.where(gt_depth == bg, 0, gt_depth)

            # Compute the loss
            depth_loss = loss_fn(depth, gt_depth)  # Compute the loss over the depth
            mask_loss = loss_fn(mask, gt_mask)     # Compute the loss over the mask

            # Extract the data
            t_depth = depth.unsqueeze(0).to("cpu")
            t_depth = depth_radial2cartesian(t_depth, hfov2focal(hdim=gt_depth.shape[1], hfov=60))  # Convert the depth from radial to cartesian coordinates
            depth = depth.squeeze(0).to("cpu").numpy()
            depth = depth_radial2cartesian(depth, hfov2focal(hdim=gt_depth.shape[1], hfov=60))      # Convert the depth from radial to cartesian coordinates
            gt_depth_cartesian = gt_depth_cartesian.squeeze(0).to("cpu").numpy()
            t_mask = mask.unsqueeze(0).to("cpu")
            mask = mask.squeeze(0).to("cpu").numpy()
            gt_mask = gt_mask.squeeze(0).to("cpu").numpy()

            # Create the folder to save the plots
            plots_dir = Path(out_path / "plots")
            Path.mkdir(plots_dir, exist_ok=True)

            # Save the plots
            save_test_plots(
                (gt_depth_cartesian, depth),  # type: ignore
                (gt_mask, mask), 
                (depth_loss.item(), mask_loss.item()), 
                i,
                plots_dir)

            tmp_out = torch.cat((t_depth, t_mask), dim=1)  # Concatenate the depth and the mask

            # Append the output
            out = np.concatenate((out, tmp_out), axis=0)
    
    out = np.delete(out, 0, axis=0)  # Delete the first empty array

    # Save the output npy
    np.save(out_path / "results.npy", out)
    #save_np_as_img(out, Path(str(out_path)[:-12] + "_images"))  # type: ignore
    