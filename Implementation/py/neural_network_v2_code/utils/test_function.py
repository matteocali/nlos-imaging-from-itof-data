import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from utils.utils import generate_fig


def test(net: nn.Module, data_loader: DataLoader, depth_loss_fn: torch.nn.Module, mask_loss_fn: torch.nn.Module, l: float, device: torch.device, out_path: Path = None) -> tuple:  # type: ignore
    """
    Function to test the network
    param:
        - net: network to test
        - data_loader: data loader
        - depth_loss_fn: loss function used to compute the loss over the depth
        - mask_loss_fn: loss function used to compute the loss over the mask
        - l: lambda parameter used to balance the two losses (l * depth_loss + (1 - l) * mask_loss)
        - device: device to use
    return:
        - average loss over all the batches
        - numpy array containing the output depth and mask
    """

    epoch_loss = []                                        # Initialize the loss
    out = np.empty((1, 2, 320, 240), dtype=np.float32)     # Initialize the output depth maps

    # Set the network in evaluation mode
    net.eval()

    with torch.no_grad():
        for sample in data_loader:
            # Get the input and the target
            itof_data = sample["itof_data"].to(device)  # Extract the input itof data
            gt_depth = sample["gt_depth"].to(device)    # Extract the ground truth depth
            gt_mask = sample["gt_mask"].to(device)      # Extract the ground truth mask

            # Forward pass
            depth, mask = net(itof_data)

            # Compute the loss
            depth_loss = depth_loss_fn(depth, gt_depth)  # Compute the loss over the depth
            mask_loss = mask_loss_fn(mask, gt_mask)      # Compute the loss over the mask
            loss = l * depth_loss + (1 - l) * mask_loss  # Compute the total loss

            # Append the loss
            epoch_loss.append(loss.item())

            tmp_out = torch.cat((depth.to('cpu').numpy(), mask.to('cpu').numpy()), dim=1)  # Concatenate the depth and the mask

            # Append the output
            out = np.concatenate((out, tmp_out), axis=0)
    
    out = np.delete(out, 0, axis=0)  # Delete the first empty array

    # Save the output depth maps
    if out_path is not None:
        np.save(out_path, out)
        save_np_as_img(out, Path(str(out_path)[:-12] + "_images"))  # type: ignore

    # Return the average loss over al the batches
    return float(np.mean(epoch_loss)), out
    