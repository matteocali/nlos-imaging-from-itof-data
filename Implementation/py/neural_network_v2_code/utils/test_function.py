import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path


def test(net: nn.Module, data_loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device, out_path: Path = None) -> tuple:  # type: ignore
    """
    Function to test the network
    param:
        - net: network to test
        - data_loader: data loader
        - loss_fn: loss function
        - device: device to use
    return:
        - average loss over all the batches
        - list of output depth maps
    """
    epoch_loss = []                      # Initialize the loss
    out_depth = np.empty((1, 320, 240))  # Initialize the output depth maps

    # Set the network in evaluation mode
    net.eval()

    with torch.no_grad():
        for sample in data_loader:
            # Get the input and the target
            itof_data = sample[0].to(device)
            gt_depth = sample[1].to(device)

            # Forward pass
            output = net(itof_data)

            # Compute the loss
            loss = loss_fn(output, gt_depth)

            # Append the loss
            epoch_loss.append(loss.item())

            # Append the output
            out_depth = np.concatenate(out_depth, axis=0)
    
    out_depth = np.delete(out_depth, 0, axis=0)  # Delete the first empty array

    # Save the output depth maps
    if out_path is not None:
        np.save(out_path, out_depth)

    # Return the average loss over al the batches
    return float(np.mean(epoch_loss)), out_depth
    