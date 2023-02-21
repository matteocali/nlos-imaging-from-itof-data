import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import numpy as np
import time
import socket
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from utils.utils import format_time, generate_fig
from utils.EarlyStopping import EarlyStopping
from utils.utils import update_lr


def train_fn(net: torch.nn.Module, data_loader: DataLoader, optimizer: Optimizer, depth_loss_fn: torch.nn.Module, mask_loss_fn: torch.nn.Module, l: float, device: torch.device) -> tuple[float, float, float]:
    """
    Function to train the network on the training set
        param:
            - net: network to train
            - data_loader: data loader containing the training set
            - optimizer: optimizer used to update the weights
            - depth_loss_fn: loss function used to compute the loss over the depth
            - mask_loss_fn: loss function used to compute the loss over the mask
            - l: lambda parameter used to balance the two losses (l * depth_loss + (1 - l) * mask_loss)
            - device: device used to train the network
        return:
            - tuple containing the average loss, the average depth loss and the average mask loss
    """

    epoch_loss = []  # Initialize the list that will contain the loss for each batch

    # Set the network in training mode
    net.train()

    for batch in data_loader:
        # Get the input and the targetcc
        itof_data = batch["itof_data"].to(device)  # Extract the input itof data
        gt_depth = batch["gt_depth"].to(device)    # Extract the ground truth depth
        gt_mask = batch["gt_mask"].to(device)      # Extract the ground truth mask

        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass
        depth, mask = net(itof_data)

        # Detach the mask from the graph in order to avoid the (of depth * mask) gradient to be propagated over the mask branch
        mask_d = mask.detach()
        # Force the mask to assume only value 0 or 1
        mask_d = torch.where(mask > 0.5, 1, 0)
        # Compute the masked depth (create correlation bertween the depth and the mask)
        masked_depth = depth * mask_d

        # Compute the losses
        depth_loss = depth_loss_fn(masked_depth, gt_depth, gt_mask)  # Compute the loss over the depth
        mask_loss = mask_loss_fn(mask, gt_mask)                      # Compute the loss over the mask

        if l == 0.5:
            loss = depth_loss + mask_loss                            # Compute the total loss
        else:
            loss = l * depth_loss + (1 - l) * mask_loss              # Compute the total loss
        
        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Append the loss
        epoch_loss.append([loss.item(), depth_loss.item(), mask_loss.item()])

    # Return the average loss over al the batches
    return tuple([float(np.mean(loss)) for loss in zip(*epoch_loss)])


def val_fn(net: torch.nn.Module, data_loader: DataLoader, depth_loss_fn: torch.nn.Module, mask_loss_fn: torch.nn.Module, l: float, device: torch.device) -> tuple[tuple[float, float, float], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Function to validate the network on the validation set
        param:
            - net: network to validate
            - data_loader: data loader containing the validation set
            - depth_loss_fn: loss function used to compute the loss over the depth
            - mask_loss_fn: loss function used to compute the loss over the mask
            - l: lambda parameter used to balance the two losses (l * depth_loss + (1 - l) * mask_loss)
            - device: device used to validate the network
        return:
            - tuple containing the average loss, the average depth loss, the average mask loss, the last gt_depth and the last predicted depth
        """
    
    epoch_loss = []

    # Initialize the last gt_depth and the last_pred depth
    last_gt = tuple(np.empty((1, 1)))
    last_pred = tuple(np.empty((1, 1)))

    # Set the network in evaluation mode
    net.eval()

    with torch.no_grad():
        for batch in data_loader:
            # Get the input and the target
            itof_data = batch["itof_data"].to(device)  # Extract the input itof data
            gt_depth = batch["gt_depth"].to(device)    # Extract the ground truth depth
            gt_mask = batch["gt_mask"].to(device)      # Extract the ground truth mask

            # Forward pass
            depth, mask = net(itof_data)

            # Force the mask to assume only value 0 or 1
            mask_d = torch.where(mask > 0.5, 1, 0)
            # Compute the masked depth (create correlation bertween the depth and the mask)
            masked_depth = depth * mask_d

            # Compute the loss
            depth_loss = depth_loss_fn(masked_depth, gt_depth, gt_mask)  # Compute the loss over the depth
            mask_loss = mask_loss_fn(mask, gt_mask)                      # Compute the loss over the mask
            
            if l == 0.5:
                loss = depth_loss + mask_loss                            # Compute the total loss
            else:
                loss = l * depth_loss + (1 - l) * mask_loss              # Compute the total loss

            # Append the loss
            epoch_loss.append([loss.item(), depth_loss.item(), mask_loss.item()])

            # Update the last gt_depth and the last predicted depth
            last_gt = (gt_depth.to("cpu").numpy()[-1, ...], gt_mask.to("cpu").numpy()[-1, ...])
            last_pred = (masked_depth.to("cpu").numpy()[-1, ...], mask.to("cpu").numpy()[-1, ...])

    # Return the average loss over al the batches
    return tuple([float(np.mean(loss)) for loss in zip(*epoch_loss)]), last_gt, last_pred


def train(attempt_name: str, net: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, depth_loss_fn: torch.nn.Module, mask_loss_fn: torch.nn.Module, l: float, device: torch.device, n_epochs: int, save_path: Path) -> None:
    """
    Function to train the network
        param:
            - attempt_name: name of the attempt
            - net: network to train
            - train_loader: data loader containing the training set
            - val_loader: data loader containing the validation set
            - optimizer: optimizer used to update the weights
            - depth_loss_fn: loss function used to compute the loss over the depth
            - mask_loss_fn: loss function used to compute the loss over the mask
            - l: lambda parameter used to balance the two losses (l * depth_loss + (1 - l) * mask_loss)
            - device: device used to train the network
            - n_epochs: number of epochs to train the network
            - save_path: path where to save the model
    """
    
    # Initialize the tensorboard writer
    current_dir = os.path.dirname(__file__)
    current_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    hostname = socket.gethostname()
    writer = SummaryWriter(log_dir=f"{current_dir}/../tensorboard_logs/{current_date}_{hostname}_{attempt_name}_{net.__class__.__name__}_LR_{optimizer.param_groups[0]['lr']}_L_{l}")
    
    # Initialize the early stopping
    early_stopping = EarlyStopping(tollerance=100, min_delta=0.05, save_path=save_path, net=net)

    # Initialize the variable that contains the overall time
    overall_time = 0

    for epoch in range(n_epochs):
        # Start the timer
        start_time = time.time()

        # Train the network
        train_loss, train_loss_depth, train_loss_mask = train_fn(net, train_loader, optimizer, depth_loss_fn, mask_loss_fn, l, device)  # Execute the training function

        # Validate the network
        val_loss, gt, pred = val_fn(net, val_loader, depth_loss_fn, mask_loss_fn, l, device)  # Execute the validation function
        val_loss, val_loss_depth, val_loss_mask = val_loss                                    # Unpack the losses
        gt_depth, gt_mask = gt                                                                # Unpack the ground truth
        pred_depth, pred_mask = pred                                                          # Unpack the predictions

        # End the timer 
        end_time = time.time()

        # Compute the ETA using the mean time per epoch
        overall_time += end_time - start_time
        mean_time = overall_time / (epoch + 1)
        eta = format_time(0, (n_epochs - epoch - 1) * mean_time)

        # Check if the validation loss is the best one and save the model or stop the training
        best_loss, stop = early_stopping(val_loss)

        # Update the learning rate
        #update_lr(optimizer, epoch)

        # Print the results
        print(f"Epoch: {epoch + 1}/{n_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Best loss: {best_loss:.4f} | Time for epoch: {format_time(start_time, end_time)} | ETA: {eta}")

        # Print to file
        with open(save_path.parent.absolute() / f"{attempt_name}_log.txt", "a") as f:
            f.write(f"Epoch: {epoch + 1}/{n_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Best loss: {best_loss:.4f} | Time for epoch: {format_time(start_time, time.time())} | ETA: {eta}\n")

        # Write the results to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Depth loss/train", train_loss_depth, epoch)
        writer.add_scalar("Mask loss/train", train_loss_mask, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Depth loss/val", val_loss_depth, epoch)
        writer.add_scalar("Mask loss/val", val_loss_mask, epoch)

        # Add the images to tensorboard
        writer.add_figure("Depth/val", generate_fig((gt_depth.T, pred_depth.T), (np.min(gt_depth), np.max(gt_depth))), epoch)
        writer.add_figure("Mask/val", generate_fig((gt_mask.T, pred_mask.T), (0, 1)), epoch)

        # Stop the training if the early stopping is triggered
        if stop:
            print("Early stopping triggered")
            with open(save_path.parent.absolute() / f"{attempt_name}_log.txt", "a") as f:
                f.write("Early stopping triggered\n")
            break

    # Close the tensorboard writer
    writer.flush()
    writer.close()
