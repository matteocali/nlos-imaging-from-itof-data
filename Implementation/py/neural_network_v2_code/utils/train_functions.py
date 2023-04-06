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
from utils.utils import itof2depth, update_lr
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM


def compute_loss_itof(itof: torch.Tensor, gt: torch.Tensor, loss_fn: torch.nn.Module, ssim_fn: SSIM, mse_loss: torch.nn.Module, l: float = 0.5) -> torch.Tensor:
    """
    Function to compute the loss using itof data
        param:
            - itof: predicted itof
            - gt: ground truth itof
            - loss_fn: loss function to use
            - ssim_loss: ssim loss function
            - mse_loss: mse loss function
            - l: lambda value
        return:
            - final loss
    """

    # Compute the amplitude and the phase (prediction)
    # ampl = torch.sqrt(itof[:, 0, ...]**2 + itof[:, 1, ...]**2)
    # phase = torch.atan2(itof[:, 1, ...], itof[:, 0, ...])

    # Compute the amplitude and the phase (ground truth)
    # gt_ampl = torch.sqrt(gt[:, 0, ...]**2 + gt[:, 1, ...]**2)
    # gt_phase = torch.atan2(gt[:, 1, ...], gt[:, 0, ...])

    # Compute the gradient of the prediction and gt
    grad_itof = torch.stack(
        (torch.abs(torch.gradient(itof[:, 0, ...], dim=1)[0]), 
         torch.abs(torch.gradient(itof[:, 0, ...], dim=2)[0]),
         torch.abs(torch.gradient(itof[:, 1, ...], dim=1)[0]), 
         torch.abs(torch.gradient(itof[:, 1, ...], dim=2)[0])), 
        dim=1)
    grad_gt = torch.stack(
        (torch.abs(torch.gradient(gt[:, 0, ...], dim=1)[0]), 
         torch.abs(torch.gradient(gt[:, 0, ...], dim=2)[0]),
         torch.abs(torch.gradient(gt[:, 1, ...], dim=1)[0]), 
         torch.abs(torch.gradient(gt[:, 1, ...], dim=2)[0])), 
        dim=1)

    # Compute the losses
    loss_itof = loss_fn(itof, gt)
    #loss_ssim = 1 - ssim_fn(itof, gt)
    loss_grad = mse_loss(grad_itof, grad_gt)
    # loss_phase = loss_fn(phase, gt_phase)

    # return loss_itof + loss_phase

    # Compute the final loss
    # loss = loss_itof + l * loss_ssim
    loss = loss_itof + l * loss_grad

    return loss


def compute_loss_depth(itof: torch.Tensor, gt: torch.Tensor, loss_fn: torch.nn.Module) -> torch.Tensor:
    """
    Function to compute the loss using itof data
        param:
            - itof: predicted itof
            - gt: ground truth depth
            - loss_fn: loss function to use
        return:
            - final loss
    """

    depth = itof2depth(itof, 20e06)  # type: ignore

    return loss_fn(depth, gt)


def train_fn(net: torch.nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn: torch.nn.Module, ssim_fn: SSIM, mse_loss: torch.nn.Module, device: torch.device, l: float) -> float:
    """
    Function to train the network on the training set
        param:
            - net: network to train
            - data_loader: data loader containing the training set
            - optimizer: optimizer used to update the weights
            - loss_fn: loss function to use
            - ssim_loss: ssim loss function
            - mse_loss: mse loss function
            - device: device used to train the network
            - l: lambda value
        return:
            - average loss
    """

    epoch_loss = []  # Initialize the list that will contain the loss for each batch

    # Set the network in training mode
    net.train(True)

    for batch in data_loader:
        # Get the input and the targetcc
        # Extract the input itof data
        itof_data = batch["itof_data"].to(device)  
        # Extract the ground truth itof data
        gt_itof = batch["gt_itof"].to(device)
        # Extract the ground truth depth
        gt_depth = batch["gt_depth"].to(device)

        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass
        itof = net(itof_data)

        # Compute the loss
        loss = compute_loss_itof(itof, gt_itof, loss_fn, ssim_fn, mse_loss, l)
        # loss = compute_loss_depth(itof, gt_depth, loss_fn)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Append the loss
        epoch_loss.append(loss.item())

    # Disable the training mode
    net.train(False)

    # Return the average loss over al the batches
    return float(np.mean(epoch_loss))


def val_fn(net: torch.nn.Module, data_loader: DataLoader, loss_fn: torch.nn.Module, ssim_fn: SSIM, mse_loss: torch.nn.Module, device: torch.device, l: float) -> tuple[float, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Function to validate the network on the validation set
        param:
            - net: network to validate
            - data_loader: data loader containing the validation set
            - loss_fn: loss function to use
            - ssim_loss: ssim loss function
            - mse_loss: mse loss function
            - device: device used to validate the network
            - l: lambda value
        return:
            - tuple containing the average loss, the last gt_itof and gt_depth and the last predicted itof
        """

    epoch_loss = []

    # Initialize the last_gt and the last_pred depth
    last_gt = tuple(np.empty((1, 1)))
    last_pred = tuple(np.empty((1, 1)))

    # Set the network in evaluation mode
    net.eval()

    with torch.no_grad():
        for batch in data_loader:
            # Get the input and the target
            # Extract the input itof data
            itof_data = batch["itof_data"].to(device)  
            # Extract the ground truth itof data
            gt_itof = batch["gt_itof"].to(device)
            # Extract the ground truth depth
            gt_depth = batch["gt_depth"].to(device)

            # Forward pass
            itof = net(itof_data)

            loss = compute_loss_itof(itof, gt_itof, loss_fn, ssim_fn, mse_loss, l)
            # loss = compute_loss_depth(itof, gt_depth, loss_fn)

            # Append the loss
            epoch_loss.append(loss.item())

            # Cleen iToF data to avoid noise in the depth reconstruction
            clean_itof = torch.where(abs(itof.detach()) < 0.05, 0, itof.detach())

            # Update the last gt_depth and the last predicted depth
            last_gt = (gt_itof.to("cpu").numpy()[-1, ...], gt_depth.to("cpu").numpy()[-1, ...])
            last_pred = (itof.to("cpu").numpy()[-1, ...], itof2depth(clean_itof.to("cpu").numpy()[-1, ...], 20e06))  # type: ignore
    
    # Return the average loss over al the batches
    return float(np.mean(epoch_loss)), last_gt, last_pred # type: ignore


def train(attempt_name: str, net: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, loss_fn: torch.nn.Module, l: float, device: torch.device, n_epochs: int, save_path: Path) -> None:
    """
    Function to train the network
        param:
            - attempt_name: name of the attempt
            - net: network to train
            - train_loader: data loader containing the training set
            - val_loader: data loader containing the validation set
            - optimizer: optimizer used to update the weights
            - loss_fn: loss function to use
            - l: lambda value to balance the mae loss and the ssim loss
            - device: device used to train the network
            - n_epochs: number of epochs to train the network
            - save_path: path where to save the model
    """

    # Initialize the tensorboard writer
    current_dir = os.path.dirname(__file__)
    current_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    hostname = socket.gethostname()
    writer = SummaryWriter(
        log_dir=f"{current_dir}/../tensorboard_logs/{current_date}_{hostname}_{attempt_name}_{net.__class__.__name__}_LR_{optimizer.param_groups[0]['lr']}_l_{l}")

    # Initialize the early stopping
    early_stopping = EarlyStopping(
        tollerance=100, min_delta=0.05, save_path=save_path, net=net)
    
    # Initialize the SSIM loss
    ssim_loss = SSIM().to(device)

    # Initialize the MSE loss
    mse_loss = torch.nn.MSELoss().to(device)

    # Initialize the variable that contains the overall time
    overall_time = 0

    for epoch in range(n_epochs):
        # Start the timer
        start_time = time.time()

        # Train the network
        train_loss = train_fn(net, train_loader, optimizer, loss_fn, ssim_loss, mse_loss, device, l)

        # Validate the network
        # Execute the validation function
        val_loss, gt, pred = val_fn(net, val_loader, loss_fn, ssim_loss, mse_loss, device, l)
        # Unpack the ground truth
        _, gt_depth = gt
        # Unpack the predictions
        _, pred_depth = pred

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
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Add the images to tensorboard
        writer.add_figure("Depth/val", generate_fig((gt_depth.T,
                          pred_depth.T), (np.min(gt_depth), np.max(gt_depth))), epoch)

        # Stop the training if the early stopping is triggered
        if stop:
            print("Early stopping triggered")
            with open(save_path.parent.absolute() / f"{attempt_name}_log.txt", "a") as f:
                f.write("Early stopping triggered\n")

            torch.save(net.state_dict(), str(save_path)[:-3] + "_FINAL.pt")

            break

    # Close the tensorboard writer
    writer.flush()
    writer.close()
