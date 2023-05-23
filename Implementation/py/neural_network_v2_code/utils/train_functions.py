import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from torch.backends.cudnn import benchmark
from torch.cuda import amp
benchmark = True
import numpy as np
import time
import socket
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from utils.utils import format_time, generate_fig, itof2depth
from utils.CustomLosses import BinaryMeanIntersectionOverUnion as miou
from torchmetrics.functional.classification.jaccard import binary_jaccard_index
from utils.lovasz_losses import lovasz_hinge
from utils.EarlyStopping import EarlyStopping
from utils.SobelGradient import SobelGrad


def compute_loss_itof(itof: torch.Tensor, gt: torch.Tensor, depth: torch.Tensor, gt_depth: torch.Tensor, loss_fn: torch.nn.Module, add_loss: torch.nn.Module | SSIM | None = None, l: float = 0.5, l2: float = 0.0) -> torch.Tensor:
    """
    Function to compute the loss using itof data
        param:
            - itof: predicted itof
            - gt: ground truth itof
            - depth: predicted depth
            - gt_depth: ground truth depth
            - loss_fn: loss function to use
            - add_loss: additional loss function to use
            - l: lambda value for the additional loss
            - l2: lambda value for the IoU loss
        return:
            - final loss
    """

    # Compute the amplitude and the phase (prediction)
    # ampl = torch.sqrt(itof[:, 0, ...]**2 + itof[:, 1, ...]**2)
    # phase = torch.atan2(itof[:, 1, ...], itof[:, 0, ...])

    # Compute the amplitude and the phase (ground truth)
    # gt_ampl = torch.sqrt(gt[:, 0, ...]**2 + gt[:, 1, ...]**2)
    # gt_phase = torch.atan2(gt[:, 1, ...], gt[:, 0, ...])
    
    # Compute the main loss (Balanced MAE)
    loss_itof = loss_fn(itof, gt) # nn.MAELoss(reduction="none")

    # Compute the additional loss if any (SSIM or Gradient)
    if isinstance(add_loss, SSIM):
        # Compute the ssim loss
        second_loss = 1 - add_loss(depth.unsqueeze(0), gt_depth.unsqueeze(0))  # type: ignore
    elif isinstance(add_loss, torch.nn.Module):
        # The followig commented grad computation should be remove since is not precise as the Sobel operator
        """
        Compute the gradient of the prediction and gt
        grad_itof = torch.stack(
            (torch.gradient(clean_itof[:, 0, ...], dim=1, edge_order= 2)[0], 
            torch.gradient(clean_itof[:, 0, ...], dim=2, edge_order= 2)[0],
            torch.gradient(clean_itof[:, 1, ...], dim=1, edge_order= 2)[0], 
            torch.gradient(clean_itof[:, 1, ...], dim=2, edge_order= 2)[0]), 
            dim=1)
        grad_gt = torch.stack(
            (torch.gradient(gt[:, 0, ...], dim=1, edge_order= 2)[0], 
            torch.gradient(gt[:, 0, ...], dim=2, edge_order= 2)[0],
            torch.gradient(gt[:, 1, ...], dim=1, edge_order= 2)[0], 
            torch.gradient(gt[:, 1, ...], dim=2, edge_order= 2)[0]), 
            dim=1)
        """

        # Initialize the Sobel gradient operator with window size 7
        grad = SobelGrad(window_size=7)

        # Clean the itof data t oreduce noise
        clean_itof = torch.where(abs(itof.detach()) < 0.05, 0, itof.detach())

        # Initilaize the gradient tensors
        grad_itof = torch.empty((1, 2, itof.shape[2], itof.shape[3])).to(itof.device)
        grad_gt = torch.empty((1, 2, gt.shape[2], gt.shape[3])).to(itof.device)

        grad_itof[:, 0, ...], grad_itof[:, 1, ...] = grad(clean_itof)
        grad_gt[:, 0, ...], grad_gt[:, 1, ...] = grad(gt)

        # Compute the gradient loss (MSE or MAE)
        second_loss = add_loss(grad_itof, grad_gt)
    else:
        # If no additional loss is used, set the second loss to 0
        second_loss = 0

    # Compute the Intersection over Union loss (only if l2 is not 0)
    if l2 != 0:
        # iou_loss = miou()(depth, gt_depth, 0)  # type: ignore
        iou_loss = 1 - binary_jaccard_index(depth, torch.where(gt_depth > 0, 1, 0))
        # iou_loss = lovasz_hinge(depth, torch.where(gt_depth > 0, 1, 0)).item() # type: ignore
    else:
        iou_loss = 0

    # Compose the losses based on the lambda value
    return loss_itof + (l * second_loss) + (l2 * iou_loss)


def train_fn(net: torch.nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn: torch.nn.Module, add_loss: torch.nn.Module | SSIM | None, device: torch.device, l: float, l2: float, scaler: amp.GradScaler) -> float:  # type: ignore
    """
    Function to train the network on the training set
        param:
            - net: network to train
            - data_loader: data loader containing the training set
            - optimizer: optimizer used to update the weights
            - loss_fn: loss function to use
            - add_loss: additional loss function to use
            - device: device used to train the network
            - l: lambda value for the additional loss
            - l2: lambda value for the depth loss
            - scaler: scaler used for the mixed precision training
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
        itof, depth = net(itof_data)

        # Compute the loss
        with amp.autocast():  # type: ignore
            loss = compute_loss_itof(itof, gt_itof, depth, gt_depth, loss_fn, add_loss, l, l2)

        # Backward pass
        scaler.scale(loss).backward()  # type: ignore
        # loss.backward()

        # Update the weights
        scaler.step(optimizer)  # type: ignore
        scaler.update()  # type: ignore
        # optimizer.step()

        # Append the loss
        epoch_loss.append(loss.item())

    # Disable the training mode
    net.train(False)

    # Return the average loss over al the batches
    return float(np.mean(epoch_loss))


def val_fn(net: torch.nn.Module, data_loader: DataLoader, loss_fn: torch.nn.Module, add_loss: torch.nn.Module | SSIM | None, device: torch.device, l: float, l2: float, scaler: amp.GradScaler) -> tuple[float, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:  # type: ignore
    """
    Function to validate the network on the validation set
        param:
            - net: network to validate
            - data_loader: data loader containing the validation set
            - loss_fn: loss function to use
            - add_loss: additional loss function to use
            - device: device used to validate the network
            - l: lambda value for the additional loss
            - l2: lambda value for the depth loss
            - scaler: scaler used for the mixed precision training
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
            # Extract the ground truth depthS
            gt_depth = batch["gt_depth"].to(device)

            # Forward pass
            itof, depth = net(itof_data)

            # Compute the loss
            with amp.autocast():  # type: ignore
                loss = compute_loss_itof(itof, gt_itof, depth, gt_depth, loss_fn, add_loss, l, l2)
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


def train(attempt_name: str, net: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, loss_fn: torch.nn.Module, l: float, l2:float, add_loss: str, device: torch.device, n_epochs: int, save_path: Path) -> None:
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
            - l2: lambda value to balance the depth loss
            - add_loss: additional loss to use
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
        tollerance=150, min_delta=0.015, save_path=save_path, net=net)
    
    # Initialize the chosed additional loss
    if add_loss == "ssim":
        # SSIM loss
        add_loss_fn = SSIM().to(device)
    elif add_loss == "grad-mse":
        # MSE loss
        add_loss_fn = torch.nn.MSELoss(reduction="mean").to(device)
    elif add_loss == "grad-mae":
        # MAE loss
        add_loss_fn = torch.nn.L1Loss(reduction="mean").to(device)
    else:
        # No additional loss
        add_loss_fn = None

    # Initialize the variable that contains the overall time
    overall_time = 0

    # Define the scaler for the mixed precision training
    scaler = amp.GradScaler()  # type: ignore

    for epoch in range(n_epochs):
        # Start the timer
        start_time = time.time()

        # Train the network
        train_loss = train_fn(net, train_loader, optimizer, loss_fn, add_loss_fn, device, l, l2, scaler)

        # Validate the network
        # Execute the validation function
        val_loss, gt, pred = val_fn(net, val_loader, loss_fn, add_loss_fn, device, l, l2, scaler )
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

            # Save the final model
            # torch.save(net.state_dict(), str(save_path)[:-3] + "_FINAL.pt")

            break

    # Close the tensorboard writer
    writer.flush()
    writer.close()
