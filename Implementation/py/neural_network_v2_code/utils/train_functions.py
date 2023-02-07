import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from utils.utils import format_time
from torch.nn import BCEWithLogitsLoss
from torch.nn import MSELoss
from utils.utils import generate_fig


def train_fn(net: torch.nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn: torch.nn.Module, device: torch.device) -> float:
    epoch_loss = []

    # Set the network in training mode
    net.train()

    for sample in data_loader:
        # Get the input and the target
        itof_data = sample[0].to(device)
        gt_depth = sample[1].to(device)
        gt_mask = sample[2].to(device)

        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass
        output = net(itof_data)
        depth = output[:, 0, ...]
        mask = output[:, 1, ...]

        # Compute the loss
        #loss = loss_fn(output, gt_depth)
        depth_loss = MSELoss()(depth, gt_depth)
        mask_loss = BCEWithLogitsLoss()(mask, gt_mask)
        loss = depth_loss + mask_loss
        

        # Backward pass
        #loss.backward()
        depth_loss.backward(retain_graph=True)
        mask_loss.backward(retain_graph=True)
        #loss.backward()

        # Update the weights
        optimizer.step()

        # Append the loss
        epoch_loss.append(loss.item())

    # Return the average loss over al the batches
    return float(np.mean(epoch_loss))


def val_fn(net: torch.nn.Module, data_loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> tuple:
    epoch_loss = []

    # Initialize variable that will ocntains the last gt_depth and the last predicted depth
    last_gt_depth = None
    last_depth = None

    # Set the network in evaluation mode
    net.eval()

    with torch.no_grad():
        for sample in data_loader:
            # Get the input and the target
            itof_data = sample[0].to(device)
            gt_depth = sample[1].to(device)
            gt_mask = sample[2].to(device)

            # Forward pass
            output = net(itof_data)
            depth = output[:, 0, ...]
            mask = output[:, 1, ...]

            # Compute the loss
            #loss = loss_fn(output, gt_depth)
            depth_loss = MSELoss()(depth, gt_depth)
            mask_loss = BCEWithLogitsLoss()(mask, gt_mask)
            loss = depth_loss + mask_loss

            # Append the loss
            epoch_loss.append(loss.item())

            # Update the last gt_depth and the last predicted depth
            last_gt_depth = gt_depth.to("cpu").numpy()[-1, ...]
            last_depth = output.to("cpu").numpy()[-1, ...]

    # Return the average loss over al the batches
    return float(np.mean(epoch_loss)), last_gt_depth, last_depth


def train(net: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, loss_fn: torch.nn.Module, device: torch.device, n_epochs: int, save_path: Path) -> tuple:
    # Initialize the tensorboard writer
    writer = SummaryWriter(comment=f"_{net.__class__.__name__}_LR_{optimizer.param_groups[0]['lr']}_BS_{train_loader.batch_size}")
    
    # Initialize the best loss
    best_loss = float("inf")

    # Initialize the variable that contains the overall time
    overall_time = 0

    # Initialize the lists for the losses
    train_loss_tot = []
    val_loss_tot = []

    for epoch in range(n_epochs):
        # Start the timer
        start_time = time.time()

        # Train the network
        train_loss = train_fn(net, train_loader, optimizer, loss_fn, device)

        # Validate the network
        val_loss, gt_depth, depth = val_fn(net, val_loader, loss_fn, device)

        # End the timer
        end_time = time.time()

        # Compute the ETA using the mean time per epoch
        overall_time += end_time - start_time
        mean_time = overall_time / (epoch + 1)
        eta = format_time(0, (n_epochs - epoch - 1) * mean_time)

        # Print the results
        print(f"Epoch: {epoch + 1}/{n_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Time for epoch: {format_time(start_time, end_time)} | ETA: {eta}")

        # Print to file
        with open(save_path.parent / "log.txt", "a") as f:
            f.write(f"Epoch: {epoch + 1}/{n_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Time for epoch: {format_time(start_time, time.time())} | ETA: {eta}\n")
        

        # Write the results to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Check if the validation loss is the best
        if val_loss < best_loss:
            # Save the model
            torch.save(net.state_dict(), save_path)

            # Update the best loss
            best_loss = val_loss

        # Update the learning rate
        update_lr(optimizer, epoch)

        # Append the losses
        train_loss_tot.append(train_loss)
        val_loss_tot.append(val_loss)

        # Add the images to tensorboard
        #writer.add_figure("Target", generate_fig(depth), epoch)
        #writer.add_figure("Prediction", gt_depth, epoch)

    # Close the tensorboard writer
    writer.flush()
    writer.close()

    return train_loss_tot, val_loss_tot


def update_lr(optimizer: Optimizer, epoch: int) -> None:
    # Update the learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * 0.9