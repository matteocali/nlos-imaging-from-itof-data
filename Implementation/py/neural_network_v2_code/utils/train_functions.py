import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from pathlib import Path


def train_fn(net: torch.nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn: torch.nn.Module, device: torch.device) -> float:
    epoch_loss = []

    # Set the network in training mode
    net.train()

    for sample in data_loader:
        # Get the input and the target
        itof_data = sample[0].to(device)
        gt_depth = sample[1].to(device)

        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass
        output = net(itof_data)

        # Compute the loss
        loss = loss_fn(output, gt_depth)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Append the loss
        epoch_loss.append(loss.item())

    # Return the average loss over al the batches
    return float(np.mean(epoch_loss))


def val_fn(net: torch.nn.Module, data_loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> float:
    epoch_loss = []

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

    # Return the average loss over al the batches
    return float(np.mean(epoch_loss))


def train(net: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, loss_fn: torch.nn.Module, device: torch.device, n_epochs: int, save_path: Path) -> tuple:
    # Initialize the best loss
    best_loss = float("inf")

    # Initialize the lists for the losses
    train_loss_tot = []
    val_loss_tot = []

    for epoch in range(n_epochs):
        # Train the network
        train_loss = train_fn(net, train_loader, optimizer, loss_fn, device)

        # Validate the network
        val_loss = val_fn(net, val_loader, loss_fn, device)

        # Print the results
        print(f"Epoch: {epoch + 1}/{n_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

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

    return train_loss_tot, val_loss_tot


def update_lr(optimizer: Optimizer, epoch: int) -> None:
    # Update the learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * 0.9