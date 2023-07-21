import torch
import numpy as np
import pickle
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from utils.utils import mean_intersection_over_union as miou
from utils.utils import save_test_plots_itof, depth_radial2cartesian, hfov2focal, plt_loss_hists


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

    epoch_loss = []  # Initialize the loss
    iou_loss = []    # Initialize the IoU loss
    out_dict = {
        "pred": {
            "depth": np.empty((len(data_loader.dataset), 320, 240)),   # type: ignore
            "itof": np.empty((len(data_loader.dataset), 2, 320, 240))  # type: ignore
            }, 
        "gt": {
            "depth": np.empty((len(data_loader.dataset), 320, 240)),   # type: ignore
            "itof": np.empty((len(data_loader.dataset), 2, 320, 240))  # type: ignore
            }
        }  # Initialize the output dictionary

    # Set the network in evaluation mode
    net.eval()

    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="Testing", total=len(data_loader)):
            # Get the input and the target
            itof_data = sample["itof_data"].to(device)  # Extract the input itof data
            gt_itof = sample["gt_itof"].to(device)      # Extract the ground truth itof data
            gt_depth = sample["gt_depth"].to(device)    # Extract the ground truth depth

            # Forward pass
            itof, depth, _ = net(itof_data)

            # Change the background value if needed
            if bg != 0:
                itof = torch.where(itof == bg, 0, itof)
                gt_itof = torch.where(gt_itof == bg, 0, gt_itof)
            
            # Compute the losses
            itof_loss_real = loss_fn(itof.squeeze(0)[0, ...], gt_itof.squeeze(0)[0, ...])  # Compute the loss over the itof data (real)
            itof_loss_imag = loss_fn(itof.squeeze(0)[1, ...], gt_itof.squeeze(0)[1, ...])  # Compute the loss over the itof data (imaginary)
            depth_loss = loss_fn(depth, gt_depth)                                          # Compute the loss over the depth

            # Compute the mean intersection over union on the depth
            iou = miou(depth.to(device), gt_depth.to(device), 0).item()  # type: ignore

            # Update the loss list
            epoch_loss.append(depth_loss.to(device).item())
            iou_loss.append(iou)

            # Remove the batch dimension and convert the data to numpy
            n_itof = itof.to("cpu").squeeze(0).numpy()
            n_gt_itof = gt_itof.to("cpu").squeeze(0).numpy()
            n_depth = depth.to("cpu").squeeze(0).numpy()  # type: ignore
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
                plots_dir,
                iou)

            # Save the output
            out_dict["pred"]["depth"][i, ...] = n_depth
            out_dict["pred"]["itof"][i, ...] = n_itof
            out_dict["gt"]["depth"][i, ...] = n_gt_depth
            out_dict["gt"]["itof"][i, ...] = n_gt_itof

    # Save the overall depth loss
    mae_mean_loss = np.round(np.mean(epoch_loss), 4)
    mae_min_loss = np.round(np.min(epoch_loss), 4)
    mae_max_loss = np.round(np.max(epoch_loss), 4)
    iou_mean_loss = np.round(np.mean(iou_loss), 4)
    iou_min_loss = np.round(np.min(iou_loss), 4)
    iou_max_loss = np.round(np.max(iou_loss), 4)

    # Compute the accuracy
    accuracies = np.mean((1 - np.array(epoch_loss), np.array(iou_loss)), axis=0)
    accuracy_std = np.round(np.std(accuracies), 4)  # Compute the std
    accuracy = np.round(np.mean(accuracies) * 100, 1)

    # Compute the overall loss (mae_loss + 1 - iou_loss)
    overall_losses = np.mean((np.array(epoch_loss), 1 - np.array(iou_loss)), axis=0)
    overall_losses_std = np.round(np.std(overall_losses), 4)  # Compute the std
    overall_loss = np.round(np.mean(overall_losses), 4)

    # Plot the histograms of the loss and the accuracy
    plt_loss_hists(overall_losses, accuracies, plots_dir)  # type: ignore

    with open(out_path / "loss.txt", "w") as f:
        f.write(f"Overall accuracy: {accuracy}%\n")
        f.write(f"Overall accuracy std: {accuracy_std}\n")
        f.write(f"Overall loss: {overall_loss}\n")
        f.write(f"Overall loss std: {overall_losses_std}\n\n")
        f.write(f"Mean Absolute Error summary:\n")
        f.write(f"   - Overall depth loss (MAE): {mae_mean_loss}\n")
        f.write(f"   - Min depth loss: {mae_min_loss}\n")
        f.write(f"   - Max depth loss: {mae_max_loss}\n\n")
        f.write(f"Intersection over Union summary:\n")
        f.write(f"   - Overall IoU: {iou_mean_loss}\n")
        f.write(f"   - Min IoU: {iou_min_loss}\n")
        f.write(f"   - Max IoU: {iou_max_loss}")

    # Save the output npy
    with open(out_path / "results.pkl", 'wb') as f:
        pickle.dump(out_dict, f)
    