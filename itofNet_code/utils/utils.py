import numpy as np
import scipy.constants as const
import smtplib
import torch
import seaborn as sns
import tikzplotlib
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.optim import Optimizer
from torchmetrics.functional.classification.jaccard import binary_jaccard_index
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def format_time(s_time: float, f_time: float):
    """
    Function used to format the time in a human readable format
        param:
            - s_time: start time
            - f_time: finish time
        return:
            - string containing the time in a human readable format
    """

    minutes, seconds = divmod(f_time - s_time, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 24:
        days, hours = divmod(hours, 24)
        return "%d:%02d:%02d:%02d" % (days, hours, minutes, seconds)
    return "%d:%02d:%02d" % (hours, minutes, seconds)


def phi_func(freqs, dim_t=2000, exp_time=0.01):
    """
    Function used to generate the phi matrix
        param:
            - freqs: frequencies
            - dim_t: number of time steps
            - exp_time: exposure time
        return:
            - phi matrix
    """

    min_t = 0
    max_t = 2 * exp_time / const.c * dim_t
    step_t = (max_t - min_t) / dim_t
    times = np.arange(dim_t) * step_t  # type: ignore
    phi_arg = 2 * const.pi * np.matmul(freqs.reshape(-1, 1), times.reshape(1, -1))
    phi = np.concatenate([np.cos(phi_arg), np.sin(phi_arg)], axis=0)
    return phi


def row_subplot(fig, ax, data: tuple[np.ndarray, np.ndarray], titles: tuple[str, str], loss: float or None = None, iou: float | None = None, clim: bool = True) -> None:  # type: ignore
    """
    Function used to generate the row subplot
        param:
            - fig: figure
            - ax: axis
            - data: tuple containing the ground truth and the predicted data
            - title: tuple containing the title of the subplot
            - loss: loss value
            - iou: intersection over union
            - clim: if True set the colorbar limits based on the ground truth data
    """

    # Generate the plts for the depth
    for i in range(2):
        img = ax[i].matshow(data[i].T, cmap="jet")                             # Plot the sx plot # type: ignore
        if clim:
            img.set_clim(np.min(data[0]), np.max(data[0]))                     # Set the colorbar limits based on the ground truth data
        divider = make_axes_locatable(ax[i])                                   # Defien the colorbar axis
        cax = divider.append_axes("right", size="5%", pad=0.05)                # Set the colorbar location
        fig.colorbar(img, cax=cax, label="Depth [m]")                          # Plot the colorbar
        
        if i == 1 and loss is not None:                                        # If the plot is the predicted one and the loss is not None
            box_style = dict(boxstyle="round", fc="w", ec="black", alpha=0.9)  # Define the box style
            ax[i].text(20, 20, f"MAE: {round(loss, 3)}", 
                          ha='left', va='top', fontsize=11, 
                          fontfamily='monospace',
                          color='black', bbox=box_style)                       # Add the box to the plot # type: ignore
            if iou is not None:                                                # If the miou is not None
                ax[i].text(20, 40, f"IoU: {round(iou, 3)}",
                            ha='left', va='top', fontsize=11,
                            fontfamily='monospace',
                            color='black', bbox=box_style)                     # Add the box for the miou to the plot # type: ignore
        ax[i].set_title(titles[i])                                             # Set the title of the subplot
        ax[i].set_xlabel("X")                                                  # Set the x label of the subplot
        ax[i].set_ylabel("Y", rotation=0)                                      # Set the y label of the subplot and rotrate it


def save_test_plots(depth_data: tuple[np.ndarray, np.ndarray], mask_data: tuple[np.ndarray, np.ndarray], losses: tuple[float, float], index: int, path: Path):
    """
    Function used to save the test plots
        param:
            - depth_data: (gt_depth, predicted depth)
            - mask_data: (gt_mask, predicted mask)
            - losses: tuple containing the loss and the accuracy
            - index: index of the test sample
            - path: path where to save the plots
    """

    # Generate the plot
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))

    # Generate the plts for the depth
    titles = ("Grount truth depth", "Predicted depth")
    row_subplot(fig, ax[0], (depth_data[0], depth_data[1]), titles, losses[0])
    # Generate the plts for the mask
    titles = ("Grount truth mask", "Predicted mask")
    row_subplot(fig, ax[1], (mask_data[0], mask_data[1]), titles, losses[1])
    
    plt.tight_layout()
    plt.savefig(str(path / f"{index + 1}.svg"))
    plt.close()


def save_test_plots_itof(depth_data: tuple[np.ndarray, np.ndarray], itof_data: tuple[np.ndarray, np.ndarray], losses: tuple[float, float, float], index: int, path: Path, iou:float | None = None, tex: bool = False) -> None:
    """
    Function used to save the test plots
        param:
            - depth_data: (gt_depth, predicted depth)
            - itof_data: (gt_itof, predicted itof)
            - losses: tuple containing the loss and the accuracy
            - index: index of the test sample
            - path: path where to save the plots
            - iou: mean intersection over union
            - tex: if True save the plots in tex format
    """

    # Create the tex folder if needed
    if tex:
        tex_path = path / "tex"
        Path.mkdir(tex_path, exist_ok=True, parents=True)

    # Generate the plot
    fig, ax = plt.subplots(3, 2, figsize=(16, 16))

    # Force the clim
    clim = True

    # Generate the plts for the depth
    titles = ("Grount truth depth", "Predicted depth")
    row_subplot(fig, ax[0], (depth_data[0], depth_data[1]), titles, losses[0], iou, clim)
    # Generate the plts for the itof
    # Real iToF
    titles = ("Grount truth real iToF", "Predicted real iToF")
    row_subplot(fig, ax[1], (itof_data[0][0, ...], itof_data[1][0, ...]), titles, losses[1], clim)
    # Imaginary iToF
    titles = ("Grount truth imaginary iToF", "Predicted imaginary iToF")
    row_subplot(fig, ax[2], (itof_data[0][1, ...], itof_data[1][1, ...]), titles, losses[2], clim)
    
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(str(path / f"{index + 1}.svg"))
    if tex:
        tikzplotlib.save(str(tex_path / f"{index + 1}.tex"))  # type: ignore
    plt.close()
        

def generate_fig(data: tuple[np.ndarray, np.ndarray], c_range: tuple[float, float] = None):  # type: ignore
    """
    Function used to generate the figures to visualize the target and the prediction on tensorboard
        param:
            - data: tuple containing the target and the prediction
            - c_range: range of the colorbar
        return:
            - figure
    """
    
    # Set seaborn style
    sns.set_style()

    titles = ["Target", "Prediction"]
    fig, ax = plt.subplots(1, 2)
    for i in range(2):
        img_t = ax[i].matshow(data[i], cmap="jet")
        if range is not None:
            img_t.set_clim(c_range[0], c_range[1])
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mappable=img_t, cax=cax)
        ax[i].set_title(titles[i])
        ax[i].set_xlabel("Column pixel")
        ax[i].set_ylabel("Row pixel")
    plt.tight_layout()
    return fig


def plt_itof(itof: np.ndarray, path: Path) -> None:
    """
    Function used to plot the iToF
        param:
            - itof: iToF to plot
    """

    # Set seaborn style
    sns.set_style()

    # Generate the plot
    fig, ax = plt.subplots(3, 2, figsize=(16, 16))

    # Generate the plts for the itof at 20MHz
    titles = ("20MHz real", "20MHz imaginary")
    row_subplot(fig, ax[0], (itof[0, ...], itof[3, ...]), titles, clim=False)
    # Generate the plts for the itof at 50MHz
    titles = ("50MHz real", "50MHz imaginary")
    row_subplot(fig, ax[1], (itof[1, ...], itof[4, ...]), titles, clim=False)
    # Generate the plts for the itof at 60MHz
    titles = ("60MHz real", "60MHz imaginary")
    row_subplot(fig, ax[2], (itof[2, ...], itof[5, ...]), titles, clim=False)
    
    plt.tight_layout()
    plt.savefig(str(path))
    plt.close()    


def row_subplot_diff(fig, ax, data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], titles: tuple[str, str, str, str] | tuple[str, str, str, str, str]) -> None:  # type: ignore
    """
    Function used to generate the row subplot
        param:
            - fig: figure
            - ax: axis
            - data: tuple containing the ground truth and the predicted data
            - titles: tuple containing the title of the subplot
    """

    # Generate the plts for the depth

    row_length = len(data)

    if row_length == 4:
        ax[0].axis("off")
        r = range(1, 5)
        k = 1
    else:
        r = range(5)
        k = 0

    for i in r:
        img = ax[i].matshow(data[i - k].T, cmap="jet")           # Plot the sx plot # type: ignore
        img.set_clim(np.min(data[i - k]), np.max(data[i - k]))   # Set the colorbar limits based on the ground truth data
        divider = make_axes_locatable(ax[i])                     # Defien the colorbar axis
        cax = divider.append_axes("right", size="5%", pad=0.05)  # Set the colorbar location
        fig.colorbar(img, cax=cax)                               # Plot the colorbar
        ax[i].set_title(titles[i - k])                           # Set the title of the subplot
        ax[i].set_xlabel("x")                                    # Set the x label of the subplot
        ax[i].set_ylabel("y")                                    # Set the y label of the subplot


def plot_difference(depth: torch.Tensor | np.ndarray, itof: torch.Tensor | np.ndarray, empty: torch.Tensor | np.ndarray, index: int, path: Path) -> None:
    """
    Function used to save the test plots
        param:
            - depth: depth map
            - itof: itof data
            - empty: itof data of the empty scene (only white wall)
            - index: index of the test sample
            - path: path where to save the plots
    """

    # If necessary convert the data to numpy
    if isinstance(depth, torch.Tensor):
        depth = depth.to("cpu").numpy()
    if isinstance(itof, torch.Tensor):
        itof = itof.to("cpu").numpy()
    if isinstance(empty, torch.Tensor):
        empty = empty.to("cpu").numpy()

    # Compute the difference
    diff = itof - empty

    # Set seaborn style
    sns.set_style()

    # Generate the plot
    fig, ax = plt.subplots(3, 5, figsize=(32, 16))

    # Generate the plts for the 20MHz
    titles = ("iToF data at 20MHz (real)", "iToF data at 20MHz (imaginary)", "Difference with the empty scene 20MHz (real)", "Difference with the empty scene 20MHz (imaginary)")
    row_subplot_diff(fig, ax[0], (itof[0, ...], itof[1, ...], diff[0, ...], diff[1, ...]), titles)  # type: ignore
    # Generate the plts for the 50MHz
    titles = ("GT depth map", "iToF data at 50MHz (real)", "iToF data at 50MHz (imaginary)", "Difference with the empty scene 50MHz (real)", "Difference with the empty scene 50MHz (imaginary)")
    row_subplot_diff(fig, ax[1], (depth, itof[2, ...], itof[3, ...], diff[2, ...], diff[3, ...]), titles)  # type: ignore
    # Generate the plts for the 60MHz
    titles = ("iToF data at 60MHz (real)", "iToF data at 60MHz (imaginary)", "Difference with the empty scene 60MHz (real)", "Difference with the empty scene 60MHz (imaginary)")
    row_subplot_diff(fig, ax[2], (itof[4, ...], itof[5, ...], diff[4, ...], diff[5, ...]), titles)  # type: ignore
    
    plt.tight_layout()
    plt.savefig(str(path / f"{index + 1}.png"))
    plt.close()


def send_email(receiver_email: str, subject: str, body: str):
    """
    Function used to send an email
        param:
            - receiver_email: email address of the receiver
            - subject: subject of the email
            - body: body of the email
    """

    email = 'py.script.notifier@gmail.com'
    password = 'sxruxiufydfhknov'

    message = MIMEMultipart()
    message["To"] = receiver_email
    message["From"] = "Python Notifier"
    message["Subject"] = subject

    messageText = MIMEText(body, 'html')
    message.attach(messageText)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo('Gmail')
    server.starttls()
    server.login(email, password)
    server.sendmail(email, receiver_email, message.as_string())

    server.quit()


def update_lr(optimizer: Optimizer, epoch: int) -> None:
    """
    Function to update the learning rate
        param:
            - optimizer: optimizer used to update the weights
            - epoch: current epoch
    """

    if epoch == 10:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.1


def hfov2focal(hdim: int, hfov: float) -> float:
    """
    Function used to convert the horizontal field of view to the focal length
        param:
            - hdim: horizontal dimension of the image (pixels)
            - hfov: horizontal field of view (degrees)
        return:
            - focal length
    """

    return 0.5 * hdim / np.tan(0.5 * hfov * np.pi / 180)


def depth_cartesian2radial(depth: torch.Tensor or np.ndarray, focal: float) -> torch.Tensor or np.ndarray:
    """
    Function used to convert the depth map from cartesian to radial coordinates
        param:
            - depth: depth map in cartesian coordinates
            - focal: focal length of the camera
        return:
            - depth map in radial coordinates
    """

    if isinstance(depth, np.ndarray):
        env = np
    else:
        env = torch

    res_v = depth.shape[0]
    res_h = depth.shape[1]
    
    axis_v = env.linspace(-res_v/2 + 1/2, res_v/2 - 1/2, res_v)
    axis_h = env.linspace(-res_h/2 + 1/2, res_h/2 - 1/2, res_h)

    conversion_matrix = env.zeros((res_v, res_h))
    for i in range(res_v):
        for j in range(res_h):
            conversion_matrix[i, j] = 1 / env.sqrt(1 + (axis_v[i] / focal)**2 + (axis_h[j] / focal)**2)  # type: ignore

    return depth / conversion_matrix


def depth_radial2cartesian(depth: torch.Tensor or np.ndarray, focal: float) -> torch.Tensor or np.ndarray:
    """
    Function used to convert the depth map from radial to cartesian coordinates
        param:
            - depth: depth map in radial coordinates
            - focal: focal length of the camera
        return:
            - depth map in cartesian coordinates
    """

    if isinstance(depth, np.ndarray):
        env = np
    else:
        env = torch

    res_v = depth.shape[0]
    res_h = depth.shape[1]
    axis_v = env.linspace(-res_v/2 + 1/2, res_v/2 - 1/2, res_v)
    axis_h = env.linspace(-res_h/2 + 1/2, res_h/2 - 1/2, res_h)

    conversion_matrix = env.zeros((res_v, res_h))
    for i in range(res_v):
        for j in range(res_h):
            conversion_matrix[i, j] = env.sqrt(1 + (axis_v[i] / focal)**2 + (axis_h[j] / focal)**2)  # type: ignore

    return depth * conversion_matrix


def normalize(data: np.ndarray or torch.Tensor, bounds: dict[str, dict[str, float]]) -> np.ndarray or torch.Tensor:
    """
    Function used to normalize the data
        param:
            - data: data to be normalized
            - bounds: bounds of the data
                - actual: actual bounds of the data
                    - lower: lower bound of the data
                    - upper: upper bound of the data
                - desired: desired bounds of the data
                    - lower: lower bound of the data
                    - upper: upper bound of the data
        return:
            - normalized data
    """

    return bounds['desired']['lower'] + (data - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower'])


def itof2depth(itof: torch.Tensor | np.ndarray, freqs: tuple | float | int) -> torch.Tensor | np.ndarray:
    """
    Function used to convert the itof depth map to the correspondent radial depth map
        param:
            - itof: itof depth map
            - freqs: frequencies of the itof sensor (Hz)
        return:
            - radial depth map
    """

    # Select the correct data type
    if isinstance(itof, np.ndarray):
        env = np
        arr = np.array
    else:
        env = torch
        arr = torch.Tensor

    # Perform a check on freqs tu ensure that it is a tuple
    freqs = tuple([freqs]) if (isinstance(freqs, float) or isinstance(freqs, int)) else freqs

    # Check if there is the batch dimension
    if len(itof.shape) == 3 and isinstance(itof, torch.Tensor):
        itof = itof.unsqueeze(0)
    elif len(itof.shape) == 3 and isinstance(itof, np.ndarray):
        itof = itof[np.newaxis, ...]

    n_freqs = 1 if isinstance(freqs, float) or isinstance(freqs, int) else len(freqs)  # Number of frequencies used by the iToF sensor

    if n_freqs != itof.shape[1] // 2:
        raise ValueError("The number of frequencies is not equal to the number of channels in the itof map")

    # Compute the phase shift value (for each frequency)
    phi = env.arctan2(itof[:, n_freqs:, ...], itof[:, :n_freqs, ...]).squeeze(0)  # type: ignore

    # Compute the conversion value (for each frequency)
    conv_value =  const.c / (4 * const.pi * arr(freqs))
    # If necessary change the device of the conversion value
    if isinstance(itof, torch.Tensor):
        conv_value = conv_value.to(itof.device)  # type: ignore

    # Compute the radialdepth map
    depth = phi * conv_value

    # Set nan values to 0
    depth = env.nan_to_num(depth, nan=0, posinf=1e10, neginf=-1e10)  # type: ignore

    # Remove unnecessary dimensions
    if isinstance(depth, torch.Tensor) and len(depth.shape) == 4:
        depth = depth.squeeze(1)
    
    return depth  # type: ignore


def depth2itof(depth: torch.Tensor or np.ndarray, freq: float, ampl: float = 1.) -> torch.Tensor or np.ndarray:
    """
    Function used to convert the depth map to the correspondent itof depth map
        param:
            - depth: radial depth map
            - freq: frequency of the itof sensor (Hz)
            - ampl: amplitude of the data
        return:
            - itof data at the given frequence (Hz) as the real and immaginary part of the correspondent phasor
    """

    # Select the correct data type
    if isinstance(depth, np.ndarray):
        env = np
    else:
        env = torch

    # Compute the conversion value
    conv_value = (4 * const.pi * freq) / const.c
    # Computhe the shift value
    phi = depth * conv_value

    # Compute the real and imaginary part of the phasor
    real_phi = ampl * env.cos(phi)
    im_phi = ampl * env.sin(phi)

    # Compute the iToF data
    itof = env.stack((real_phi, im_phi), 0)  # type: ignore

    return itof  # type: ignore


def mean_intersection_over_union(pred: torch.Tensor, target: torch.Tensor, bg_class_value: int) -> torch.Tensor:
    """
    Function used to compute the binary mean Intersection over Union (mIoU)
        param:
            - pred: predicted data
            - target: target data
            - bg_class_value: background class value
        return:
            - mean intersection over union
    """

    iou_1 = binary_jaccard_index(torch.where(pred > bg_class_value, 1, 0), torch.where(target > bg_class_value, 1, 0))
    iou_2 = binary_jaccard_index(torch.where(pred == bg_class_value, 1, 0), torch.where(target == bg_class_value, 1, 0))

    return (iou_1 + iou_2) / 2


def plt_loss_hists(losses:np.ndarray, accuracies:np.ndarray, path: Path, bins: int = 40, a_only: bool = True, tex: bool = False) -> None:
    """
    Function that plots the histograms of the losses and accuracies
        param:
            - losses: losses
            - accuracies: accuracies
            - path: path where to save the plots
            - bins: number of bins
            - a_only: if True plot only the accuracies
            - tex: if True save the plots in tex format
    """

    # Set seaborn style
    sns.set_theme()

    fig = plt.figure(figsize=(16, 8))

    titles = ("Losses", "Accuracies")
    x_labels = ("Loss value", "Accuracy value")
    y_label = "Number of occurrences"
    data = (losses, accuracies)


    if not a_only:
        # Set the main title
        fig.suptitle("Histograms of the losses and accuracies on the test set", fontsize=14)
        
        # Define the axis
        ax = []
        ax.append(fig.add_subplot(1, 2, 1))
        ax.append(fig.add_subplot(1, 2, 2, sharey=ax[0]))

        for i, a in enumerate(ax):
            a.set_xlabel(x_labels[i])
            if i == 0:
                a.set_ylabel(y_label)
            a.set_title(titles[i])
            a.hist(data[i], bins=bins)
    else:
        plt.title(titles[1])
        plt.xlabel(x_labels[1])
        plt.ylabel(y_label)
        plt.hist(data[1], bins=bins)
    
    plt.tight_layout()
    plt.savefig(str(path.parent / "losses_histograms.svg"))
    if tex:
        tikzplotlib.save(str(path.parent / "losses_histograms.tex"))
    plt.close()


def plt_mae_hist(mae_losses: np.ndarray, path: Path, tex: bool = False) -> None:
    """
    Function that plots the histogram of the mae losses
        param:
            - mae_losses: mae losses
            - path: path where to save the plots
            - tex: if True save the plots in tex format
    """

    # Set seaborn style
    sns.set_theme()

    fig = plt.figure(figsize=(8, 7))

    # Set the main title
    fig.suptitle("Histogram of the MAE losses on the test set", fontsize=14)

    # Define the axis
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Loss value")
    ax.set_ylabel("Number of occurrences")
    ax.hist(mae_losses, bins=20)
    
    plt.tight_layout()
    plt.savefig(str(path.parent / "mae_losses_histogram.svg"))
    if tex:
        tikzplotlib.save(str(path.parent / "mae_losses_histogram.tex"))
    plt.close()