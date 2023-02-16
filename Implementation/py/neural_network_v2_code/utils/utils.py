import numpy as np
import scipy.constants as const
import smtplib
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.optim import Optimizer
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
    titles = ["Grount truth depth", "Predicted depth"]
    for i in range(2):
        img = ax[0, i].matshow(depth_data[i].T, cmap="jet")  # type: ignore
        img.set_clim(np.min(depth_data[0]), np.max(mask_data[0]))
        divider = make_axes_locatable(ax[0, i])  # type: ignore
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mappable=img, cax=cax)
        if i ==1:
            box_style = dict(boxstyle="round", fc="w", ec="black", alpha=0.9)
            ax[0, i].text(20, 20, f"MAE: {round(losses[0], 3)}", ha='left', va='top', fontsize=11, color='black', bbox=box_style)  # type: ignore
        ax[0, i].set_title(titles[i])        # type: ignore
        ax[0, i].set_xlabel("Column pixel")  # type: ignore
        ax[0, i].set_ylabel("Row pixel")     # type: ignore
    # Generate the plts for the mask
    titles = ["Grount truth mask", "Predicted mask"]
    for i in range(2):
        img = ax[1, i].matshow(mask_data[i].T, cmap="jet")  # type: ignore
        img.set_clim(np.min(mask_data[0]), np.max(mask_data[0]))
        divider = make_axes_locatable(ax[1, i])  # type: ignore
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mappable=img, cax=cax)
        if i ==1:
            box_style = dict(boxstyle="round", fc="w", ec="black", alpha=0.9)
            ax[0, i].text(20, 20, f"MAE: {round(losses[0], 3)}", ha='left', va='top', fontsize=11, color='black', bbox=box_style)  # type: ignore
        ax[1, i].set_title(titles[i])        # type: ignore
        ax[1, i].set_xlabel("Column pixel")  # type: ignore
        ax[1, i].set_ylabel("Row pixel")     # type: ignore
    
    plt.tight_layout()
    plt.savefig(str(path / f"{index}.svg"))
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
