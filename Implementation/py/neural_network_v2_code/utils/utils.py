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


def save_np_as_img(data: np.ndarray, path: Path):
    """
    Function used to save each element of the numpy array as an image
        param:
            - data: numpy array
            - path: path where to save the images
    """

    # Create the folder where to save the images
    path.mkdir(parents=True, exist_ok=True)

    # For each element of the numpy array
    for i in range(data.shape[0]):
        # Save the image
        titles = ["Depth", "Mask"]
        c_range = [(np.min(data[i, 0, ...]), np.max(data[i, 0, ...])), (0, 1)]
        fig, ax = plt.subplots(1, 2)
        for j in range(2):
            img_t = ax[j].matshow(data[i, j, ...], cmap="jet")
            if range is not None:
                img_t.set_clim(c_range[0], c_range[1])
            divider = make_axes_locatable(ax[j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(mappable=img_t, cax=cax)
            ax[j].set_title(titles[j])
            ax[j].set_xlabel("Column pixel")
            ax[j].set_ylabel("Row pixel")
        plt.tight_layout()
        plt.imsave(str(path / f"{i}.png"), data[i, :, :])


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


def mae(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.numpy()
    gt = gt.numpy()
    pred_masked_ones = pred * gt
    num_ones = np.sum(gt)
    obj_mae = np.sum(np.abs(pred_masked_ones - gt)) / num_ones
    pred_masked_zeros = pred * (1 - gt)
    num_zeros = np.sum(1 - gt)
    alpha_mae_bkg = np.sum(np.abs(pred_masked_zeros - np.zeros(gt.shape, dtype=np.float32))) / num_zeros
    return np.sum(obj_mae + alpha_mae_bkg) / 2
