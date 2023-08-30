import numpy as np
import seaborn as sns
import tikzplotlib
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        img = ax[i].matshow(data[i].T, cmap="jet")  # Plot the sx plot # type: ignore
        if clim:
            img.set_clim(
                np.min(data[0]), np.max(data[0])
            )  # Set the colorbar limits based on the ground truth data
        divider = make_axes_locatable(ax[i])  # Defien the colorbar axis
        cax = divider.append_axes(
            "right", size="5%", pad=0.05
        )  # Set the colorbar location
        fig.colorbar(img, cax=cax, label="Depth [m]")  # Plot the colorbar

        if (
            i == 1 and loss is not None
        ):  # If the plot is the predicted one and the loss is not None
            box_style = dict(
                boxstyle="round", fc="w", ec="black", alpha=0.9
            )  # Define the box style
            ax[i].text(
                20,
                20,
                f"MAE: {round(loss, 3)}",
                ha="left",
                va="top",
                fontsize=11,
                fontfamily="monospace",
                color="black",
                bbox=box_style,
            )  # Add the box to the plot # type: ignore
            if iou is not None:  # If the miou is not None
                ax[i].text(
                    20,
                    40,
                    f"IoU: {round(iou, 3)}",
                    ha="left",
                    va="top",
                    fontsize=11,
                    fontfamily="monospace",
                    color="black",
                    bbox=box_style,
                )  # Add the box for the miou to the plot # type: ignore
        ax[i].set_title(titles[i])  # Set the title of the subplot
        ax[i].set_xlabel("X")  # Set the x label of the subplot
        ax[i].set_ylabel(
            "Y", rotation=0
        )  # Set the y label of the subplot and rotrate it


def save_test_plots_itof(
    depth_data: tuple[np.ndarray, np.ndarray],
    itof_data: tuple[np.ndarray, np.ndarray],
    losses: tuple[float, float, float],
    index: int,
    path: Path,
    iou: float | None = None,
    tex: bool = False,
) -> None:
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
    row_subplot(
        fig, ax[0], (depth_data[0], depth_data[1]), titles, losses[0], iou, clim
    )
    # Generate the plts for the itof
    # Real iToF
    titles = ("Grount truth real iToF", "Predicted real iToF")
    row_subplot(
        fig,
        ax[1],
        (itof_data[0][0, ...], itof_data[1][0, ...]),
        titles,
        losses[1],
        clim,
    )
    # Imaginary iToF
    titles = ("Grount truth imaginary iToF", "Predicted imaginary iToF")
    row_subplot(
        fig,
        ax[2],
        (itof_data[0][1, ...], itof_data[1][1, ...]),
        titles,
        losses[2],
        clim,
    )

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


def plt_loss_hists(
    losses: np.ndarray,
    accuracies: np.ndarray,
    path: Path,
    bins: int = 40,
    a_only: bool = True,
    tex: bool = False,
) -> None:
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
        fig.suptitle(
            "Histograms of the losses and accuracies on the test set", fontsize=14
        )

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
