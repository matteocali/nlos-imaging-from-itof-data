import numpy as np
from tqdm import tqdm
from .. import utilities as ut
from matplotlib import pyplot as plt, animation
from pathlib import Path


def plt_transient_video(images, out_path, alpha, normalize):
    """
    Function that generate a video of the transient and save it in the matplotlib format
    (code from: https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib)
    :param images: list of all the transient images
    :param out_path: path where to save the video
    :param alpha: define if it has to use the alpha channel or not
    :param normalize: choose ti perform normalization or not
    """

    frames = []  # For storing the generated images
    fig = plt.figure()  # Create the figure

    mono = len(images.shape) == 3  # Check if the images are Mono or RGBA

    for img in tqdm(images):
        if normalize and not mono:
            img = ut.normalize_img(img[:, :, :-1])
        elif normalize and mono:
            img[:, :, :-1] = ut.normalize_img(img)

        if alpha or mono:
            frames.append([plt.imshow(img, animated=True)])  # Create each frame
        else:
            frames.append(
                [plt.imshow(img[:, :, :-1], animated=True)]
            )  # Create each frame without the alphamap

    ani = animation.ArtistAnimation(
        fig, frames, interval=50, blit=True, repeat_delay=1000
    )  # Create the animation
    ani.save(out_path)


def histo_plt(
    radiance: np.ndarray,
    exp_time: float,
    interval: list = None,
    stem: bool = True,
    file_path: Path = None,
):
    """
    Function that plot the transient histogram of a single pixel (for each channel)
    :param radiance: radiance value (foe each channel) of the given pixel [radiance_values, n_channel]
    :param exp_time: exposure time used during the rendering
    :param interval: list containing the min and max value of x-axis
    :param stem: flag to choose the type of graph
    :param file_path: file path where to save
    """

    mono = len(radiance.shape) == 1

    if interval is not None:
        if not mono:
            plt_start_pos = [int(interval[0] * 3e8 / exp_time * 1e-9)] * 3
            plt_end_pos = [int(interval[1] * 3e8 / exp_time * 1e-9)] * 3
        else:
            plt_start_pos = int(interval[0] * 3e8 / exp_time * 1e-9)
            plt_end_pos = int(interval[1] * 3e8 / exp_time * 1e-9)
    else:
        try:
            if not mono:
                plt_start_pos = [
                    np.where(radiance[:, channel] != 0)[0][0] - 10
                    for channel in range(0, 3)
                ]
                plt_end_pos = [
                    np.where(radiance[:, channel] != 0)[0][-1] + 11
                    for channel in range(0, 3)
                ]
            else:
                plt_start_pos = np.where(radiance != 0)[0][0] - 10
                plt_end_pos = np.where(radiance != 0)[0][-1] + 11
        except IndexError:
            if not mono:
                plt_start_pos = [0] * 3
                plt_end_pos = [len(radiance[:, channel]) for channel in range(0, 3)]
            else:
                plt_start_pos = 0
                plt_end_pos = len(radiance)

    radiance[np.where(radiance < 0)] = 0

    # Define the scale on the x-axis
    if str(exp_time).split(".")[0] == "0":
        unit_of_measure = 1e9  # nano seconds
        unit_of_measure_name = "ns"
    else:
        unit_of_measure = 1e6  # nano seconds
        unit_of_measure_name = r"$\mu s$"

    if not mono:
        colors = ["r", "g", "b"]
        colors_name = ["Red", "Green", "Blu"]

        # Plot hte transient histogram for each channel
        fig, axs = plt.subplots(1, 3, figsize=(24, 6))
        for i in range(radiance.shape[1] - 1):
            if stem:
                markers, stemlines, baseline = axs[i].stem(
                    range(0, len(radiance[plt_start_pos[i] : plt_end_pos[i], i])),
                    radiance[plt_start_pos[i] : plt_end_pos[i], i],
                )
                plt.setp(stemlines, color=colors[i])
                plt.setp(
                    baseline,
                    linestyle="dashed",
                    color="black",
                    linewidth=1,
                    visible=False,
                )
                plt.setp(markers, color=colors[i], markersize=1)
            else:
                axs[i].plot(
                    range(0, len(radiance[plt_start_pos[i] : plt_end_pos[i], i])),
                    radiance[plt_start_pos[i] : plt_end_pos[i], i],
                    color=colors[i],
                )
            axs[i].set_xticks(
                range(
                    0,
                    len(radiance[plt_start_pos[i] : plt_end_pos[i], i]) + 1,
                    int(len(radiance[plt_start_pos[i] : plt_end_pos[i], i] + 1) / 13),
                )
            )
            axs[i].set_xticklabels(
                [
                    "{:.2f}".format(round(value * exp_time / 3e8 * unit_of_measure, 2))
                    for value in range(
                        plt_start_pos[i],
                        plt_end_pos[i] + 1,
                        int(
                            len(radiance[plt_start_pos[i] : plt_end_pos[i], i] + 1) / 13
                        ),
                    )
                ],
                rotation=45,
            )
            axs[i].set_title(f"{colors_name[i]} channel histogram")
            axs[i].set_xlabel(f"Time instants [{unit_of_measure_name}]")
            axs[i].set_ylabel(r"Radiance value [$W/(m^{2}·sr)$]")
            axs[i].grid()
        fig.tight_layout()
    else:
        fig = plt.figure()
        if stem:
            markers, stemlines, baseline = plt.stem(
                range(0, len(radiance[plt_start_pos:plt_end_pos])),
                radiance[plt_start_pos:plt_end_pos],
            )
            plt.setp(stemlines, color="black")
            plt.setp(baseline, linestyle=" ", color="black", linewidth=1, visible=False)
            plt.setp(markers, color="black", markersize=1)
        else:
            plt.plot(
                range(0, len(radiance[plt_start_pos:plt_end_pos])),
                radiance[plt_start_pos:plt_end_pos],
                color="black",
            )
        plt.xticks(
            range(
                0,
                len(radiance[plt_start_pos:plt_end_pos]) + 1,
                int(len(radiance[plt_start_pos:plt_end_pos] + 1) / 13),
            ),
            [
                "{:.2f}".format(round(value * exp_time / 3e8 * unit_of_measure, 2))
                for value in range(
                    plt_start_pos,
                    plt_end_pos + 1,
                    int(len(radiance[plt_start_pos:plt_end_pos] + 1) / 13),
                )
            ],
            rotation=45,
        )
        plt.xlabel(f"Time instants [{unit_of_measure_name}]")
        plt.ylabel(r"Radiance value [$W/(m^{2}·sr)$]")
        plt.grid()
        fig.tight_layout()

    if file_path is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(str(file_path))
        plt.close()


def plot_phi(
    phi_matrix: np.ndarray,
    freq_values: np.ndarray,
    file_path: Path = None,
    exp_time: float = 0.01,
) -> None:
    """
    Function to plot the sine of the phi matrix
    :param phi_matrix: phi matrix data
    :param freq_values: used frequencies values
    :param file_path: file path + name where to save the plot (if not provided the plot will not be saved)
    :param exp_time: exposure_time
    """

    file_path = str(
        ut.add_extension(str(file_path), ".svg")
    )  # If necessary add the .svg extension to the file name

    # Define the scale on the x-axis based on the exposure time value
    if str(exp_time).split(".")[0] == "0":
        unit_of_measure = 1e9  # nano seconds
        unit_of_measure_name = "ns"
    else:
        unit_of_measure = 1e6  # nano seconds
        unit_of_measure_name = r"$\mu s$"

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))  # Create the figure of the plot
    index = 0  # Index used to select the right phi value to plot
    for c in range(2):
        for r in range(3):
            axs[r, c].plot(phi_matrix[index, :])  # Plot the phi values
            axs[r, c].set_xticks(
                range(0, phi_matrix.shape[1] + 1, 500)
            )  # Put a tick on the x-axis every 500 time bins
            axs[r, c].set_xticklabels(
                [
                    "{:.2f}".format(round(value * exp_time / 3e8 * unit_of_measure, 2))
                    for value in range(0, phi_matrix.shape[1] + 1, 500)
                ]
            )  # Change the labels of the x-axis in order to display the time delay in the right unit of measure
            if c == 0:
                axs[r, c].title.set_text(
                    f"Cosine at freq.: {np.format_float_scientific(freq_values[r], trim='-', exp_digits=1)} Hz"
                )  # Add a title to each subplot
            else:
                axs[r, c].title.set_text(
                    f"Sine at freq.: {np.format_float_scientific(freq_values[r], trim='-', exp_digits=1)} Hz"
                )  # Add a title to each subplot
            axs[r, c].set_xlabel(
                f"Time instants [{unit_of_measure_name}]"
            )  # Define the label on the x-axis using the correct unit of measure
            axs[r, c].grid()  # Add the grid
            index += 1
    fig.tight_layout()  # adapt the subplots dimension to the one of the figure

    if file_path is not None:
        plt.savefig(file_path)  # If a path is provided save the plot
    else:
        plt.show()  # Otherwise display it

    plt.close()
