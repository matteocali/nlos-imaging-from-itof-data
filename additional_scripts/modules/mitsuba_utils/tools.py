import numpy as np
from matplotlib import pyplot as plt, cm, colors
from .utils import theta_calculator
from .plots import save_cross_section_plot
from .. import transient_utils as tr, utilities as ut, exr_handler as exr


def cross_section_tester(images, tot_img, exp_time, fov, output_path):
    """
    Function that generate the graphs of the cross-section analysis on the principal row and column
    :param images: np array containing the images of the transient [n_beans, n_row, n_col, 3]
    :param tot_img: image corresponding to the sum of all the transient images over the temporal dimension
    :param exp_time: exposure time used during the rendering
    :param fov: field of view of the camera
    :param output_path: path of the output folder
    """
    theta_row, theta_col, row_distance, col_distance = theta_calculator(
        peak_pos=tr.tools.extract_center_peak(images)[0][0],
        peak_row_values=images[0].shape[1],
        peak_col_values=images[0].shape[0],
        e_time=exp_time,
        fov=fov,
    )

    save_cross_section_plot(
        theta_r=theta_row,
        theta_c=theta_col,
        row_distance=row_distance,
        col_distance=col_distance,
        r_values=list(tot_img[int(tot_img.shape[1] / 2), :, 0]),
        c_values=list(tot_img[:, int(tot_img.shape[1] / 2), 0]),
        output=str(output_path / "cross_section"),
    )


def img_comparison(o_img, t_img, out_path, diff_limits=None, ratio_limits=None):
    """
    Function to plot the comparison between the real image and the one obtained by summing the transient over the temporal direction (+ compute the MSE)
    :param out_path: folder path where to save the graphs
    :param diff_limits: fixed np.min and np.max value of the diff colorbar (tuple)
    :param ratio_limits: fixed np.min and np.max value of the ratio colorbar (tuple)
    :param o_img: original image [R, G, B]
    :param t_img: transient image [R, G, B]
    """
    print(
        "Compare the original images with the one obtained summing all the transient ones"
    )
    print(f"The MSE is {ut.compute_mse(o_img, t_img)}\n")

    # Extract the minimum and maximum displayed value to normalize the colors
    min_val = np.min([np.min(o_img), np.min(t_img)])
    max_val = np.max([np.max(o_img), np.max(t_img)])

    # Plot each channel of both the image, together with the colorbar
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs[0, 0].matshow(
        o_img[:, :, 0],
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs[0, 0].set_title("Red channel of the original image")
    axs[1, 0].matshow(
        o_img[:, :, 1],
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs[1, 0].set_title("Green channel of the original image")
    axs[2, 0].matshow(
        o_img[:, :, 2],
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs[2, 0].set_title("Blu channel of the original image")
    axs[0, 1].matshow(
        t_img[:, :, 0],
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs[0, 1].set_title("Red channel of the transient image")
    axs[1, 1].matshow(
        t_img[:, :, 1],
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs[1, 1].set_title("Green channel of the transient image")
    axs[2, 1].matshow(
        t_img[:, :, 2],
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs[2, 1].set_title("Blu channel of the transient image")
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.Normalize(vmin=min_val, vmax=max_val), cmap=cm.get_cmap("jet")
        ),
        ax=axs,
        label=r"Radiance [$W/(m^{2}·sr)$]",
    )
    plt.savefig(str(out_path / "channel_comparison.svg"))

    # Compute the differences between the original and transient image, channel by channel
    r_diff = abs(t_img[:, :, 0] - o_img[:, :, 0])
    g_diff = abs(t_img[:, :, 1] - o_img[:, :, 1])
    b_diff = abs(t_img[:, :, 2] - o_img[:, :, 2])

    # Extract the minimum and maximum displayed value to normalize the colors
    if diff_limits is None:
        min_val = np.min([np.min(r_diff), np.min(g_diff), np.min(b_diff)])
        max_val = np.max([np.max(r_diff), np.max(g_diff), np.max(b_diff)])
    else:
        min_val = diff_limits[0]
        max_val = diff_limits[1]

    # Plot the difference between the two images, channel by channel
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
    axs2[0].matshow(
        r_diff,
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs2[0].set_title("Difference on the red channel")
    axs2[1].matshow(
        g_diff,
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs2[1].set_title("Difference on the green channel")
    axs2[2].matshow(
        b_diff,
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs2[2].set_title("Difference on the blu channel")
    fig2.colorbar(
        cm.ScalarMappable(
            norm=colors.Normalize(vmin=min_val, vmax=max_val), cmap=cm.get_cmap("jet")
        ),
        ax=axs2,
        orientation="horizontal",
    )
    plt.savefig(str(out_path / "channel_differences.svg"))

    o_img[np.where(o_img == 0)] = 1  # Remove eventual 0 values

    # Compute the ratio between the original and transient image, channel by channel
    r_div = t_img[:, :, 0] / o_img[:, :, 0]
    g_div = t_img[:, :, 1] / o_img[:, :, 1]
    b_div = t_img[:, :, 2] / o_img[:, :, 2]

    # Extract the minimum and maximum displayed value to normalize the colors
    if ratio_limits is None:
        min_val = np.min([np.min(r_div), np.min(g_div), np.min(b_div)])
        max_val = np.max([np.max(r_div), np.max(g_div), np.max(b_div)])
    else:
        min_val = ratio_limits[0]
        max_val = ratio_limits[1]

    # Plot the ratio between the two images, channel by channel
    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 6))
    axs3[0].matshow(
        r_div,
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs3[0].set_title("Ratio on the red channel (original/transient)")
    axs3[1].matshow(
        g_div,
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs3[1].set_title("Ratio on the green channel (original/transient)")
    axs3[2].matshow(
        b_div,
        cmap=cm.get_cmap("jet"),
        norm=colors.Normalize(vmin=min_val, vmax=max_val),
    )
    axs3[2].set_title("Ratio on the blu channel (original/transient)")
    fig3.colorbar(
        cm.ScalarMappable(
            norm=colors.Normalize(vmin=min_val, vmax=max_val), cmap=cm.get_cmap("jet")
        ),
        ax=axs3,
        orientation="horizontal",
    )
    plt.savefig(str(out_path / "channel_ratio.svg"))


def tot_img_tester(
    rgb_img_path, total_img, out_path, diff_limits=None, ratio_limits=None
):
    """
    Function that compare the total image with the standard rgb render
    :param out_path: folder path where to save the graphs
    :param ratio_limits: fixed np.min and np.max value of the ratio colorbar (tuple)
    :param diff_limits: fixed np.min and np.max value of the diff colorbar (tuple)
    :param rgb_img_path: path of the standard RGB image
    :param total_img: total image
    """
    original_img = exr.load_exr(str(rgb_img_path))  # Load the original image
    original_img[np.isnan(original_img[:, :, 0])] = 0  # Remove the nan value
    original_img = original_img[:, :, 1:]  # Remove the alpha channel

    img_comparison(
        original_img, total_img, out_path, diff_limits, ratio_limits
    )  # Compare the original render with the one obtained by summing up all the transient images
