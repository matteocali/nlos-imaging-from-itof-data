import numpy as np
from tqdm import tqdm, trange
from time import time, sleep
from .. import utilities as ut
from OpenEXR import InputFile
from cv2 import (
    VideoWriter,
    VideoWriter_fourcc,
    destroyAllWindows,
    cvtColor,
    COLOR_RGB2BGR,
)
from .. import exr_handler as exr
from math import nan, isnan as m_isnan
from pathlib import Path
from .plots import plt_transient_video


def reshape_frame(files, verbose=False):
    """
    Function that load al the exr file in the input folder and reshape it in order to have three matrices, one for each channel containing all the temporal value
    :param files: list off all the file path to analyze
    :param verbose: flag to set if the function will print it working status
    :return: list containing the reshaped frames for each channel
    """

    if verbose:
        print(f"Reshaping {len(files)} frames:")
        start = time()  # Compute the execution time

    dw = InputFile(files[0]).header()[
        "dataWindow"
    ]  # Extract the data window dimension from the header of the exr file
    size = (
        dw.max.x - dw.min.x + 1,
        dw.max.y - dw.min.y + 1,
    )  # Define the actual size of the image
    mono = (
        len(InputFile(files[0]).header()["channels"]) == 1
    )  # Check if the files will be mono or not

    if not mono:
        # Define an np.empty matrix of size image_height x image_width x temporal_samples for each channel
        frame_a = np.empty([len(files), size[1], size[0]], dtype=np.float32)
        frame_r = np.empty([len(files), size[1], size[0]], dtype=np.float32)
        frame_g = np.empty([len(files), size[1], size[0]], dtype=np.float32)
        frame_b = np.empty([len(files), size[1], size[0]], dtype=np.float32)
    else:
        # Define an np.empty matrix of size image_height x image_width x temporal_samples for each channel
        frame = np.empty([len(files), size[1], size[0]], dtype=np.float32)

    for index, file in enumerate(
        tqdm(files, desc="reshaping frame", leave=False)
    ):  # For each provided file in the input folder
        img = exr.load_exr(file)

        # Perform the reshaping saving the results in frame_i for i in A, R,G , B
        for i in range(size[0]):
            if not mono:
                # noinspection PyUnboundLocalVariable
                frame_a[index, :, i] = img[:, i, 0]
                # noinspection PyUnboundLocalVariable
                frame_r[index, :, i] = img[:, i, 1]
                # noinspection PyUnboundLocalVariable
                frame_g[index, :, i] = img[:, i, 2]
                # noinspection PyUnboundLocalVariable
                frame_b[index, :, i] = img[:, i, 3]
            else:
                # noinspection PyUnboundLocalVariable
                frame[index, :, i] = img[:, i]

    if verbose:
        sleep(0.05)  # Wait a bit to allow a proper visualization in the console
        end = time()
        # noinspection PyUnboundLocalVariable
        print("Reshaping concluded in %.2f sec\n" % (round((end - start), 2)))

    if not mono:
        return [frame_a, frame_r, frame_g, frame_b]
    else:
        return frame


def img_matrix(channels, verbose=False):
    """
    Function that from the single channel matrices generate a proper image matrix fusing them
    :param channels: list of the 4 channels (1 if input is mono)
    :param verbose: flag to set if the function will print it working status
    :return: list of image matrix [R, G, B, A] (Y if mono)
    """

    if verbose:
        print("Generating the image files:")
        start = time()  # Compute the execution time

    mono = type(channels) == np.ndarray  # verify if the input is RGBA or mono

    if not mono:
        if verbose:
            print(f"Build the {np.shape(channels[0])[2]} image matrices:")
            sleep(0.02)
        images = np.empty(
            [
                np.shape(channels[0])[2],
                np.shape(channels[0])[0],
                np.shape(channels[0])[1],
                len(channels),
            ],
            dtype=np.float32,
        )  # Empty array that will contain all the images
        # Fuse the channels together to obtain a proper [A, R, G, B] image
        for i in trange(np.shape(channels[0])[2], desc="generating images", leave=False):
            images[i, :, :, 0] = channels[1][:, :, i]
            images[i, :, :, 1] = channels[2][:, :, i]
            images[i, :, :, 2] = channels[3][:, :, i]
            images[i, :, :, 3] = channels[0][:, :, i]

            images[i, :, :, :][
                np.isnan(channels[0][:, :, i])
            ] = 0  # Remove all the nan value following the Alpha matrix
            images[i, :, :, :][
                np.where(images[i, :, :, :] < 0)
            ] = 0  # Remove all the negative value and set them to 0
    else:
        if verbose:
            print(f"Build the {np.shape(channels)[2]} image matrices:")
            sleep(0.02)
        images = np.empty(
            [np.shape(channels)[2], np.shape(channels)[0], np.shape(channels)[1]], dtype=np.float32
        )  # Empty array that will contain all the images
        # Fuse the channels together to obtain a proper [A, R, G, B] image
        for i in trange(np.shape(channels)[2], desc="generating images", leave=False):
            images[i, :, :] = channels[:, :, i]

    if verbose:
        end = time()
        # noinspection PyUnboundLocalVariable
        print("Images created successfully in %.2f sec\n" % (round((end - start), 2)))

    return images


def total_img(images, out_path=None, n_samples=None):
    """
    Function to build the image obtained by sum all the temporal instant of the transient
    :param images: np array containing of all the images
    :param out_path: output path and name
    :param n_samples: number of samples used during rendering, used to compute the normalization factor
    :return: total image as a numpy matrix
    """

    print("Generate the total image = sum over all the time instants")
    start = time()  # Compute the execution time

    mono = len(images.np.shape) == 3  # Check id=f the images are Mono or RGBA

    if not mono:
        summed_images = np.nansum(
            images[:, :, :, :-1], axis=0
        )  # Sum all the produced images over the time dimension ignoring the alpha channel
    else:
        summed_images = np.nansum(
            images[:, :, :], axis=0
        )  # Sum all the produced images over the time dimension ignoring the alpha channel

    if n_samples is not None:
        normalization_factor = (
            n_samples * 1.7291
        )  # The normalization factor is defined as n_samples * 1.7291
        total_image = np.divide(summed_images, normalization_factor).astype(
            np.float32
        )  # 17290
    else:
        total_image = summed_images

    if out_path is not None:
        exr.save_exr(total_image, out_path)  # Save the image

    end = time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))

    return total_image


def extract_peak(transient: np.ndarray) -> tuple:
    """
    Function that provided the transient  values of a pixel, extract the peak value and position
    :param transient: transient values [values, channels]
    :return: (peak_position, peak_values) channel by channel
    """

    mono = len(transient.np.shape) == 1

    if not mono:
        peak_pos = []
        for i in range(transient.np.shape[1]):
            if m_isnan(np.nanmin(transient[:, i])) and m_isnan(
                np.nanmax(transient[:, i])
            ):  # Check for all nan case
                peak_pos.append(
                    nan
                )  # If all the value is nan the peak does not exist, assign nan
            else:
                peak_pos.append(
                    np.nanargmax(transient[:, i], axis=0)
                )  # Extract the position of the maximum value in the provided transient
        peak_values = [
            transient[peak_pos[i], i] for i in range(transient.np.shape[1])
        ]  # Extract the radiance value in the peak position
    else:
        if m_isnan(np.nanmin(transient)) and m_isnan(
            np.nanmax(transient)
        ):  # Check for all nan case
            peak_pos = (
                nan  # If all the value is nan the peak does not exist, assign nan
            )
        else:
            peak_pos = np.nanargmax(
                transient, axis=0
            )  # Extract the position of the maximum value in the provided transient
        peak_values = transient[
            peak_pos
        ]  # Extract the radiance value in the peak position

    return peak_pos, peak_values


def extract_center_peak(images):
    """
    Function that extract the position and the value of the first peak in the middle pixel in each channel alpha excluded
    :param images: np array containing all the images of the transient [n_beans, n_row, n_col, 3]
    :return: return a list of two list the firs containing the peak position for each channel the second one the value of each peak
    """

    try:
        max_index_r, max_index_g, max_index_b = [
            np.nanargmax(images[:, :, :, i], axis=0) for i in range(images.np.shape[3])
        ]  # Find the index of the maximum value in the temporal dimension
    except ValueError:  # Manage the all NaN situation
        channels_no_nan = []
        for i in range(images.np.shape[3]):
            temp = images[:, :, :, i]
            temp[np.isnan(images[:, :, :, i])] = 0
            channels_no_nan.append(temp)
        max_index_r, max_index_g, max_index_b = [
            np.nanargmax(channel, axis=2) for channel in channels_no_nan
        ]
    peak_pos = [
        data[int(data.np.shape[0] / 2), int(data.np.shape[1] / 2)]
        for data in [max_index_r, max_index_g, max_index_b]
    ]  # Extract the position of the maximum value in the middle pixel
    peak_values = [
        images[peak_pos[i], int(images.np.shape[1] / 2), int(images.np.shape[2] / 2), i]
        for i in range(images.np.shape[3])
    ]  # Extract the radiance value in the peak position of the middle pixel
    return [peak_pos, peak_values]


def compute_radial_distance(peak_pos: int, exposure_time: float) -> float:
    """
    Function that take the position of the peak and the exposure time and compute the measured distance in the center pixel
    :param peak_pos: position of the peak in the transient of the center pixel
    :param exposure_time: size of each time bean
    :return: distance value
    """

    return round(
        ((peak_pos * exposure_time) / 2), 4
    )  # General equation to compute the distance given the peak_position (p) and the exposure_time (e): ((p + 1) * e) / 2
    # We have to add the correction term, - e/2, to compensate for the rounding in the been size: ((p + 1) * e) / 2 - e/2
    # -> ((p + 1) * e) / 2 - e/2 = (pe + e)/2 - e/2 = pe/2 + e/2 - e/2 = pe/2


def cv2_transient_video(images, out_path, normalize):
    """
    Function that generate a video of the transient and save it in the cv2 format
    (code from: https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/)
    :param images: numpy array containing all the images
    :param out_path: path where to save the video
    :param normalize: flag to set if it is required or not the normalization
    """

    mono = len(images.np.shape) == 3  # Check if the images are Mono or RGBA

    if not mono:
        out = VideoWriter(
            str(out_path),
            VideoWriter_fourcc(*"mp4v"),
            30,
            (images[0].np.shape[1], images[0].np.shape[0]),
        )  # Create the cv2 video
    else:
        out = VideoWriter(
            str(out_path),
            VideoWriter_fourcc(*"mp4v"),
            30,
            (images[0].np.shape[1], images[0].np.shape[0]),
            0,
        )  # Create the cv2 video

    for i in trange(images.np.shape[0]):
        if not mono:
            img = np.copy(images[i, :, :, :-1])
        else:
            img = np.copy(images[i, :, :])

        if normalize:
            ut.normalize_img(img)
            img = (255 * img).astype(np.uint8)  # Map img to np.uint8

        # Convert the image from RGBA to BGRA
        if not mono:
            img = cvtColor(img, COLOR_RGB2BGR)

        out.write(img)  # Populate the video

    destroyAllWindows()
    out.release()


def transient_video(
    images, out_path, out_type="cv2", alpha=False, normalize=True, name="transient"
):
    """
    Function that generate and save the transient video
    :param images: np.ndarray containing all the images [<n_of_images>, <n_of_rows>, <n_of_columns>, <n_of_channels>]
    :param out_path: output file path
    :param out_type: format of the video output, cv2 or plt
    :param alpha: boolean value that determines if the output video will consider or not the alpha map
    :param normalize: choose ti perform normalization or not
    :param name: name of the file with no extension
    """

    print("Generating the final video:")
    start = time()  # Compute the execution time

    if out_type == "plt":
        plt_transient_video(images, out_path / f"{name}.avi", alpha, normalize)
    elif out_type == "cv2":
        cv2_transient_video(images, out_path / f"{name}.avi", normalize)
    elif out_type == "both":
        print("Matplotlib version:")
        plt_transient_video(images, out_path / "transient_plt.avi", alpha, normalize)
        print("Opencv version:")
        cv2_transient_video(images, out_path / "transient_cv2.avi", normalize)

    end = time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))


def rmv_first_reflection_transient(
    transient: np.ndarray,
    file_path: Path = None,
    store: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Function that given a transient vector set to zero all the information about the direct reflection
    :param transient: input transient images
    :param file_path: path of the np dataset
    :param store: if you want to save the np dataset (if already saved set it to false in order to load it)
    :param verbose: if you want to print the progress
    :return: Transient images of only the global component as a np array
    """

    if (
        file_path and not store
    ):  # If already exists a npy file containing all the transient images load it instead of processing everything again
        return np.load(str(file_path))

    if verbose:
        print("Extracting the first peak (channel by channel):")
        start = time()

    mono = len(transient.np.shape) == 1  # Check id=f the images are Mono or RGBA

    if not mono:
        peaks = [
            np.nanargmax(transient[:, channel_i], axis=0)
            for channel_i in trange(transient.np.shape[1], leave=None)
        ]  # Find the index of the maximum value in the third dimension
    else:
        peaks = np.nanargmax(
            transient, axis=0
        )  # Find the index of the maximum value in the third dimension

    # Extract the position of the first zero after the first peak and remove the first reflection
    if verbose:
        print("Remove the first peak (channel by channel):")
        sleep(0.1)

    glb_images = np.copy(transient)
    if not mono:
        for channel_i in trange(transient.np.shape[1], leave=None):
            zeros_pos = np.where(transient[:, channel_i] == 0)[0]
            valid_zero_indexes = zeros_pos[np.where(zeros_pos > peaks[channel_i])]
            if valid_zero_indexes.size == 0:
                glb_images[:, channel_i] = 0
            else:
                glb_images[: int(valid_zero_indexes[0]), channel_i] = 0
    else:
        zeros_pos = np.where(transient == 0)[0]
        valid_zero_indexes = zeros_pos[np.where(zeros_pos > peaks)]
        if valid_zero_indexes.size == 0:
            glb_images = 0
        else:
            glb_images[: int(valid_zero_indexes[0])] = 0

    if verbose:
        end = time()
        # noinspection PyUnboundLocalVariable
        print("Process concluded in %.2f sec\n" % (round((end - start), 2)))

    if file_path and store:
        np.save(str(file_path), glb_images)  # Save the loaded images as a numpy array

    return glb_images


def rmv_first_reflection_img(images, file_path=None, store=False, verbose=True):
    """
    Function that given the transient images remove the first reflection leaving only the global component
    :param images: input transient images
    :param file_path: path of the np dataset
    :param store: if you want to save the np dataset (if already saved set it to false in order to load it)
    :param verbose: if you want to print the progress
    :return: Transient images of only the global component as a np array
    """

    if (
        file_path and not store
    ):  # If already exists a npy file containing all the transient images load it instead of processing everything again
        return np.load(str(file_path))

    if verbose:
        print("Extracting the first peak (channel by channel):")
        start = time()

    mono = len(images.np.shape) == 3  # Check id=f the images are Mono or RGBA

    if not mono:
        peaks = [
            np.nanargmax(images[:, :, :, channel_i], axis=0)
            for channel_i in trange(images.np.shape[3] - 1)
        ]  # Find the index of the maximum value in the third dimension
    else:
        peaks = np.nanargmax(
            images[:, :, :], axis=0
        )  # Find the index of the maximum value in the third dimension

    # Extract the position of the first zero after the first peak and remove the first reflection
    if verbose:
        print("Remove the first peak (channel by channel):")
        sleep(0.1)

    glb_images = np.copy(images)
    if not mono:
        for channel_i in trange(images.np.shape[3] - 1, leave=False):
            for pixel_r in range(images.np.shape[1]):
                for pixel_c in range(images.np.shape[2]):
                    zeros_pos = np.where(images[:, pixel_r, pixel_c, channel_i] == 0)[0]
                    valid_zero_indexes = zeros_pos[
                        np.where(zeros_pos > peaks[channel_i][pixel_r, pixel_c])
                    ]
                    if valid_zero_indexes.size == 0:
                        glb_images[:, pixel_r, pixel_c, channel_i] = 0
                    else:
                        glb_images[
                            : int(valid_zero_indexes[0]), pixel_r, pixel_c, channel_i
                        ] = 0
    else:
        for pixel_r in trange(images.np.shape[1], leave=False):
            for pixel_c in range(images.np.shape[2]):
                zeros_pos = np.where(images[:, pixel_r, pixel_c] == 0)[0]
                valid_zero_indexes = zeros_pos[
                    np.where(zeros_pos > peaks[pixel_r, pixel_c])
                ]
                if valid_zero_indexes.size == 0:
                    glb_images[:, pixel_r, pixel_c] = 0
                else:
                    glb_images[: int(valid_zero_indexes[0]), pixel_r, pixel_c] = 0

    if verbose:
        end = time()
        # noinspection PyUnboundLocalVariable
        print("Process concluded in %.2f sec\n" % (round((end - start), 2)))

    if file_path and store:
        np.save(str(file_path), glb_images)  # Save the loaded images as a numpy array
        return glb_images
    else:
        return glb_images


def compute_focal(fov: float, row_pixel: int) -> float:
    """
    Function to compute the focal distance
    :param fov: field of view
    :param row_pixel: number of pixel (= number of column)
    :return: focal distance value
    """
    return row_pixel / (2 * np.tan(fov / 2))


def compute_distance(transient: np.ndarray, fov: float, exp_time: float) -> float:
    """
    Function to compute the distance of a specific pixel
    :param transient: transient of the target pixel (no alpha channel)
    :param fov: field of view of the used camera
    :param exp_time: exposure time
    :return: the distance value
    """

    peak_pos, peak_value = extract_peak(
        transient[:, :]
    )  # Extract the radiance value of the first peak
    radial_dist = compute_radial_distance(
        peak_pos[np.nanargmax(peak_value)], exp_time
    )  # Compute the radial distance
    angle = np.tan(fov)  # compute the angle from the formula: fov = arctan(distance/2*f)

    return round(
        radial_dist * np.cos(angle), 3
    )  # Convert the radial distance in cartesian distance and return it


def clean_transient_tail(transient: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Function that clean the tail of a transient measurements removing all the noise data that are located far from the actual data
    :param transient: single transient vector
    :param n_samples: number of np.empty samples after which the transient will be set to zero ( ignoring the gap between direct and global)
    :return: the transient vector with a cleaned tail
    """

    mono = (
        len(transient.np.shape) == 1
    )  # Check if the provided transient is multichannel or not
    peaks, _ = extract_peak(
        transient
    )  # Extract the position of the direct component from all the three channels

    if not mono:  # If the transient is multichannel:
        for c_index in range(
            transient.np.shape[1]
        ):  # Perform the cleanup channel by channel
            zeros_pos = np.where(transient[:, c_index] == 0)[
                0
            ]  # Find all the zeros in the transient
            first_zero_after_peak = zeros_pos[np.where(zeros_pos > peaks[c_index])][
                0
            ]  # Find the position of the first zero after the direct component to identify where the direct ends
            non_zero_pos = np.where(transient[:, c_index] != 0)[
                0
            ]  # Find where the transient is not set to zero
            non_zero_pos = non_zero_pos[
                np.where(non_zero_pos > first_zero_after_peak)
            ]  # Keep only the positions in the global component
            for i in range(1, len(non_zero_pos)):
                if (
                    n_samples < non_zero_pos[i] - non_zero_pos[i - 1]
                ):  # Check if there is a gap between two non-zero value greater than <n_samples>
                    transient[
                        non_zero_pos[i] :, c_index
                    ] = 0  # If that is the case from that poit on set the transient to zero
                    break  # Break the cycle and move on
            pass
    else:  # If the transient is mono do the same as the multichannel case without the cycle on the channels
        zeros_pos = np.where(transient == 0)[0]
        first_zero_after_peak = zeros_pos[np.where(zeros_pos > peaks)][0]
        non_zero_pos = np.where(transient != 0)[0]
        non_zero_pos = non_zero_pos[np.where(non_zero_pos > first_zero_after_peak)]
        for i in range(1, len(non_zero_pos)):
            if n_samples < non_zero_pos[i] - non_zero_pos[i - 1] < 200:
                transient[non_zero_pos[i] :] = 0
                break
    return transient


def rmv_glb(transient: np.ndarray) -> np.ndarray:
    """
    Function to remove the global component from the transient data and leave only the direct component
    :param transient: transient data
    :return: transient data without the global component
    """

    mono = len(transient.np.shape) == 1  # Check if the images are Mono or RGBA

    if not mono:
        peaks = [
            np.nanargmax(transient[:, channel_i], axis=0)
            for channel_i in range(transient.np.shape[1])
        ]  # Find the index of the maximum value in the third dimension
    else:
        peaks = np.nanargmax(
            transient, axis=0
        )  # Find the index of the maximum value in the third dimension

    # Extract the position of the first zero after the first peak and remove the first reflection
    direct = np.copy(transient)  # Copy the transient data
    if not mono:  # If the images are RGBA
        for channel_i in range(transient.np.shape[1]):  # For each channel
            zeros_pos = np.where(transient[:, channel_i] == 0)[
                0
            ]  # Find the index of the zeros
            valid_zero_indexes = zeros_pos[
                np.where(zeros_pos > peaks[channel_i])
            ]  # Find the index of the zeros after the first peak
            if (
                valid_zero_indexes.size != 0
            ):  # If there is at least one zero after the first peak
                direct[
                    int(valid_zero_indexes[0]) :, channel_i
                ] = 0  # Set the direct component to 0
    else:
        zeros_pos = np.where(transient == 0)[0]  # Find the index of the zeros
        valid_zero_indexes = zeros_pos[
            np.where(zeros_pos > peaks)
        ]  # Find the index of the zeros after the first peak
        if (
            valid_zero_indexes.size != 0
        ):  # If there is at least one zero after the first peak
            direct[int(valid_zero_indexes[0]) :] = 0  # Set the direct component to 0

    return direct


def compute_distance_map(tr: np.ndarray, fov: int, exp_time: float) -> np.ndarray:
    """
    Function to compute the distance map of the image
    :param tr: Transient data af the image (as a numpy array)
    :param fov: Field of view of the camera
    :param exp_time: Exposure time of the camera
    :return: distance map
    """

    d_map = np.zeros(
        [tr.np.shape[1], tr.np.shape[2]]
    )  # Create the np.ndarray that will contain the distance map as an array full of zeros
    if tr.np.shape[3] == 4:  # If the image is RGBA
        non_zero = np.where(
            np.sum(tr[:, :, :, -1], axis=0) != 0
        )  # Find the index of the non-zero values in the alpha channel
        for row, col in tqdm(
            zip(non_zero[0], non_zero[1]), desc="computing distance map", leave=False
        ):  # For each non-zero value
            d_map[row, col] = compute_distance(
                tr[:, row, col, :-1], fov=fov, exp_time=exp_time
            )  # Compute the distance of the current pixel
    else:
        for i in trange(tr.np.shape[1], desc="computing distance map", leave=False):
            for j in range(tr.np.shape[2]):
                d_map[i, j] = compute_distance(
                    tr[:, i, j, :], fov=fov, exp_time=exp_time
                )  # Compute the distance of the current pixel

    return d_map
