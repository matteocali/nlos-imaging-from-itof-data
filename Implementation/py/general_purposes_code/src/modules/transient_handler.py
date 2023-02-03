import numpy as np
from tqdm import tqdm, trange
from time import time, sleep
from modules import utilities as ut
from OpenEXR import InputFile
from cv2 import VideoWriter, VideoWriter_fourcc, destroyAllWindows, cvtColor, COLOR_RGB2BGR
from numpy import empty, shape, where, divide, copy, save, load, ndarray, arange, matmul, concatenate, cos, sin, tan, sum as np_sum, zeros, histogram as np_hist, max as np_max, mean
from numpy import isnan, nansum, nanargmax, nanmax, nanmin
from numpy import uint8, float32
from modules import exr_handler as exr
from matplotlib import pyplot as plt
from matplotlib import animation
from math import pi, nan
from math import isnan as m_isnan
from pathlib import Path


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

    dw = InputFile(files[0]).header()['dataWindow']  # Extract the data window dimension from the header of the exr file
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)  # Define the actual size of the image
    mono = len(InputFile(files[0]).header()['channels']) == 1  # Check if the files will be mono or not

    if not mono:
        # Define an empty matrix of size image_height x image_width x temporal_samples for each channel
        frame_a = empty([len(files), size[1], size[0]], dtype=float32)
        frame_r = empty([len(files), size[1], size[0]], dtype=float32)
        frame_g = empty([len(files), size[1], size[0]], dtype=float32)
        frame_b = empty([len(files), size[1], size[0]], dtype=float32)
    else:
        # Define an empty matrix of size image_height x image_width x temporal_samples for each channel
        frame = empty([len(files), size[1], size[0]], dtype=float32)

    for index, file in enumerate(tqdm(files, desc="reshaping frame", leave=False)):  # For each provided file in the input folder
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

    mono = type(channels) == ndarray  # verify if the input is RGBA or mono

    if not mono:
        if verbose:
            print(f"Build the {shape(channels[0])[2]} image matrices:")
            sleep(0.02)
        images = empty([shape(channels[0])[2], shape(channels[0])[0], shape(channels[0])[1], len(channels)], dtype=float32)  # Empty array that will contain all the images
        # Fuse the channels together to obtain a proper [A, R, G, B] image
        for i in trange(shape(channels[0])[2], desc="generating images", leave=False):
            images[i, :, :, 0] = channels[1][:, :, i]
            images[i, :, :, 1] = channels[2][:, :, i]
            images[i, :, :, 2] = channels[3][:, :, i]
            images[i, :, :, 3] = channels[0][:, :, i]

            images[i, :, :, :][isnan(channels[0][:, :, i])] = 0  # Remove all the nan value following the Alpha matrix
            images[i, :, :, :][where(images[i, :, :, :] < 0)] = 0  # Remove all the negative value and set them to 0
    else:
        if verbose:
            print(f"Build the {shape(channels)[2]} image matrices:")
            sleep(0.02)
        images = empty([shape(channels)[2], shape(channels)[0], shape(channels)[1]], dtype=float32)  # Empty array that will contain all the images
        # Fuse the channels together to obtain a proper [A, R, G, B] image
        for i in trange(shape(channels)[2], desc="generating images", leave=False):
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

    mono = len(images.shape) == 3  # Check id=f the images are Mono or RGBA

    if not mono:
        summed_images = nansum(images[:, :, :, :-1], axis=0)  # Sum all the produced images over the time dimension ignoring the alpha channel
    else:
        summed_images = nansum(images[:, :, :], axis=0)  # Sum all the produced images over the time dimension ignoring the alpha channel

    if n_samples is not None:
        normalization_factor = n_samples * 1.7291  # The normalization factor is defined as n_samples * 1.7291
        total_image = divide(summed_images, normalization_factor).astype(float32)  # 17290
    else:
        total_image = summed_images

    if out_path is not None:
        exr.save_exr(total_image, out_path)  # Save the image

    end = time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))

    return total_image


def extract_peak(transient: ndarray) -> tuple:
    """
    Function that provided the transient  values of a pixel, extract the peak value and position
    :param transient: transient values [values, channels]
    :return: (peak_position, peak_values) channel by channel
    """

    mono = len(transient.shape) == 1

    if not mono:
        peak_pos = []
        for i in range(transient.shape[1]):
            if m_isnan(nanmin(transient[:, i])) and m_isnan(nanmax(transient[:, i])):  # Check for all nan case
                peak_pos.append(nan)  # If all the value is nan the peak does not exist, assign nan
            else:
                peak_pos.append(nanargmax(transient[:, i], axis=0))  # Extract the position of the maximum value in the provided transient
        peak_values = [transient[peak_pos[i], i] for i in range(transient.shape[1])]  # Extract the radiance value in the peak position
    else:
        if m_isnan(nanmin(transient)) and m_isnan(nanmax(transient)):  # Check for all nan case
            peak_pos = nan  # If all the value is nan the peak does not exist, assign nan
        else:
            peak_pos = nanargmax(transient, axis=0)  # Extract the position of the maximum value in the provided transient
        peak_values = transient[peak_pos]  # Extract the radiance value in the peak position

    return peak_pos, peak_values


def extract_center_peak(images):
    """
    Function that extract the position and the value of the first peak in the middle pixel in each channel alpha excluded
    :param images: np array containing all the images of the transient [n_beans, n_row, n_col, 3]
    :return: return a list of two list the firs containing the peak position for each channel the second one the value of each peak
    """

    try:
        max_index_r, max_index_g, max_index_b = [nanargmax(images[:, :, :, i], axis=0) for i in range(images.shape[3])]  # Find the index of the maximum value in the temporal dimension
    except ValueError:  # Manage the all NaN situation
        channels_no_nan = []
        for i in range(images.shape[3]):
            temp = images[:, :, :, i]
            temp[isnan(images[:, :, :, i])] = 0
            channels_no_nan.append(temp)
        max_index_r, max_index_g, max_index_b = [nanargmax(channel, axis=2) for channel in channels_no_nan]
    peak_pos = [data[int(data.shape[0] / 2), int(data.shape[1] / 2)] for data in [max_index_r, max_index_g, max_index_b]]  # Extract the position of the maximum value in the middle pixel
    peak_values = [images[peak_pos[i], int(images.shape[1] / 2), int(images.shape[2] / 2), i] for i in range(images.shape[3])]  # Extract the radiance value in the peak position of the middle pixel
    return [peak_pos, peak_values]


def compute_radial_distance(peak_pos: int, exposure_time: float) -> float:
    """
    Function that take the position of the peak and the exposure time and compute the measured distance in the center pixel
    :param peak_pos: position of the peak in the transient of the center pixel
    :param exposure_time: size of each time bean
    :return: distance value
    """

    return round(((peak_pos * exposure_time) / 2), 4)  # General equation to compute the distance given the peak_position (p) and the exposure_time (e): ((p + 1) * e) / 2
                                                       # We have to add the correction term, - e/2, to compensate for the rounding in the been size: ((p + 1) * e) / 2 - e/2
                                                       # -> ((p + 1) * e) / 2 - e/2 = (pe + e)/2 - e/2 = pe/2 + e/2 - e/2 = pe/2


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
            img = ut.normalize_img(img[:, :, : -1])
        elif normalize and mono:
            img[:, :, :-1] = ut.normalize_img(img)

        if alpha or mono:
            frames.append([plt.imshow(img, animated=True)])  # Create each frame
        else:
            frames.append([plt.imshow(img[:, :, :-1], animated=True)])  # Create each frame without the alphamap

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)  # Create the animation
    ani.save(out_path)


def cv2_transient_video(images, out_path, normalize):
    """
    Function that generate a video of the transient and save it in the cv2 format
    (code from: https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/)
    :param images: numpy array containing all the images
    :param out_path: path where to save the video
    :param normalize: flag to set if it is required or not the normalization
    """

    mono = len(images.shape) == 3  # Check if the images are Mono or RGBA

    if not mono:
        out = VideoWriter(str(out_path), VideoWriter_fourcc(*"mp4v"), 30, (images[0].shape[1], images[0].shape[0]))  # Create the cv2 video
    else:
        out = VideoWriter(str(out_path), VideoWriter_fourcc(*"mp4v"), 30, (images[0].shape[1], images[0].shape[0]), 0)  # Create the cv2 video

    for i in trange(images.shape[0]):
        if not mono:
            img = copy(images[i, :, :, :-1])
        else:
            img = copy(images[i, :, :])

        if normalize:
            ut.normalize_img(img)
            img = (255 * img).astype(uint8)  # Map img to uint8

        # Convert the image from RGBA to BGRA
        if not mono:
            img = cvtColor(img, COLOR_RGB2BGR)
        
        out.write(img)  # Populate the video

    destroyAllWindows()
    out.release()


def transient_video(images, out_path, out_type="cv2", alpha=False, normalize=True, name="transient"):
    """
    Function that generate and save the transient video
    :param images: ndarray containing all the images [<n_of_images>, <n_of_rows>, <n_of_columns>, <n_of_channels>]
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


def rmv_first_reflection_transient(transient: ndarray, file_path: Path = None, store: bool = False, verbose: bool = True) -> ndarray:
    """
    Function that given a transient vector set to zero all the information about the direct reflection
    :param transient: input transient images
    :param file_path: path of the np dataset
    :param store: if you want to save the np dataset (if already saved set it to false in order to load it)
    :param verbose: if you want to print the progress
    :return: Transient images of only the global component as a np array
    """

    if file_path and not store:  # If already exists a npy file containing all the transient images load it instead of processing everything again
        return load(str(file_path))

    if verbose:
        print("Extracting the first peak (channel by channel):")
        start = time()

    mono = len(transient.shape) == 1  # Check id=f the images are Mono or RGBA

    if not mono:
        peaks = [nanargmax(transient[:, channel_i], axis=0) for channel_i in trange(transient.shape[1], leave=None)]  # Find the index of the maximum value in the third dimension
    else:
        peaks = nanargmax(transient, axis=0)  # Find the index of the maximum value in the third dimension

    # Extract the position of the first zero after the first peak and remove the first reflection
    if verbose:
        print("Remove the first peak (channel by channel):")
        sleep(0.1)

    glb_images = copy(transient)
    if not mono:
        for channel_i in trange(transient.shape[1], leave=None):
            zeros_pos = where(transient[:, channel_i] == 0)[0]
            valid_zero_indexes = zeros_pos[where(zeros_pos > peaks[channel_i])]
            if valid_zero_indexes.size == 0:
                glb_images[:, channel_i] = 0
            else:
                glb_images[:int(valid_zero_indexes[0]), channel_i] = 0
    else:
        zeros_pos = where(transient == 0)[0]
        valid_zero_indexes = zeros_pos[where(zeros_pos > peaks)]
        if valid_zero_indexes.size == 0:
            glb_images = 0
        else:
            glb_images[:int(valid_zero_indexes[0])] = 0

    if verbose:
        end = time()
        # noinspection PyUnboundLocalVariable
        print("Process concluded in %.2f sec\n" % (round((end - start), 2)))

    if file_path and store:
        save(str(file_path), glb_images)  # Save the loaded images as a numpy array

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

    if file_path and not store:  # If already exists a npy file containing all the transient images load it instead of processing everything again
        return load(str(file_path))

    if verbose:
        print("Extracting the first peak (channel by channel):")
        start = time()

    mono = len(images.shape) == 3  # Check id=f the images are Mono or RGBA

    if not mono:
        peaks = [nanargmax(images[:, :, :, channel_i], axis=0) for channel_i in trange(images.shape[3] - 1)]  # Find the index of the maximum value in the third dimension
    else:
        peaks = nanargmax(images[:, :, :], axis=0)  # Find the index of the maximum value in the third dimension

    # Extract the position of the first zero after the first peak and remove the first reflection
    if verbose:
        print("Remove the first peak (channel by channel):")
        sleep(0.1)

    glb_images = copy(images)
    if not mono:
        for channel_i in trange(images.shape[3] - 1, leave=False):
            for pixel_r in range(images.shape[1]):
                for pixel_c in range(images.shape[2]):
                    zeros_pos = where(images[:, pixel_r, pixel_c, channel_i] == 0)[0]
                    valid_zero_indexes = zeros_pos[where(zeros_pos > peaks[channel_i][pixel_r, pixel_c])]
                    if valid_zero_indexes.size == 0:
                        glb_images[:, pixel_r, pixel_c, channel_i] = 0
                    else:
                        glb_images[:int(valid_zero_indexes[0]), pixel_r, pixel_c, channel_i] = 0
    else:
        for pixel_r in trange(images.shape[1], leave=False):
            for pixel_c in range(images.shape[2]):
                zeros_pos = where(images[:, pixel_r, pixel_c] == 0)[0]
                valid_zero_indexes = zeros_pos[where(zeros_pos > peaks[pixel_r, pixel_c])]
                if valid_zero_indexes.size == 0:
                    glb_images[:, pixel_r, pixel_c] = 0
                else:
                    glb_images[:int(valid_zero_indexes[0]), pixel_r, pixel_c] = 0

    if verbose:
        end = time()
        # noinspection PyUnboundLocalVariable
        print("Process concluded in %.2f sec\n" % (round((end - start), 2)))

    if file_path and store:
        save(str(file_path), glb_images)  # Save the loaded images as a numpy array
        return glb_images
    else:
        return glb_images


def transient_loader(img_path, np_path=None, store=False):
    """
    Function that starting from the raw mitsuba transient output load the transient and reshape it
    :param img_path: path of the transient images
    :param np_path: path of the np dataset
    :param store: boolean value that determines if we want to store the loaded transient in np format
    :return: a np array containing all the transient
    """

    if np_path and not store:  # If already exists a npy file containing all the transient images load it instead of processing everything again
        return load(str(np_path))
    else:
        files = ut.read_files(str(img_path), "exr")  # Load the path of all the files in the input folder with extension .exr
        channels = reshape_frame(files)  # Reshape the frame in a standard layout
        images = img_matrix(channels)  # Create the image files
        if store:
            ut.create_folder(np_path.parent.absolute(), "all")  # Create the output folder if not already present
            save(str(np_path), images)  # Save the loaded images as a numpy array
        return images


def grid_transient_loader(transient_path: Path, np_path: Path = None, store: bool = False) -> ndarray:
    """
    Function that starting from the raw mitsuba transient output load the transient and reshape it
    :param transient_path: path of the transient images
    :param np_path: path of the np dataset
    :param store: boolean value that determines if we want to store the loaded transient in np format
    :return: a np array containing all the transient
    """

    if np_path and not store:  # If already exists a npy file containing all the transient images load it instead of processing everything again
        return load(str(np_path))
    else:
        folder_path = ut.read_folders(folder_path=transient_path, reorder=True)
        dw = InputFile(ut.read_files(str(folder_path[0]), "exr")[0]).header()['dataWindow']  # Extract the data window dimension from the header of the exr file
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)  # Define the actual size of the image
        transient = empty([len(folder_path), size[0], 3])
        print("Loading all the transient data:\n")
        for index, img_path in enumerate(tqdm(folder_path, desc="loading files")):
            files = ut.read_files(str(img_path),
                                  "exr")  # Load the path of all the files in the input folder with extension .exr
            channels = reshape_frame(files, verbose=False)  # Reshape the frame in a standard layout
            images = img_matrix(channels, verbose=False)  # Create the image files
            transient[index, :] = images[:, 0, 0, :-1]
        transient[where(transient < 0)] = 0  # Remove from the transient all the negative data
        if store:
            save(str(np_path), transient)  # Save the loaded images as a numpy array
        print("Loading completed")
        return transient


def histo_plt(radiance: ndarray, exp_time: float, interval: list = None, stem: bool = True, file_path: Path = None):
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
            plt_start_pos = [int(interval[0] * 3e8 / exp_time*1e-9)] * 3
            plt_end_pos = [int(interval[1] * 3e8 / exp_time*1e-9)] * 3
        else:
            plt_start_pos = int(interval[0] * 3e8 / exp_time * 1e-9)
            plt_end_pos = int(interval[1] * 3e8 / exp_time * 1e-9)
    else:
        try:
            if not mono:
                plt_start_pos = [where(radiance[:, channel] != 0)[0][0] - 10 for channel in range(0, 3)]
                plt_end_pos = [where(radiance[:, channel] != 0)[0][-1] + 11 for channel in range(0, 3)]
            else:
                plt_start_pos = where(radiance != 0)[0][0] - 10
                plt_end_pos = where(radiance != 0)[0][-1] + 11
        except IndexError:
            if not mono:
                plt_start_pos = [0] * 3
                plt_end_pos = [len(radiance[:, channel]) for channel in range(0, 3)]
            else:
                plt_start_pos = 0
                plt_end_pos = len(radiance)

    radiance[where(radiance < 0)] = 0

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
                markers, stemlines, baseline = axs[i].stem(range(0, len(radiance[plt_start_pos[i]:plt_end_pos[i], i])),
                                                           radiance[plt_start_pos[i]:plt_end_pos[i], i])
                plt.setp(stemlines, color=colors[i])
                plt.setp(baseline, linestyle="dashed", color="black", linewidth=1, visible=False)
                plt.setp(markers, color=colors[i], markersize=1)
            else:
                axs[i].plot(range(0, len(radiance[plt_start_pos[i]:plt_end_pos[i], i])), radiance[plt_start_pos[i]:plt_end_pos[i], i], color=colors[i])
            axs[i].set_xticks(range(0, len(radiance[plt_start_pos[i]:plt_end_pos[i], i]) + 1, int(len(radiance[plt_start_pos[i]:plt_end_pos[i], i] + 1) / 13)))
            axs[i].set_xticklabels(["{:.2f}".format(round(value * exp_time / 3e8 * unit_of_measure, 2)) for value in range(plt_start_pos[i], plt_end_pos[i] + 1, int(len(radiance[plt_start_pos[i]:plt_end_pos[i], i] + 1) / 13))], rotation=45)
            axs[i].set_title(f"{colors_name[i]} channel histogram")
            axs[i].set_xlabel(f"Time instants [{unit_of_measure_name}]")
            axs[i].set_ylabel(r"Radiance value [$W/(m^{2}·sr)$]")
            axs[i].grid()
        fig.tight_layout()
    else:
        fig = plt.figure()
        if stem:
            markers, stemlines, baseline = plt.stem(range(0, len(radiance[plt_start_pos:plt_end_pos])), radiance[plt_start_pos:plt_end_pos])
            plt.setp(stemlines, color="black")
            plt.setp(baseline, linestyle=" ", color="black", linewidth=1, visible=False)
            plt.setp(markers, color="black", markersize=1)
        else:
            plt.plot(range(0, len(radiance[plt_start_pos:plt_end_pos])), radiance[plt_start_pos:plt_end_pos], color="black")
        plt.xticks(range(0, len(radiance[plt_start_pos:plt_end_pos]) + 1, int(len(radiance[plt_start_pos:plt_end_pos] + 1) / 13)), ["{:.2f}".format(round(value * exp_time / 3e8 * unit_of_measure, 2)) for value in range(plt_start_pos, plt_end_pos + 1, int(len(radiance[plt_start_pos:plt_end_pos] + 1) / 13))], rotation=45)
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


def phi(freqs: ndarray, exp_time: float = 0.01, dim_t: int = 2000) -> ndarray:
    """
    Function to convert dToF output (transient) into iToF measurements (phi values for the different frequencies used)
    :param freqs: target frequency values to be used
    :param exp_time: exposure time (time bin size * c)
    :param dim_t: total number of temporal bins
    :return: matrix phy containing all the phi measurements
    """

    c = 3e8
    #min_t = 0.1 / c
    min_t = 0
    max_t = 2*exp_time / c*dim_t
    step_t = (max_t - min_t) / dim_t
    times = arange(dim_t) * step_t
    phi_arg = 2 * pi * matmul(freqs.reshape(-1, 1), times.reshape(1, -1))
    return concatenate([cos(phi_arg), sin(phi_arg)], axis=0)


def amp_phi_compute(v_in):
    n_fr = int(v_in.shape[-1] / 2)
    # Compute useful additional fields
    amp_in = np.sqrt(v_in[:, :, :n_fr] ** 2 + v_in[:, :, n_fr:] ** 2)
    phi_in = np.arctan2(v_in[:, :, :n_fr], v_in[:, :, n_fr:])
    return amp_in, phi_in


def compute_focal(fov: float, row_pixel: int) -> float:
    """
    Function to compute the focal distance
    :param fov: field of view
    :param row_pixel: number of pixel (= number of column)
    :return: focal distance value
    """
    return row_pixel / (2 * tan(fov / 2))


def compute_distance(transient: ndarray, fov: float, exp_time: float) -> float:
    """
    Function to compute the distance of a specific pixel
    :param transient: transient of the target pixel (no alpha channel)
    :param fov: field of view of the used camera
    :param exp_time: exposure time
    :return: the distance value
    """

    peak_pos, peak_value = extract_peak(transient[:, :])  # Extract the radiance value of the first peak
    radial_dist = compute_radial_distance(peak_pos[nanargmax(peak_value)], exp_time)  # Compute the radial distance
    angle = tan(fov)  # compute the angle from the formula: fov = arctan(distance/2*f)

    return round(radial_dist * cos(angle), 3)  # Convert the radial distance in cartesian distance and return it


def clean_transient_tail(transient: ndarray, n_samples: int) -> ndarray:
    """
    Function that clean the tail of a transient measurements removing all the noise data that are located far from the actual data
    :param transient: single transient vector
    :param n_samples: number of empty samples after which the transient will be set to zero ( ignoring the gap between direct and global)
    :return: the transient vector with a cleaned tail
    """

    mono = len(transient.shape) == 1  # Check if the provided transient is multichannel or not
    peaks, _ = extract_peak(transient)  # Extract the position of the direct component from all the three channels

    if not mono:  # If the transient is multichannel:
        for c_index in range(transient.shape[1]):  # Perform the cleanup channel by channel
            zeros_pos = where(transient[:, c_index] == 0)[0]  # Find all the zeros in the transient
            first_zero_after_peak = zeros_pos[where(zeros_pos > peaks[c_index])][0]  # Find the position of the first zero after the direct component to identify where the direct ends
            non_zero_pos = where(transient[:, c_index] != 0)[0]  # Find where the transient is not set to zero
            non_zero_pos = non_zero_pos[where(non_zero_pos > first_zero_after_peak)]  # Keep only the positions in the global component
            for i in range(1, len(non_zero_pos)):
                if n_samples < non_zero_pos[i] - non_zero_pos[i - 1]:  # Check if there is a gap between two non-zero value greater than <n_samples>
                    transient[non_zero_pos[i]:, c_index] = 0  # If that is the case from that poit on set the transient to zero
                    break  # Break the cycle and move on
            pass
    else:  # If the transient is mono do the same as the multichannel case without the cycle on the channels
        zeros_pos = where(transient == 0)[0]
        first_zero_after_peak = zeros_pos[where(zeros_pos > peaks)][0]
        non_zero_pos = where(transient != 0)[0]
        non_zero_pos = non_zero_pos[where(non_zero_pos > first_zero_after_peak)]
        for i in range(1, len(non_zero_pos)):
            if n_samples < non_zero_pos[i] - non_zero_pos[i - 1] < 200:
                transient[non_zero_pos[i]:] = 0
                break
    return transient


def active_beans_percentage(transient: ndarray) -> float:
    """
    Function that compute the percentage of active beans given a transient vector (single channel)
    :param transient: single channel transient vector
    :return: the percentage value
    """

    non_zero_beans = where(transient != 0)[0]  # Find all the non-zero bins
    return len(non_zero_beans) / len(transient) * 100  # Divide the number of active bins (len(non_zero_beans)) by the total number of bins (len(transient)) and multiply by 100 (in order to obtain a percentage)


def plot_phi(phi_matrix: ndarray, freq_values: ndarray, file_path: Path = None, exp_time: float = 0.01) -> None:
    """
    Function to plot the sine of the phi matrix
    :param phi_matrix: phi matrix data
    :param freq_values: used frequencies values
    :param file_path: file path + name where to save the plot (if not provided the plot will not be saved)
    :param exp_time: exposure_time
    """

    file_path = str(ut.add_extension(str(file_path), ".svg"))  # If necessary add the .svg extension to the file name

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
            axs[r, c].set_xticks(range(0, phi_matrix.shape[1] + 1, 500))  # Put a tick on the x-axis every 500 time bins
            axs[r, c].set_xticklabels(["{:.2f}".format(round(value * exp_time / 3e8 * unit_of_measure, 2)) for value in range(0, phi_matrix.shape[1] + 1, 500)])  # Change the labels of the x-axis in order to display the time delay in the right unit of measure
            if c == 0:
                axs[r, c].title.set_text(f"Cosine at freq.: {np.format_float_scientific(freq_values[r], trim='-', exp_digits=1)} Hz")  # Add a title to each subplot
            else:
                axs[r, c].title.set_text(f"Sine at freq.: {np.format_float_scientific(freq_values[r], trim='-', exp_digits=1)} Hz")  # Add a title to each subplot
            axs[r, c].set_xlabel(f"Time instants [{unit_of_measure_name}]")  # Define the label on the x-axis using the correct unit of measure
            axs[r, c].grid()  # Add the grid
            index += 1
    fig.tight_layout()  # adapt the subplots dimension to the one of the figure

    if file_path is not None:
        plt.savefig(file_path)  # If a path is provided save the plot
    else:
        plt.show()  # Otherwise display it

    plt.close()


def direct_global_ratio(transient: ndarray) -> list:
    """
    Function to compute the global direct ratio
    :param transient: single transient vector
    :return: a list containing the ratio between the global and the direct component for each channel
    """

    mono = len(transient.shape) == 1  # Check if the images are Mono or RGBA

    _, p_value = extract_peak(transient)  # Compute the direct component location
    glb = rmv_first_reflection_transient(transient, verbose=False)  # Extract the global component from the transient data
    glb_sum = np_sum(glb, axis=0)  # Sum oll the global component

    if not mono:
        ratio = []  # Define an empty list that will contain the ratio value for each channel
        for c_index in range(glb_sum.shape[0]):  # For each channel
            ratio.append(glb_sum[c_index] / p_value[c_index])  # Append to the list the ratio value
    else:
        ratio = [glb_sum / p_value]  # Compute the ratio value

    return ratio


def rmv_glb(transient: ndarray) -> ndarray:
    """
    Function to remove the global component from the transient data and leave only the direct component
    :param transient: transient data
    :return: transient data without the global component
    """

    mono = len(transient.shape) == 1  # Check if the images are Mono or RGBA

    if not mono:
        peaks = [nanargmax(transient[:, channel_i], axis=0) for channel_i in range(transient.shape[1])]  # Find the index of the maximum value in the third dimension
    else:
        peaks = nanargmax(transient, axis=0)  # Find the index of the maximum value in the third dimension

    # Extract the position of the first zero after the first peak and remove the first reflection
    direct = copy(transient)  # Copy the transient data
    if not mono:  # If the images are RGBA
        for channel_i in range(transient.shape[1]):  # For each channel
            zeros_pos = where(transient[:, channel_i] == 0)[0]  # Find the index of the zeros
            valid_zero_indexes = zeros_pos[where(zeros_pos > peaks[channel_i])]  # Find the index of the zeros after the first peak
            if valid_zero_indexes.size != 0:  # If there is at least one zero after the first peak
                direct[int(valid_zero_indexes[0]):, channel_i] = 0  # Set the direct component to 0
    else:
        zeros_pos = where(transient == 0)[0]  # Find the index of the zeros
        valid_zero_indexes = zeros_pos[where(zeros_pos > peaks)]  # Find the index of the zeros after the first peak
        if valid_zero_indexes.size != 0:  # If there is at least one zero after the first peak
            direct[int(valid_zero_indexes[0]):] = 0  # Set the direct component to 0

    return direct


def clear_tr_ratio(transient: ndarray, method: str = "otsu") -> ndarray:
    """
    Function to remove all the imprecise transient based on an Otsu thresholding
    :param transient: transient data
    :param method: method used to compute the threshold
    :return: transient data with spurious one set to 0 (only the global is set to zero)
    """

    tr = copy(transient)  # Copy the transient information in order to not override the given one

    ratio = zeros(tr.shape[0])  # Create the ndarray that will contain the ratio data as an array full of zeros
    for i in range(tr.shape[0]):
        ratio[i] = direct_global_ratio(tr[i, :, 1])[0]  # Populate the ratio ndarray computing the ratio value for each transient

    h_data = np_hist(ratio, 100)  # Compute the histogram values of the provided data using 100 bins

    if method == "otsu":
        t_value, threshold = ut.otsu_hist_threshold(h_data)  # Compute the threshold using the Otsu's method
    elif method == "balanced":
        t_value, threshold = ut.balanced_hist_thresholding(h_data)  # Compute the threshold using the balanced method

    # noinspection PyUnboundLocalVariable
    indexes = where(ratio < t_value)[0]  # Find all the transient that does not satisfy the requirements of the threshold

    for index in indexes:  # For each index of interest
        glb = rmv_first_reflection_transient(tr[index, :, 1], verbose=False)  # Extract the global component
        try:
            start = where(glb != 0)[0][0]  # Define the starting location of the global component
            tr[index, start:, :] = zeros([tr[index, start:, 1].shape[0], tr.shape[2]])  # Set the global component to 0
        except IndexError:
            pass  # If the global component is empty, do nothing
    return tr


def clear_tr_energy(transient: ndarray, threshold: int) -> ndarray:
    """
    Function to remove all the imprecise transient based on an energy thresholding
    :param transient: Transient data af the image (as a long list)
    :param threshold: Threshold value (percentage of the maximum energy)
    :return: cleaned transient data
    """

    tr = copy(transient)  # Copy the transient information in order to not override the given one
    glb = zeros(transient.shape)  # Create the ndarray that will contain the global component as an array full of zeros
    glb_sum = zeros(transient.shape[0])  # Create the ndarray that will contain the sum of the global component as an array full of zeros
    max_glb_value = 0  # Create the variable that will contain the maximum value of the global component

    for i in trange(transient.shape[0], desc="cleaning transient"):
        glb[i, :, :] = rmv_first_reflection_transient(transient[i, :, :], verbose=False)  # Extract the global component
        glb_sum[i] = mean(np_sum(glb[i, :, :], axis=1))  # Compute the sum of the global component
        max_glb_value = max(max_glb_value, np_max(glb_sum[i]))  # Find the maximum value of the global component

    for i in range(transient.shape[0]):
        if glb_sum[i] < max_glb_value * (threshold / 100):  # If the sum of the global component is less than 10% of the maximum value
            tr[i, :, :] = rmv_glb(tr[i, :, :])  # Set the global component to 0

    return tr


def compute_distance_map(tr: ndarray, fov: int, exp_time: float) -> ndarray:
    """
    Function to compute the distance map of the image
    :param tr: Transient data af the image (as a numpy array)
    :param fov: Field of view of the camera
    :param exp_time: Exposure time of the camera
    :return: distance map
    """

    d_map = zeros([tr.shape[1], tr.shape[2]])  # Create the ndarray that will contain the distance map as an array full of zeros
    if tr.shape[3] == 4:  # If the image is RGBA
        non_zero = where(np_sum(tr[:, :, :, -1], axis=0) != 0)  # Find the index of the non-zero values in the alpha channel
        for row, col in tqdm(zip(non_zero[0], non_zero[1]), desc="computing distance map", leave=False):  # For each non-zero value
            d_map[row, col] = compute_distance(tr[:, row, col, :-1], fov=fov, exp_time=exp_time)  # Compute the distance of the current pixel
    else:
        for i in trange(tr.shape[1], desc="computing distance map", leave=False):
            for j in range(tr.shape[2]):
                d_map[i, j] = compute_distance(tr[:, i, j, :], fov=fov, exp_time=exp_time)  # Compute the distance of the current pixel

    return d_map