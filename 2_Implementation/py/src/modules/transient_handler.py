from tqdm import tqdm
from time import time, sleep
from modules import utilities as ut
from OpenEXR import InputFile
from cv2 import VideoWriter, VideoWriter_fourcc, destroyAllWindows, cvtColor, COLOR_RGBA2BGRA
from numpy import empty, shape, where, divide, zeros, copy, save, load, ndarray
from numpy import isnan, nansum, nanargmax, nanmin, nanmax
from numpy import uint8, float32
from modules import exr_handler as exr
from matplotlib import pyplot as plt
from matplotlib import animation


def reshape_frame(files):
    """
    Function that load al the exr file in the input folder and reshape it in order to have three matrices, one for each channel containing all the temporal value
    :param files: list off all the file path to analyze
    :return: list containing the reshaped frames for each channel
    """

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

    for index, file in enumerate(tqdm(files)):  # For each provided file in the input folder
        img = exr.load_exr(file)

        # Perform the reshaping saving the results in frame_i for i in A, R,G , B
        for i in range(size[0]):
            if not mono:
                frame_a[index, :, i] = img[:, i, 0]
                frame_r[index, :, i] = img[:, i, 1]
                frame_g[index, :, i] = img[:, i, 2]
                frame_b[index, :, i] = img[:, i, 3]
            else:
                frame[index, :, i] = img[:, i]

    sleep(0.05)  # Wait a bit to allow a proper visualization in the console
    end = time()
    print("Reshaping concluded in %.2f sec\n" % (round((end - start), 2)))

    if not mono:
        return [frame_a, frame_r, frame_g, frame_b]
    else:
        return frame


def img_matrix(channels):
    """
    Function that from the single channel matrices generate a proper image matrix fusing them
    :param channels: list of the 4 channels (1 if input is mono)
    :return: list of image matrix [R, G, B, A] (Y if mono)
    """

    print("Generating the image files:")
    start = time()  # Compute the execution time

    mono = type(channels) == ndarray  # verify if the input is RGBA or mono

    if not mono:
        print(f"Build the {shape(channels[0])[2]} image matrices:")
        sleep(0.02)
        images = empty([shape(channels[0])[2], shape(channels[0])[0], shape(channels[0])[1], len(channels)], dtype=float32)  # Empty array that will contain all the images
        # Fuse the channels together to obtain a proper [A, R, G, B] image
        for i in tqdm(range(shape(channels[0])[2])):
            images[i, :, :, 0] = channels[1][:, :, i]
            images[i, :, :, 1] = channels[2][:, :, i]
            images[i, :, :, 2] = channels[3][:, :, i]
            images[i, :, :, 3] = channels[0][:, :, i]

            images[i, :, :, :][isnan(channels[0][:, :, i])] = 0  # Remove all the nan value following the Alpha matrix
    else:
        print(f"Build the {shape(channels)[2]} image matrices:")
        sleep(0.02)
        images = empty([shape(channels)[2], shape(channels)[0], shape(channels)[1]], dtype=float32)  # Empty array that will contain all the images
        # Fuse the channels together to obtain a proper [A, R, G, B] image
        for i in tqdm(range(shape(channels)[2])):
            images[i, :, :] = channels[:, :, i]

    end = time()
    print("Images created successfully in %.2f sec\n" % (round((end - start), 2)))

    return images


def total_img(images, out_path=None, normalize=True):
    """
    Function to build the image obtained by sum all the temporal instant of the transient
    :param images: np array containing of all the images
    :param out_path: output path and name
    :param normalize: normalize or not the output
    :return: total image as a numpy matrix
    """

    print("Generate the total image = sum over all the time instants")
    start = time()  # Compute the execution time

    mono = len(images.shape) == 3  # Check id=f the images are Mono or RGBA

    if not mono:
        summed_images = nansum(images[:, :, :, :-1], axis=0)  # Sum all the produced images over the time dimension ignoring the alpha channel
    else:
        summed_images = nansum(images[:, :, :], axis=0)  # Sum all the produced images over the time dimension ignoring the alpha channel

    # Generate a mask matrix that will contain the number of active beans in each pixel (needed to normalize the image)
    mask = zeros(summed_images.shape, dtype=float32)
    for img in images:
        if not mono:
            mask[img[:, :, :-1].nonzero()] += 1
        else:
            mask[img[:, :].nonzero()] += 1
    mask[where(mask == 0)] = 1  # Remove eventual 0 values

    if normalize:
        total_image = divide(summed_images, mask).astype(float32)
    else:
        total_image = summed_images

    if out_path is not None:
        exr.save_exr(total_image, out_path)  # Save the image

    end = time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))

    return total_image


def extract_center_peak(channels):
    """
    Function that extract the position and the value of the first peak in the middle pixel in each channel alpha excluded
    :param channels: list of all the channels
    :return: return a list of two list the firs containing the peak position for each channel the second one the value of each peak
    """

    try:
        max_index_r, max_index_g, max_index_b = [nanargmax(channel, axis=2) for channel in
                                                 channels]  # Find the index of the maximum value in the third dimension
    except ValueError:  # Manage the all NaN situation
        channels_no_nan = []
        for channel in channels:
            temp = channel
            temp[isnan(channel)] = 0
            channels_no_nan.append(temp)
        max_index_r, max_index_g, max_index_b = [nanargmax(channel, axis=2) for channel in channels_no_nan]
    peak_pos = [data[int(data.shape[0] / 2), int(data.shape[1] / 2)] for data in [max_index_r, max_index_g, max_index_b]]  # Extract the position of the maximum value in the middle pixel
    peak_values = [channel[int(channel.shape[0] / 2), int(channel.shape[1] / 2), peak_pos[index]] for index, channel in
                   enumerate(channels)]  # Extract the radiance value in the peak position of the middle pixel
    return [peak_pos, peak_values]


def compute_center_distance(peak_pos, exposure_time):
    """
    Function that take the position of the peak and the exposure time and compute the measured distance in the center pixel
    :param peak_pos: position of the peak in the transient of the center pixel
    :param exposure_time: size of each time bean
    :return: distance value
    """

    return round(((peak_pos * exposure_time) / 2), 4)  # General equation to compute the distance given the peak_position (p) and the exposure_time (e): ((p + 1) * e) / 2
                                                       # We have to add the correction term, - e/2, to compensate for the rounding in the been size: ((p + 1) * e) / 2 - e/2
                                                       # -> ((p + 1) * e) / 2 - e/2 = (pe + e)/2 - e/2 = pe/2 + e/2 - e/2 = pe/2


def normalize_img(img):
    """
    Normalize the image value in the range [0, 1]
    :param img: np aray corresponding to an image
    :return np containing the normalized img
    """

    img[where(img < 0)] = 0
    min_val = nanmin(img)
    max_val = nanmax(img)

    if max_val - min_val != 0:
        img = (img - min_val) / (max_val - min_val)  # Normalize each image in [0, 1] ignoring the alpha channel
    return img


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
            img = normalize_img(img[:, :, : -1])
        elif normalize and mono:
            img[:, :, :-1] = normalize_img(img)

        if alpha or mono:
            frames.append([plt.imshow(img, animated=True)])  # Create each frame
        else:
            frames.append([plt.imshow(img[:, :, :-1], animated=True)])  # Create each frame without the alphamap

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)  # Create the animation
    ani.save(out_path)


def cv2_transient_video(images, out_path, alpha, normalize):
    """
    Function that generate a video of the transient and save it in the cv2 format
    (code from: https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/)
    :param images: numpy array containing all the images
    :param out_path: path where to save the video
    :param alpha: define if it has to use the alpha channel or not
    :param normalize: choose ti perform normalization or not
    """

    mono = len(images.shape) == 3  # Check if the images are Mono or RGBA

    if not mono:
        out = VideoWriter(str(out_path), VideoWriter_fourcc(*"mp4v"), 30, (images[0].shape[1], images[0].shape[0]))  # Create the cv2 video
    else:
        out = VideoWriter(str(out_path), VideoWriter_fourcc(*"mp4v"), 30, (images[0].shape[1], images[0].shape[0]), 0)  # Create the cv2 video

    for img in tqdm(images):
        if normalize and not mono:
            normalize_img(img[:, :, : -1])
        if normalize and mono:
            img = normalize_img(img)

        # Map img to uint8
        img = (255 * img).astype(uint8)

        # Convert the image from RGBA to BGRA
        if not mono:
            img = cvtColor(img, COLOR_RGBA2BGRA)
        
        if alpha or mono:
            out.write(img)  # Populate the video
        elif not alpha and not mono:
            out.write(img[:, :, :-1])  # Populate the video without the alpha channel

    destroyAllWindows()
    out.release()


def transient_video(images, out_path, out_type="cv2", alpha=False, normalize=True):
    """
    Function that generate and save the transient video
    :param images: np.array containing all the images [<n_of_images>, <n_of_rows>, <n_of_columns>, <n_of_channels>]
    :param out_path: output file path
    :param out_type: format of the video output, cv2 or plt
    :param alpha: boolean value that determines if the output video will consider or not the alpha map
    :param normalize: choose ti perform normalization or not
    """

    print("Generating the final video:")
    start = time()  # Compute the execution time

    if out_type == "plt":
        plt_transient_video(images, out_path / "transient.avi", alpha, normalize)
    elif out_type == "cv2":
        cv2_transient_video(images, out_path / "transient.avi", alpha, normalize)
    elif out_type == "both":
        print("Matplotlib version:")
        plt_transient_video(images, out_path / "transient_plt.avi", alpha, normalize)
        print("Opencv version:")
        cv2_transient_video(images, out_path / "transient_cv2.avi", alpha, normalize)

    end = time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))


def rmv_first_reflection(images, file_path=None, store=False):
    """
    Function that given the transient images remove the first reflection leaving only the global component
    :param images: input transient images
    :param file_path: path of the np dataset
    :param store: if you want to save the np dataset (if already saved set it to false in order to load it)
    :return: Transient images of only the global component as a np array
    """

    if file_path and not store:  # If already exists a npy file containing all the transient images load it instead of processing everything again
        return load(str(file_path))

    print("Extracting the first peak (channel by channel):")
    start = time()

    mono = len(images.shape) == 3  # Check id=f the images are Mono or RGBA

    if not mono:
        peaks = [nanargmax(images[:, :, :, channel_i], axis=0) for channel_i in tqdm(range(images.shape[3] - 1))]  # Find the index of the maximum value in the third dimension
    else:
        peaks = nanargmax(images[:, :, :], axis=0)  # Find the index of the maximum value in the third dimension

    # Extract the position of the first zero after the first peak and remove the first reflection
    print("Remove the first peak (channel by channel):")
    sleep(0.1)

    glb_images = copy(images)
    if not mono:
        for channel_i in tqdm(range(images.shape[3] - 1)):
            for pixel_r in range(images.shape[1]):
                for pixel_c in range(images.shape[2]):
                    zeros_pos = where(images[:, pixel_r, pixel_c, channel_i] == 0)[0]
                    valid_zero_indexes = zeros_pos[where(zeros_pos > peaks[channel_i][pixel_r, pixel_c])]
                    if valid_zero_indexes.size == 0:
                        glb_images[:, pixel_r, pixel_c, channel_i] = 0
                    else:
                        glb_images[:int(valid_zero_indexes[0]), pixel_r, pixel_c, channel_i] = 0
    else:
        for pixel_r in tqdm(range(images.shape[1])):
            for pixel_c in range(images.shape[2]):
                zeros_pos = where(images[:, pixel_r, pixel_c] == 0)[0]
                valid_zero_indexes = zeros_pos[where(zeros_pos > peaks[pixel_r, pixel_c])]
                if valid_zero_indexes.size == 0:
                    glb_images[:, pixel_r, pixel_c] = 0
                else:
                    glb_images[:int(valid_zero_indexes[0]), pixel_r, pixel_c] = 0

    end = time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))

    if file_path and store:
        save(str(file_path), glb_images)  # Save the loaded images as a numpy array
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
        files = ut.reed_files(str(img_path), "exr")  # Load the path of all the files in the input folder with extension .exr
        channels = reshape_frame(files)  # Reshape the frame in a standard layout
        images = img_matrix(channels)  # Create the image files
        if store:
            save(str(np_path), images)  # Save the loaded images as a numpy array
        return images


def histo_plt(radiance, exp_time, file_path=None):

    plt_start_pos = [where(radiance[:, channel] != 0)[0][0] - 10 for channel in range(0, 3)]
    plt_end_pos = [where(radiance[:, channel] != 0)[0][-1] + 11 for channel in range(0, 3)]

    radiance[where(radiance < 0)] = 0

    mono = len(radiance.shape) == 1


    unit_of_measure = 1e9  # nano seconds
    unit_of_measure_name = "ns"

    if not mono:
        colors = ["r", "g", "b"]
        colors_name = ["Red", "Green", "Blu"]

        fig, axs = plt.subplots(1, 3, figsize=(24, 6))
        for i in range(radiance.shape[1] - 1):
            axs[i].plot(radiance[plt_start_pos[i]:plt_end_pos[i], i], colors[i])
            axs[i].set_xticks(range(0, len(radiance[plt_start_pos[i]:plt_end_pos[i], i]) + 1, int(len(radiance[plt_start_pos[i]:plt_end_pos[i], i] + 1) / 13)))
            axs[i].set_xticklabels([round(value * exp_time / 3e8 * unit_of_measure, 1) for value in range(plt_start_pos[i], plt_end_pos[i] + 1, int(len(radiance[plt_start_pos[i]:plt_end_pos[i], i] + 1) / 13))], rotation=45)
            axs[i].set_title(f"{colors_name[i]} channel histogram")
            axs[i].set_xlabel(f"Time instants [{unit_of_measure_name}]")
            axs[i].set_ylabel(r"Radiance value [$W/(m^{2}·sr)$]")
            axs[i].grid()
    else:
        plt.plot(radiance[plt_start_pos[0]:plt_end_pos[0]])
        plt.xticks([round(value * exp_time / 3e8 * unit_of_measure, 1) for value in range(plt_start_pos[i], plt_end_pos[i] + 1, 10)], rotation=45)
        plt.xlabel(f"Time instants [{unit_of_measure_name}]")
        plt.ylabel(r"Radiance value [$W/(m^{2}·sr)$]")
        plt.grid()
    
    if file_path is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(str(file_path))
        plt.close()
