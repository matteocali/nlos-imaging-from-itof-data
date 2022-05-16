from tqdm import tqdm
from time import time, sleep
import OpenEXR
from cv2 import VideoWriter, VideoWriter_fourcc, destroyAllWindows
from numpy import empty, shape, where, divide, zeros, copy, transpose
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

    dw = OpenEXR.InputFile(files[0]).header()[
        'dataWindow']  # Extract the data window dimension from the header of the exr file
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)  # Define the actual size of the image

    # Define an empty matrix of size image_height x image_width x temporal_samples for each channel
    frame_a = empty([len(files), size[1], size[0]], dtype=float32)
    frame_r = empty([len(files), size[1], size[0]], dtype=float32)
    frame_g = empty([len(files), size[1], size[0]], dtype=float32)
    frame_b = empty([len(files), size[1], size[0]], dtype=float32)

    for index, file in enumerate(tqdm(files)):  # For each provided file in the input folder
        img = exr.load_exr(file)

        # Perform the reshaping saving the results in frame_i for i in A, R,G , B
        for i in range(size[0]):
            frame_a[index, :, i] = img[:, i, 0]
            frame_r[index, :, i] = img[:, i, 1]
            frame_g[index, :, i] = img[:, i, 2]
            frame_b[index, :, i] = img[:, i, 3]

    sleep(0.05)  # Wait a bit to allow a proper visualization in the console
    end = time()
    print("Reshaping concluded in %.2f sec\n" % (round((end - start), 2)))

    return [frame_a, frame_r, frame_g, frame_b]


def img_matrix(channels):
    """
    Function that from the single channel matrices generate a proper image matrix fusing them
    :param channels: list of the 4 channels
    :return: list of image matrix [R, G, B, A]
    """

    print("Generating the image files:")
    start = time()  # Compute the execution time

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

    end = time()
    print("Images created successfully in %.2f sec\n" % (round((end - start), 2)))

    return images


def total_img(images, path=None):
    """
    Function to build the image obtained by sum all the temporal instant of the transient
    :param images: np array containing of all the images
    :param path: output path and name
    :return: total image as a numpy matrix
    """
    print("Generate the total image = sum over all the time instants")
    start = time()  # Compute the execution time

    summed_images = nansum(images[:, :, :, :-1], axis=0)  # Sum all the produced images over the time dimension ignoring the alpha channel

    # Generate a mask matrix that will contain the number of active beans in each pixel (needed to normalize the image)
    mask = zeros(summed_images.shape, dtype=float32)
    for img in images:
        mask[img[:, :, :-1].nonzero()] += 1
    mask[where(mask == 0)] = 1  # Remove eventual 0 values

    total_image = divide(summed_images, mask).astype(float32)

    if path is not None:
        exr.save_exr(total_image, path)  # Save the image

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
    peak_pos = [data[int(data.shape[0] / 2), int(data.shape[1] / 2)] for data in [max_index_r, max_index_g,
                                                                                  max_index_b]]  # Extract the position of the maximum value in the middle pixel
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


def plt_transient_video(images, path, alpha):
    """
    Function that generate a video of the transient and save it in the matplotlib format
    (code from: https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib)
    :param images: list of all the transient images
    :param path: path where to save the video
    :param alpha: define if it has to use the alpha channel or not
    """

    frames = []  # For storing the generated images
    fig = plt.figure()  # Create the figure

    for img in tqdm(images):
        if nanmax(img[:, :, : -1]) != 0 and nanmin(img[:, :, : -1]) != 0:
            img[:, :, : -1] = (img[:, :, : -1] - nanmin(img[:, :, : -1])) / (nanmax(img[:, :, : -1]) - nanmin(img[:, :, : -1]))  # Normalize each image in [0, 1] ignoring the alpha channel
        if alpha:
            frames.append([plt.imshow(img, animated=True)])  # Create each frame
        else:
            frames.append([plt.imshow(img[:, :, :-1], animated=True)])  # Create each frame without the alphamap

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)  # Create the animation
    ani.save(path)


def cv2_transient_video(images, out_path, alpha):
    """
    Function that generate a video of the transient and save it in the cv2 format
    (code from: https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/)
    :param images: numpy array containing all the images
    :param out_path: path where to save the video
    :param alpha: define if it has to use the alpha channel or not
    """

    out = VideoWriter(str(out_path), VideoWriter_fourcc(*"mp4v"), 30, (images[0].shape[1], images[0].shape[0]))  # Create the cv2 video

    for img in tqdm(images):
        if nanmax(img[:, :, : -1]) != 0 and nanmin(img[:, :, : -1]) != 0:
            img[:, :, : -1] = (img[:, :, : -1] - nanmin(img[:, :, : -1])) / (nanmax(img[:, :, : -1]) - nanmin(img[:, :, : -1]))  # Normalize each image in [0, 1] ignoring the alpha channel

        # Convert the image from RGBA to BGRA
        tmp = copy(img[:, :, 0])
        img[:, :, 0] = copy(img[:, :, 2])
        img[:, :, 2] = tmp

        # Transpose the image to match the proper resolution
        for i in range(img.shape[-1]):
            tmp = copy(img[:, :, i])
            transpose(tmp)
            img[:, :, i] = copy(tmp)

        # Map img to uint8
        img = (255 * img).astype(uint8)

        if alpha:
            out.write(img)  # Populate the video
        else:
            out.write(img[:, :, :-1])  # Populate the video without the alpha channel

    destroyAllWindows()
    out.release()


def transient_video(images, out_path, out_type="cv2", alpha=False):
    """
    Function that generate and save the transient video
    :param images: np.array containing all the images [<n_of_images>, <n_of_rows>, <n_of_columns>, <n_of_channels>]
    :param out_path: output file path
    :param out_type: format of the video output, cv2 or plt
    :param alpha: boolean value that determines if the output video will consider or not the alpha map
    """

    print("Generating the final video:")
    start = time()  # Compute the execution time

    if out_type == "plt":
        plt_transient_video(images, out_path / "transient.avi", alpha)
    elif out_type == "cv2":
        cv2_transient_video(images, out_path / "transient.avi", alpha)
    elif out_type == "both":
        print("Matplotlib version:")
        plt_transient_video(images, out_path / "transient_plt.avi", alpha)
        print("Opencv version:")
        cv2_transient_video(images, out_path / "transient_cv2.avi", alpha)

    end = time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))
