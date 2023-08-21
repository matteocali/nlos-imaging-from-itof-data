import numpy as np
from OpenEXR import InputFile
from Imath import PixelType
from imageio import plugins, imwrite


def load_exr(path):
    """
    Function that load an exr image
    :param path: path of the image
    :return: a numpy matrix containing all the channels ([A, R, G, B] or [R, G, B])
    """

    img = InputFile(str(path))  # Open the file
    n_channels = len(
        img.header()["channels"]
    )  # Extract the number of channels of the image
    dw = img.header()[
        "dataWindow"
    ]  # Extract the data window dimension from the header of the exr file
    size = (
        dw.max.x - dw.min.x + 1,
        dw.max.y - dw.min.y + 1,
    )  # Define the actual size of the image
    if n_channels > 1:
        pt = img.header()["channels"][
            "R"
        ].type  # Recover the pixel type from the header
    else:
        pt = img.header()["channels"][
            "Y"
        ].type  # Recover the pixel type from the header

    # Check if the pt is HALF (pt.v == 1) or FLOAT (pt.v == 2)
    if pt.v == PixelType.HALF:
        np_pt = np.float16  # Define the correspondent value in numpy (np.float16 or np.float32)
    elif pt.v == PixelType.FLOAT:
        np_pt = np.float32
    else:
        np_pt = np.uint8

    if n_channels == 3:
        (R, G, B) = [
            np.frombuffer(img.channel(Chan, pt), dtype=np_pt) for Chan in ("R", "G", "B")
        ]  # Extract the four channel from each image
        if np_pt == np.float16:
            (R, G, B) = [
                data.reshape(size[1], -1).astype(np.float32) for data in [R, G, B]
            ]  # Reshape each vector to match the image size
        else:
            (R, G, B) = [
                data.reshape(size[1], -1) for data in [R, G, B]
            ]  # Reshape each vector to match the image size
        return np.stack((R, G, B), axis=2)
    elif n_channels == 4:
        (A, R, G, B) = [
            np.frombuffer(img.channel(Chan, pt), dtype=np_pt)
            for Chan in ("A", "R", "G", "B")
        ]  # Extract the four channel from each image
        if np_pt == np.float16:
            (A, R, G, B) = [
                data.reshape(size[1], -1).astype(np.float32) for data in [A, R, G, B]
            ]  # Reshape each vector to match the image size
        else:
            (A, R, G, B) = [
                data.reshape(size[1], -1) for data in [A, R, G, B]
            ]  # Reshape each vector to match the image size
        return np.stack((A, R, G, B), axis=2)
    elif n_channels == 1:
        Y = np.frombuffer(
            img.channel("Y", pt), dtype=np_pt
        )  # Extract the channel from each image
        if np_pt == np.float16:
            Y = Y.reshape(size[1], -1).astype(
                np.float32
            )  # Reshape each vector to match the image size
        else:
            Y = Y.reshape(size[1], -1)  # Reshape each vector to match the image size
        return Y
    else:
        return None


def save_exr(img, path):
    """
    Function to save a np array to a .exr image
    :param img: np array ([R, G, B] or [A, R, G, B])
    :param path: path and name of the image to save
    """
    plugins.freeimage.download()  # Download (if needed the required plugin in order to export .exr file)
    imwrite(str(path) + ".exr", img)  # Save the image
