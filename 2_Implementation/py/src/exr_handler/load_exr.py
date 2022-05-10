from OpenEXR import InputFile
from Imath import PixelType
from numpy import frombuffer, stack
from numpy import uint8, float16, float32


def load_exr(path):
    """
    Function that load an exr image
    :param path: path of the image
    :return: a numpy matrix containing all the channels ([A, R, G, B] or [R, G, B])
    """

    img = InputFile(path)  # Open the file
    n_channels = len(img.header()['channels'])  # Extract the number of channels of the image
    dw = img.header()['dataWindow']  # Extract the data window dimension from the header of the exr file
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)  # Define the actual size of the image
    pt = img.header()['channels']['R'].type  # Recover the pixel type from the header

    # Check if the pt is HALF (pt.v == 1) or FLOAT (pt.v == 2)
    if pt.v == PixelType.HALF:
        np_pt = float16  # Define the correspondent value in numpy (float16 or float32)
    elif pt.v == PixelType.FLOAT:
        np_pt = float32
    else:
        np_pt = uint8

    if n_channels == 3:
        (R, G, B) = [frombuffer(img.channel(Chan, pt), dtype=np_pt) for Chan in ("R", "G", "B")]  # Extract the four channel from each image
        if np_pt == float16:
            (R, G, B) = [data.reshape(size[1], -1).astype(float32) for data in [R, G, B]]  # Reshape each vector to match the image size
        else:
            (R, G, B) = [data.reshape(size[1], -1) for data in [R, G, B]]  # Reshape each vector to match the image size
        return stack((R, G, B), axis=2)
    elif n_channels == 4:
        (A, R, G, B) = [frombuffer(img.channel(Chan, pt), dtype=np_pt) for Chan in ("A", "R", "G", "B")]  # Extract the four channel from each image
        if np_pt == float16:
            (A, R, G, B) = [data.reshape(size[1], -1).astype(float32) for data in [A, R, G, B]]  # Reshape each vector to match the image size
        else:
            (A, R, G, B) = [data.reshape(size[1], -1) for data in [A, R, G, B]]  # Reshape each vector to match the image size
        return stack((A, R, G, B), axis=2)
