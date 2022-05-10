from numpy import uint8
from cv2 import imwrite, cvtColor, COLOR_RGBA2BGRA, COLOR_RGB2BGR


def save_png(img, path, alpha):
    """
    Function to save an image as a png
    :param alpha: define if the output will use or not the alpha channel (True/False)
    :param img: image to save
    :param path: path and name
    """
    img = (255 * img).astype(uint8)  # Rescale the input value from [0, 1] to [0, 255] and convert them to unit8
    if alpha:
        imwrite(str(path), cvtColor(img, COLOR_RGBA2BGRA))  # Save the image
    else:
        imwrite(str(path), cvtColor(img[:, :, :-1], COLOR_RGB2BGR))  # Save the image
