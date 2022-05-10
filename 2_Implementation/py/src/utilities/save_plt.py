from matplotlib import pyplot as plt


def save_plt(img, path, alpha):
    """
    Function to save an image as a matplotlib png
    :param alpha: define if the output will use or not the alpha channel (True/False)
    :param img: image to save
    :param path: path and name
    """
    if not alpha:
        plt.imsave(path, img[:, :, :-1])
    else:
        plt.imsave(path, img)
