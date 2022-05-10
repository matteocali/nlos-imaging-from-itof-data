from imageio import plugins, imwrite


def save_exr(img, path):
    """
    Function to save a np array to a .exr image
    :param img: np array ([R, G, B] or [A, R, G, B])
    :param path: path and name of the image to save
    """
    plugins.freeimage.download()  # Download (if needed the required plugin in order to export .exr file)
    imwrite(path, img)  # Save the image
