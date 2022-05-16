from modules import transient_handler as tr, utilities as ut
from pathlib import Path
from os import path
import numpy as np


if __name__ == '__main__':
    in_folder = Path("C:/Users/DECaligM/Documents/Mitsuba2 rendering environment/Tests/Cornel box (point light) - example/Transient images/transient_images_small_256")
    out_folder = Path("C:/Users/DECaligM/Desktop/Test transinet")
    out_type = "cv2"

    ut.create_folder(out_folder)  # Create the output folder if not already present

    # If present in the folder, load the np.array
    if not path.exists(out_folder / "np_images.npy"):
        files = ut.reed_files(str(in_folder), "exr")  # Load the path of all the files in the input folder with extension .exr
        channels = tr.reshape_frame(files)  # Reshape the frame in a standard layout
        images = tr.img_matrix(channels)  # Create the image files
        np.save(str(out_folder / "np_images.npy"), images)  # Save the loaded images as a numpy array
    else:  # If already exists a npy file containing all the transient images load it instead of processing everything again
        images = np.load(str(out_folder / "np_images.npy"))

    tr.transient_video(images, out_folder, out_type)  # Generate the video
