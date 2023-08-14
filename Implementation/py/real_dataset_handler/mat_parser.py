import numpy as np
from pathlib import Path
from tqdm import tqdm
from modules import itof_utils as itof
from modules import h5_utils as h5


# Constants
OBJECTS = ["cube", "cone", "cylinder", "sphere"]
DISTANCES = [0.7, 1]
FREQUENCIES = range(int(10e6), int(70e6), int(10e6))
ACCEPTED_FREQS = [int(20e6), int(50e6), int(60e6)]
DATA_TYPES = ["amplitude", "depth", "intensity"]
ACCEPTED_DTYPES = ["amplitude", "depth"]


# Define the input path
mat_files_path = Path(
    "/media/matteocali/shared_ssd/NLoS imaging using iToF/real_dts/20230622_real_NLoS_scenes/"
)
mat_files = sorted(list(mat_files_path.glob("*.mat")))

# Define the output path
h5_files_path = mat_files_path.parent / "h5_files"
h5_files_path.mkdir(exist_ok=True, parents=True)

# Split the data in the categoryes
scenes_path = dict()  # Initialize the dictionary for the scenes containing the objects
for obj in OBJECTS:
    scenes_path[obj] = dict()
    for dist in DISTANCES:
        scenes_path[obj][dist] = dict()
        for freq in ACCEPTED_FREQS:
            scenes_path[obj][dist][freq] = dict()
            for data_type in ACCEPTED_DTYPES:
                scenes_path[obj][dist][freq][data_type] = ""

empty_scenes_path = dict()  # Initialize the dictionary for the empty scenes
for freq in ACCEPTED_FREQS:
    empty_scenes_path[freq] = dict()
    for data_type in ACCEPTED_DTYPES:
        empty_scenes_path[freq][data_type] = ""

for mat_file in mat_files:  # Fill the dictionaries
    if not mat_file.stem.startswith("scene_wall"):
        obj, dist, freq, type = mat_file.stem.split("_")[1:]
        dist = float(dist[:-1])
        freq = int(freq[3:] + "000000")
        if freq in ACCEPTED_FREQS:
            scenes_path[obj][dist][freq][type] = mat_file
    else:
        freq, type = mat_file.stem.split("_")[3:]
        freq = int(freq[3:] + "000000")
        if freq in ACCEPTED_FREQS:
            empty_scenes_path[freq][type] = mat_file

# Save the data as separated files for each scene
number_of_obj_scenes = len(OBJECTS) * len(DISTANCES)
pbar = tqdm(total=number_of_obj_scenes, desc="Saving object scenes")
for obj in OBJECTS:
    for dist in DISTANCES:
        pbar.update(1)
        data = dict()
        for i in ACCEPTED_DTYPES:
            data[i] = [] 
        for freq in ACCEPTED_FREQS:
            for data_type in ACCEPTED_DTYPES:
                mat_file = scenes_path[obj][dist][freq][data_type]
                data[data_type].append(h5.load(mat_file)[data_type])
                # Add a row of zeros to the dato to reach shape 320x240
                data[data_type][-1] = np.hstack((data[data_type][-1], np.zeros((320, 1), dtype=np.float32)))
            
        itof_data = itof.depth2itof(np.array(data["depth"]), np.array(ACCEPTED_FREQS), np.array(data["amplitude"]))
        
        depth_gt = np.zeros((320, 240), dtype=np.float32)
        itof_gt = np.zeros((2, 320, 240), dtype=np.float32)
        
        name = f"{obj}_{dist}m.h5"
        out_data = dict()
        out_data["itof_data"] = itof_data
        out_data["depth_gt"] = depth_gt
        out_data["itof_gt"] = itof_gt

        h5.save(data=out_data, path=h5_files_path / name)

# Save the empty scene
data = dict()
for i in ACCEPTED_DTYPES:
        data[i] = []
for freq in ACCEPTED_FREQS:
    for data_type in ACCEPTED_DTYPES:
        mat_file = empty_scenes_path[freq][data_type]
        data[data_type].append(h5.load(mat_file)[data_type])
        # Add a row of zeros to the dato to reach shape 320x240
        data[data_type][-1] = np.hstack((data[data_type][-1], np.zeros((320, 1), dtype=np.float32)))
    
itof_data = itof.depth2itof(np.array(data["depth"]), np.array(ACCEPTED_FREQS), np.array(data["amplitude"]))
pred_depth = itof.itof2depth(itof_data, np.array(ACCEPTED_FREQS))

depth_gt = np.zeros((320, 240), dtype=np.float32)
itof_gt = np.zeros((2, 320, 240), dtype=np.float32)

name = f"empty_scene.h5"
out_data = dict()
out_data["itof_data"] = itof_data
out_data["depth_gt"] = depth_gt
out_data["itof_gt"] = itof_gt

h5.save(data=out_data, path=h5_files_path / name)
