from operator import truediv
from pathlib import Path
from os import path, listdir, remove, makedirs, mkdir
from venv import create
from natsort import natsorted
from glob import glob
from math import floor
from tqdm import tqdm, trange

current_folder = Path(path.realpath(__file__)).parent.absolute()
template = Path(current_folder / "template.sh")
out_folder = current_folder.parent.absolute()
script_folder = out_folder / "scripts"
xml_folder = current_folder.parent.absolute().parent.absolute() / "xml_files"

try:
    mkdir(script_folder)
except:
    pass
try:
    mkdir(out_folder / "logs")
except:
    pass

for b_index in range(1, 9):
    try:
        mkdir(out_folder / "logs" / f"batch0{b_index}")
    except:
        pass

for b_index in trange(1, 9, desc="batch"):
    final_folder = script_folder
    try:
        mkdir(final_folder)
    except:
        pass

    batch_folder = xml_folder / f"batch0{b_index}"

    with open(str(template), 'r', newline="\n") as file:
        filedata = file.read()

    filedata = filedata.replace('b_index', str(b_index))

    # Write the file out again
    with open(str(final_folder / f"dataset_batch0{b_index}.sh"), 'w', newline="\n") as file:
        file.write(filedata)
