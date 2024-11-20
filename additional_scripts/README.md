# ADDITIONAL SCRIPTS

This folder contains all the code except from the Neural Network implementation. All the scripts in here requires the conda environment `mitsuba2.yml` provided in the [environments folder](../tools/conda_environments).

In the following will be presented and describe how each scripts works and how to use them.

## mitsuba_tester.py

`mitsuba_tester` contains the code to test the *Mitsuba Renderer 2* (and all the related forks).\
In particular, it can perform all the followings tasks:

* `cross`: function to perform the cross-section decay test on *Mitsuba 2*. It will return the plot of the results. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path to the input folder (folder containing the *Mitsuba 2* renders in `.exr` format)
  * `-o` or `--output`: the path to the output folder (e.g. **"./plots"**)
  * `-e` or `--exp_time`: the dimension of each temporal bin (in meters, e.g. **0.01**)
  * `-f` or `fov`: the field of view of the camera (e.g. **60**)
  * `-t` or `--task` must be set to **"cross"**
* `distance_plot`: function to produce the distance plot that compares the value with the expected decaying. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path to the input folder (folder containing all the `.txt` files containing the distance measurements for ach channel) (example file: [distance_file_example.txt](../tools/example_files/distance_file_example.txt))
  * `-o` or `--output`: the path to the output folder (e.g. **"./plots"**)
  * `-t` or `--task` must be set to **"distance_plot"**
* `mm_distance_plot`: function to produce the plot that check if *Mitsuba 2* performs or not quantization. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path to the input folder (folder containing all the `.txt` files containing the distance measurements for ach channel) (example file: [distance_file_example.txt](../tools/example_files/distance_file_example.txt))
  * `-o` or `--output`: the path to the output folder (e.g. **"./plots"**)
  * `-t` or `--task` must be set to **"mm_distance_plot"**
* `tot_img_test`:function to perform the test that compares the RGB render to the one obtained summing the transient vector. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path to the input folder (folder containing the *Mitsuba 2* renders in `.exr` format)
  * `-o` or `--output`: the path to the output folder (e.g. **"./plots"**)
  * `-r` or `--rgb`: the path to the RGB render (e.g. **"./rgb.exr"**)
  * `-s` or `--samples`: the number of samples used to render the images (e.g. **10000**)
  * `-d` or `--diff`: the interval to use as the limits for the color bar of the difference plot (e.g. **"0.7,1.3"**, if not specified it will be set automatically)
  * `-l` or `--ratio`: the interval to use as the limits for the color bar of the ratio plot (e.g. **"0.7,1.3"**, if not specified it will be set automatically)
  * `-t` or `--task` must be set to **"tot_img_test"**
* `norm_factor`: function to compute the normalization factor for the *Mitsuba 2* renders. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path to the input folder (folder containing the *Mitsuba 2* renders in `.exr` format)
  * `-o` or `--output`: the path to the output folder (e.g. **"./plots"**)
  * `-r` or `--rgb`: the path to the RGB render (e.g. **"./rgb.exr"**)
  * `-t` or `--task` must be set to **"norm_factor"**
* `plot_norm_factor`: function to plot the normalization factor. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path to the input folder (folder containing the `.txt` files containing the normalization factor for each channel) (example file: [norm_factor_example.txt](../tools/example_files/norm_factor_example.txt))
  * `-o` or `--output`: the path to the output folder (e.g. **"./plots"**)
  * `-r` or `--rgb`: the path to the RGB render (e.g. **"./rgb.exr"**)
  * `-t` or `--task` must be set to **"plot_norm_factor"**

## transient_tools.py

`transient_tools` contains the code to performa all kinds of transient data manipulation and analysis.\
In particular, it can perform all the followings tasks:

* `tr_video`: function to generate the transient video starting from the transient data. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path to the input folder (folder containing the transient *Mitsuba 2* raw data)
  * `-o` or `--output`: the path to the output folder (e.g. **"./out"**)
  * `-t` or `--task` must be set to **"tr_video"**
* `total_img`: function to build the full render image starting from th transient data. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path to the input folder (folder containing the transient *Mitsuba 2* raw data)
  * `-o` or `--output`: the path to the output folder (e.g. **"./out"**)
  * `-s` or `--samples`: the number of samples used to render the images (e.g. **10000**)
  * `-t` or `--task` must be set to **"total_img"**
* `glb_tr_video`: function to generate the transient video of only the global component starting from the transient data. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path to the input folder (folder containing the transient *Mitsuba 2* raw data)
  * `-o` or `--output`: the path to the output folder (e.g. **"./out"**)
  * `-s` or `--samples`: the number of samples used to render the images (e.g. **10000**)
  * `-t` or `--task` must be set to **"glb_tr_video"**
* `hists`: function to plot the transient vector of a given transient raw data (just of one pixel). If the data has only one channel it will generate a single B&W plot otherwise if the input is RGB it will produce a plot for each channel. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path to the input folder (folder containing the transient *Mitsuba 2* raw data)
  * `-o` or `--output`: the path to the output folder (e.g. **"./out"**)
  * `-e` or `--exp_time`: the dimension of the temporal bin (e'g' **0.01**)
  * `-p` or `--pixel`: define which pixel to consider (e.g. **"160,120"**)
  * `-t` or `--task` must be set to **"hists"**
* `hists_glb`: function to plot the transient vector of only the global component of a given transient raw data (just of one pixel). If the data has only one channel it will generate a single B&W plot otherwise if the input is RGB it will produce a plot for each channel. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path to the input folder (folder containing the transient *Mitsuba 2* raw data)
  * `-o` or `--output`: the path to the output folder (e.g. **"./out"**)
  * `-e` or `--exp_time`: the dimension of the temporal bin (e'g' **0.01**)
  * `-p` or `--pixel`: define which pixel to consider (e.g. **"160,120"**)
  * `-t` or `--task` must be set to **"hists_glb"**

## dataset_generator.py

`dataset_generator` contains the code to generate the dataset.\
In particular, it can perform two different tasks:

* generate the standard dataset
* generate the dataset with a grid acquisition pattern (the one required to build the *Fermat flow* ground truth)

In order to generate the standard dataset it is required to follow the following steps:

1. run the `dataset_generator.py` script with the following arguments:
   * `-t` or `--template`: the path to the folder where the `.xml` template files are stored (e.g. **"./template"**). An example of the template files is located in the [templates](../tools/example_files/standard_dataset_sample_files/templates) folder
   * `-b` or `--batch`: the path to the output folder where the `.xml` files will be saved, divided into 8 batch folder (e.g. **"./out"**)
   * `-p` or `--perm`: the path to the folder where store the permutation list, required for generating the *Blender* scene and all the related datasets (e.g. **"./dataset_setup"**)
   * `-d` or `--dataset` the path to the folder where store the recap `.txt` file (e.g. **"./dataset_setup"**)
   * `-s` or `--seed`: the optional random seed (e.g. **2019283**)
2. use the `Blender 2.8` project located in the [blender data](../tools/example_files/standard_dataset_sample_files/blender_data) folder to generate all the meshes of the hidden object and of the walls:
   * open the `Blender 2.8` project and load the `object_generator_rnd.py` script to generate all the random object starting from the basic one following the `tr_rot_list` generated in the previous step (if needed change the path in the script inside `Blender`)
   * after all the object has been generated, using the `mitsuba2-blender-add-on` (located in the [blender add-on](../tools/example_files/standard_dataset_sample_files/blender_addon) folder) export all the object in the mitsuba2 format
   * move the `mesh` folder just generated to the same folder where the `out` folder of point (1) is located
3. in the same folder where are located the `mesh` and `out` folder copy the folder [textures](../tools/example_files/standard_dataset_sample_files/textures), in case the texture could be generated again using the `multi_tools.py` script
4. in the same folder where are located the `mesh` and `out` folder copy the folder [slurm_data](../tools/example_files/standard_dataset_sample_files/slurm_data) and the file [launch_slurm.sh](../tools/example_files/standard_dataset_sample_files/launch_slurm.sh)
5. run the script [`slurm_files_gen.py`](../tools/example_files/standard_dataset_sample_files/slurm_data/template/slurm_files_gen.py)
6. run the script [`launch_slurm.sh`](../tools/example_files/standard_dataset_sample_files/launch_slurm.sh) to generate the dataset using SLURM
   if SLURM is not available it is possible to run `mitsuba` standalone using the following command:
   ```bash
   LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" <path_to_xml_config_file> -o "<output_folder>."
   ```
   the `mitsuba` script should be launched for each `.xml` file by hand, for this reason, in order to speed up and accelerate the process it is suggested to use SLURM

In order to generate the Fermat dataset it is required to follow the following steps:

1. run the `dataset_generator.py` script with the following arguments:
   * `-t` or `--template`: the path to the folder where the `.xml` template files are stored (e.g. **"./template"**). An example of the template files is located in the [templates](../tools/example_files/standard_dataset_sample_files/templates) folder
   * `-b` or `--batch`: the path to the output folder where the `.xml` files will be saved, divided into 8 batch folder (e.g. **"./out"**)
   * `-p` or `--perm`: the path to the folder where store the permutation list, required for generating the *Blender* scene and all the related datasets (e.g. **"./dataset_setup"**)
   * `-d` or `--dataset` the path to the folder where store the recap `.txt` file (e.g. **"./dataset_setup"**)
   * `-s` or `--seed`: the optional random seed (e.g. **2019283**)
   * `-i` or `img` set to *"320,240"*, defines the resolution of the rendered images
   * `-g` or `--grid` set to *"32,24"*, defines the grid illumination pattern
2. use the `Blender 2.8` project located in the [blender data](../tools/example_files/standard_dataset_sample_files/blender_data) folder to generate all the meshes of the hidden object and of the walls:
   * open the `Blender 2.8` project and load the `object_generator_rnd.py` script to generate all the random object starting from the basic one following the `tr_rot_list` generated in the previous step (if needed change the path in the script inside `Blender`)
   * after all the object has been generated, using the `mitsuba2-blender-add-on` (located in the [blender add-on](../tools/example_files/standard_dataset_sample_files/blender_addon) folder) export all the object in the mitsuba2 format
   * move the `mesh` folder just generated to the same folder where the `out` folder of point (1) is located
3. in the same folder where are located the `mesh` and `out` folder copy the folder [textures](../tools/example_files/fermat_data_samples_files/textures), in case the texture could be generated again using the `test.py` script setting the flag to split the grid to True
4. in the same folder where are located the `mesh` and `out` folder copy the folder [slurm_data](../tools/example_files/standard_dataset_sample_files/slurm_data) and the file [launch_slurm.sh](../tools/example_files/standard_dataset_sample_files/launch_slurm.sh)
5. run the script [`slurm_files_gen.py`](../tools/example_files/standard_dataset_sample_files/slurm_data/template/slurm_files_gen.py)
6. run the script [`launch_slurm.sh`](../tools/example_files/standard_dataset_sample_files/launch_slurm.sh) to generate the dataset using SLURM
   if SLURM is not available it is possible to run `mitsuba` standalone using the following command:
   ```bash
   LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" <path_to_xml_config_file> -o "<output_folder>."
   ```
   the `mitsuba` script should be launched for each `.xml` file by hand, for this reason, in order to speed up and accelerate the process it is suggested to use SLURM
   
## ground_truth_generator.py

`ground_truth_generator` contains the code to generate the ground truth both for the *mirror trick* and the *Fermat flow* network.

In order to generate the *mirror trick* ground truth it is required to follow the following steps:

1. generate the mirror ground truth using `mittsuba2-transient-nlos`:
   1. use the `Blender 2.8` project located in the [blender data](../tools/example_files/standard_dataset_sample_files/blender_data) folder to generate all the meshes of the hidden object and of the walls:
      * open the `Blender 2.8` project and load the `object_generator_rnd.py` script to generate all the random object starting from the basic one following the `tr_rot_list` generated in the previous step (if needed change the path in the script inside `Blender`)
      * now load the `mirror.py`, select all the just generated object and run the script, it will generate all the flipped version of the object
      * after all the object has been generated, using the `mitsuba2-blender-add-on` (located in the [blender add-on](../tools/example_files/standard_dataset_sample_files/blender_addon) folder) export all the object in the mitsuba2 format except from the walls
      * move the `mesh` folder just generated to the same folder where the `out` folder of point (1) is located
   2. in the same folder where are located the `mesh` and `out` folder copy the folder textures, that could be generated using the [multi_tools.py](./multi_tools.py) script as follow:
      ```bash
      multi_tools.py -t spot_bitmap -m <img_resolution> -s <img_resolution> -o <output_folder>
      ```
      example:
      ```bash
      multi_tools.py -t spot_bitmap -m "320,240" -s "320,240" -o ./textures/
      ```
   4. in the same folder where are located the `mesh` folder copy the folder [xml_files_generator](../tools/example_files/gt_mirror_sample_files/xml_files_generator) and the file [launch_render.sh](../tools/example_files/gt_mirror_sample_files/launch_render.sh)
   5. in the folder [data_configuration](../tools/example_files/gt_mirror_sample_files/xml_files_generator/data_configuration) copy the `tr_rot_list` of the dataset
   6. run the script [xml_files_gen.py](../tools/example_files/gt_mirror_sample_files/xml_files_generator/xml_files_gen.py)
   7. run the script [`launch_render.sh`](../tools/example_files/gt_mirror_sample_files/xml_files_generator/launch_render.sh) to generate the dataset using SLURM
      if SLURM is not available it is possible to run `mitsuba` standalone using the following command:
      ```bash
      LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" <path_to_xml_config_file> -o "<output_folder>."
      ```
      the `mitsuba` script should be launched for each `.xml` file by hand, for this reason, in order to speed up and accelerate the process it is suggested to use SLURM
2. run the `ground_truth_generator.py` script with the following arguments:
   * `-g` or `--ground`: the path to the folder where the ground truth raw *Mitsuba* render is saved (e.g. **"./gt_in"**) the folder should contain 8 folder named as "batch01, batch02, ..." and inside each of them there should be one folder for each render
   * `-i` or `--input`: the path to the folder where the dataset raw *Mitsuba* render is saved (e.g. **"./dts_in"**)  the folder should contain 8 folder named as "batch01, batch02, ..." and inside each of them there should be one folder for each render
   * `-o` or `--output` the path where the output processed ground truth will be saved (e.g. **"./gt_out"**) as `.h5` files (one for each render)
   * `-d` or `--dataset`: the path where the output processed dataset will be saved (e.g. **"./dts_out"**) as `.h5` files (one for each render)
   * `-f` or `--final`: the path to the folder where the final dataset combined with the correspondent ground truth will be saved (e.g. **"./final"**) as `.h5` files (one for each render)
   * `-t` or `--type` set to *"mirror"*

In order to generate the *Fermat flow* ground truth it is required to follow the following steps:

1. build the *Fermat flow* dataset as described in the previous section
2. run the `ground_truth_generator.py` script with the following arguments:
   * `-g` or `--ground`: the path to the folder where the ground truth raw *Mitsuba* render is saved (e.g. **"./gt_in"**) the folder should contain 8 folder named as "batch01, batch02, ..." and inside each of them there should be one folder for each render
   * `-i` or `--input`: the path to the folder where the dataset raw *Mitsuba* render is saved (e.g. **"./dts_in"**)  the folder should contain 8 folder named as "batch01, batch02, ..." and inside each of them there should be one folder for each render
   * `-o` or `--output` the path where the output processed ground truth will be saved (e.g. **"./gt_out"**) as `.h5` files (one for each render)
   * `-d` or `--dataset`: the path where the output processed dataset will be saved (e.g. **"./dts_out"**) as `.h5` files (one for each render)
   * `-f` or `--final`: the path to the folder where the final dataset combined with the correspondent ground truth will be saved (e.g. **"./final"**) as `.h5` files (one for each render)
   * `-t` or `--type` set to *"fermat"*

## mirror_scene_rebuilder.py

`mirror_scene_rebuilder` contains the code to generate the point cloud from the output of the network (mainly for the *mirror trick*).\
To use this script run it with the following arguments:

* `-i` or `--input`: the path to the `.h5` file containing the test result from the network (e.g. **"./test_result.h5"**)
* `-o` or `--output`: the path of the folder where the output point cloud will be saved (e.g. **"./out"**)
* `-g` or `--gt`: define if the depth map will be masked using the ground truth or the predicted value, 1 == use ground truth, 0 == use predicted value (e.g. **1**)

## multi_tools.py

The `multi_tools.py` script is a script that can perform some tests functions:

* `spot_bitmap`: this function generate a bitmap useful for the *Fermat flow* algorithm. The bitmap is usually used es the textures applied to the *Mitsuba 2* projector. In order to run it is required to pass the following argument to the command line:
  * `-o` or `--output`: the path where the bitmap will be saved (e.g. **"C:\Users\user\Documents\"**)
  * `-m` or `--img_resolution`: determines the resolution of the bitmap (e.g. **"320,240"**)
  * `-s` or `--spot_size`: determines the pattern of the grid, so the number of dots for each row and column (e.g. **"32,24"**). If it is set to the same value of `--img_resolution` the bitmap will be fully white, useful for projection illumination
  * `-t` or `task` must be set to **"spot_bitmap"**
* `np2mat`: this function convert a numpy array into a *Matlab file*, in particular it loads the transient data (raw *Mitsuba 2* data of a spot illumination) and save them as a `.mat` file compliant with the 'Fermat flow' requirements. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path of the transient to load, the folder must contain a folder for each single illuminated pixel (e.g. **"C:\Users\user\Documents\transient_data"**)
  * `-o` or `--output`: the path where the *Matlab* file will be saved together with a *.npy* file that will contain all the transient data just loaded
  * `-m` or `--img_resolution`: determines the resolution of the render and consequently of the bitmap (e.g. **"320,240"**)
  * `-s` or `--spot_size`: determines the pattern of the illumination grid, so the number of dots for each row and column (e.g. **"32,24"**)
  * `-t` or `task` must be set to **"np2mat"**
* `point_cloud_cleaner`: this function take in input a point cloud (`.ply`) and remove all the points that are outlier using a statistical approach. In order to run it is required to pass the following argument to the command line:
  * `-i` or `--input`: the path of the point cloud to load (e.g. **"C:\Users\user\Documents\point_cloud.ply"**)
  * `-o` or `--output`: the path where the cleaned point cloud will be saved (e.g. **"C:\Users\user\Documents\"**)
  * `-t` or `task` must be set to **"point_cloud_cleaner"**

## mat_parser.py

This scripts aims to convert the `.mat` real data acquisition in `.h5` file compatible eith the synthetic dataset pipeline performed to to process the synthetic data.

In order to use this script it is enough to run it with the following argument: `-i` or `--input` the path to the folder containing the `.mat` files to convert. The script will generate a folder named `h5_files` in the parent folder of the input folder and it will save the converted files inside it.
