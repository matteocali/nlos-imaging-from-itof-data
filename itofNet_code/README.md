# ITOFNET

The code contained in this folder is used to train and test the neural network. Other than that it also includes the necessary code used to perform the splitting of the dataset into train, validation and test set.
All the scripts in here requires the conda environment `pytorch_nn.yml` provided in the [conda_environments](../tools/conda_environments/) folder.

## Folder structure

The folder structure is the following:

* `datasets`: the processed datasets data;
* `extras`: contains some *.pdf* files with the results of the tests performed on the network;
* `utils`: contains all the auxiliary functions and classes used by the others script;
* `process_dts.py`: contains the code necessary to process the *.h5* files of the raw dataset in the processed dataset, it also performs the training, validation and test split:
  * `-n` or `--name`: optinal argument to specify the name of the dataset to process (e.g. *"dataset_1"*) (default is *dts*),
  * `-i` or `--input`: the path to the folder containing the *.h5* files of the raw dataset (e.g. *"../../1_Data/1_Raw"*),
  * `-b` or `--bg-value`: optional argument to specify the background value of the dataset (default is *0*),
  * `-s` or `--shuffle`: optional argument to specify if the dataset should be shuffled (default is *True*),
  * `-l` or `--add-layer`: optional argument to specify if the iToF input data should have an additional layer at 20Mhz that is normalized on its own (default is *False*),
  * `-f` or `--multi-freqs`: optional argument to specify if the iToF input data should use more than the 3 standard frequencies 20, 50 and 60MHz frequencies (default is *False*),
  * `-a` or `--data-augment-size`: optional argument to specify the number of augmented sample to generate for each one of the augmenting category (default is *0*),
  * `-n` or `--slurm`: optional argument to specify if the script is run on a slurm cluster (default is *False*);
* `test.py`: contains the code to test the network performance on the test dataset:
  * args;
* `train`: contains the code for computing and showing the performance of the reconstruction
(the main file is `test.py`, the rest of the code is in the `src` folder). To run it use the following arguments:
  * `-n` or `--name`: the name of the attempt, it will be used to create the folder where the results will be stored and also to load the correct weights so, it has to match the desired training network(e.g. *"attempt_1"*).
