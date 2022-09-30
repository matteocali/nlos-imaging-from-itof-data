# NEURAL NETWORK CODE README #

The code contained in this folder is used to train and test the neural network. Other than that it also include the necessary code used to perform the splitting of the dataset into train, validation and test set, and to extract the patches from the train and validation set. It also performs the sampling of the patches and all the pre-processing of the data.
All the scripts in here requires the conda environment `itof2dtof.yml` provided in `../../../5_Tools/conda_environments`.\

The folder structure is the following:
* `dataset_creation`: contains the code needed for creating the training, validation and test datasets, to use it run it with the following arguments:
  * `-n` or `--name`: the name of the dataset, it will be used to create the folder where the dataset will be stored (e.g. *"dataset_1"*)
  * `-i` or `--input`: the path to the folder containing the dataset as list of `.h5` files containing both the data and the ground truth
  * `-o` or `--output`: the path to the folder where the processed dataset will be stored, both train, validation and test set as `.h5` files
  * `-s` or `--shuffle`: flag that determine if the dataset should be shuffled before splitting it into train, validation and test set (e.g. *"True"*)
  * `-p` or `--patch`: size of the patch to extract from the dataset (e.g. *"11"*)
  * `-d` or `--n_patches`: flag that determine how many patches to extract from each image (e.g. *800*)
  * `-g` or `--groups`: flag that determine which subset to generate, 100 == train, 010 == validation, 001 == test (e.g. *"111"*)
* `train`: contains all code for training the network (`train.py` is the main script, the rest of the code is in the `src` folder). To run it use the following arguments:
  * `-n` or `--name`: the name of the attempt, it will be used to create the folder where the results will be stored (e.g. *"attempt_1"*)
  * `-r` or `--lr`: the learning rate to use for training (e.g. *"0.001"*)
  * `-t` or `--train`: the name of the dataset to use for training (e.g. *"train_dataset_1"*)
  * `-v` or `--validation`: the name of the dataset to use for validation (e.g. *"val_dataset_1"*)
  * `-f` or `--filter`: the filter size to use for the convolutional layers (e.g. *"32"*)
  * `-l` or `--loss`: the loss function to use for training (options are *"mae"* or *"b_cross_entropy"*)
  * `-s` or `--n_layers`: the number of additional 1x1 layers, default *None*
  * `-b` or `--batch_size`: the batch size to use for training (e.g. *"8192"*)
  * `-e` or `--n_epochs`: the number of epochs to train for (e.g. *"100000"*)
  * `-d` or `--dropout`: the dropout rate to use for training (e.g. *"0.5"*)
  * `-a` or `--alpha_loss_scale`: the scale factor for the alpha loss (e.g. *"2"*)
  * `-p` or `--pretrained`: the path to the pretrained model to use for training (e.g. *"pretrained_model.h5"*)
* `test`: contains the code for computing and showing the performance of the reconstruction (the main file is `test.py`, the rest of the code is in the `src` folder). To run it use the following arguments:
  * `-n` or `--name`: the name of the attempt, it will be used to create the folder where the results will be stored and also to load the correct weights so, it has to match the desired training network(e.g. *"attempt_1"*)
  * `-r` or `--lr`: the learning rate to use for training (e.g. *"0.01"*)
  * `-f` or `--filter_size`: the filter size to use for the convolutional layers (e.g. *"32"*)
  * `-l` or `--loss_fn`: the loss function to use for training (options are *"mae"* or *"b_cross_entropy"*)
  * `-s` or `--n_single_layers`: the number of additional 1x1 layers, default *None*
  * `-t` or `--test_dts`: the name of the dataset to use for testing (e.g. *"test_dataset_1"*)
* `utils`: contains some useful functions that will be used thorough the scripts. 