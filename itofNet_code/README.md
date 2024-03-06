# iToFNet

The code contained in this folder is used to train and test the neural network architecture proposed **iToFNet**. Other than that it also includes all the necessary scripts to process the acquired dataset (data augmentation, splitting etc.)

All the scripts in here require the conda environment `pytorch_nn.yml` provided in the [conda_environments](../tools/conda_environments/) folder. The only exception is [point_cloud_gen.py](./point_cloud_gen.py) script that requires the `mitsuba.yml` environment provided in the same folder.

## Folder structure

The folder structure is the following:

* [datasets](./datasets): folder where to store all the processed datasets;
* [modules](./modules): folder where all the additional Python modules are stored;
* [net_state](./net_state): folder where to store the weights of the trained networks;
* All the main Python scripts.

## How to run the code

### Pre-process the dataset

The first step is to process the dataset in order to make it compatible with what the network expects. To do so, the [process_dts.py](./process_dts.py) script is used. It takes as input the path to the folder containing the `.h5` files of the raw dataset and it outputs a set of `.pt` files containing the processed dataset (saved in the [datsets](./datasets/) folder). The script also performs data augmentations and the training, validation and test split. The script handles the following arguments:

* `-n` or `--name`: optinal argument to specify the name of the processed dataset (e.g. `'dts_add_layer_aug'`) (default is `'dts'`),
* `-i` or `--input`: the path to the folder containing the `.h5` files of the raw dataset (e.g. `'../../1_Data/1_Raw'`),
* `-b` or `--bg-value`: optional argument to specify the background value of the dataset (default is `0`),
* `-s` or `--shuffle`: optional argument to specify if the dataset should be shuffled (default is `True`),
* `-l` or `--add-layer`: optional argument to specify if the iToF input data should have an additional layer containing only the acquisition at *20Mhz* that is normalized on its own (default is `True`),
* `-f` or `--multi-freqs`: optional argument to specify if the iToF input data should use more than the 3 standard frequencies *20*, *50* and *60MHz* (default is `False`),
* `-a` or `--data-augment-size`: optional argument to specify the number of augmented samples to generate for each one of the augmenting categories (default is `0` = no augmentation),
* `-N` or `--add-noise`: optional argument to specify if the iToF input data should be augmented with strong Gaussian noise (default is `False`),
* `-r` or `--real-dts`: optional argument to specify if the input raw iToF data is coming from a real sensor instead of a synthetic one (default is `False`),
* `-n` or `--slurm`: optional argument to specify if the script is run on a Slurm cluster (default is `False`);

Example of usage:

```python
python process_dts.py \
  --name dts_add_layer_aug \
  --input '../NLoS imaging using iToF/synthetic_dts' \
  --data-augment-size 403 \
  --add-noise False \
  --real-dts False \
  --slurm False"
```

### Train the network

The second step is to train the network. To do so, the [train.py](./train.py) script is used. It takes as input the path to the folder containing the pre-process `.pt` files of the dataset and it outputs a `.pt` file containing the best weights of the trained network. The script handles the following arguments:

* `-d` or `--dataset`: the name of the processed dataset (e.g. `'dts_add_layer_aug'`) (default is `'dts'`),
* `-n` or `--name`: the name of the attempt (e.g. `'attempt_1'`),
* `-r` or `--lr`: optional argument to specify the learning rate (default is `0.0001`),
* `-i` or `--encoder-channels`: optional argument to specify the number of channels of the encoder (default is `'32, 64, 128, 256, 512'`),
* `-c` or `--n-out-channels`: optional argument to specify the number of output channels of the U-Net block of the network (default is `16`),
* `-p` or `--additional-layers`: optional argument to specify how many additional convolutional layer the network should append to the final convolution block of the proposed architecture (default is `0`),
* `-e` or `--n-epochs`: optional argument to specify the number of epochs (default is `500`),
* `-P` or `--pre-train`: optional argument to specify if the network should load a pre-trained model, it should contain the path to the correspondent `.pt` weights (default is `None` = no pre-trained model),
* `-a` or `--data-augment-size`: define the number of augmented samples that have been generated for each one of the augmenting categories,
* `-N` or `--noisy-dts`: optional argument to specify if to the input raw iToF data has been applied a strong Gaussian noise or not (default is `False`),
* `-s` or `--slurm`: optional argument to specify if the script is run on a Slurm cluster (default is `False`).

Example of usage:

```python
python train.py \
  --dataset 'dts_add_layer_aug' \
  --name 'attempt_1' \
  --lr 0.0001 \
  --n-epochs 1500 \
  --pre-train None \
  --data-augment-size 403 \
  --noisy-dts False \
  --slurm False
```

### Test the network

After the model has been trained it is possible to test the network performance on the test dataset. To do so, the [test.py](./test.py) script is used. It takes as input the path to the folder containing the pre-process `.pt` files of the dataset and the path to the `.pt` file containing the weights of the trained network. The script handles the following arguments:

* `-d` or `--dts-name`: define the name of the processed dataset that will be loaded (e.g. `'dts_add_layer_aug'`),
* `-m` or `--model`: define the name of the model that will be loaded (e.g. `'attempt_1'`),
* `-i` or `--encoder-channels`: optional argument to specify the number of channels of the encoder (default is `'32, 64, 128, 256, 512'`),
* `-c` or `--n-out-channels`: optional argument to specify the number of output channels of the U-Net block of the network (default is `16`),
* `-p` or `--additional-layers`: optional argument to specify how many additional convolutional layer the network should append to the final convolution block of the proposed architecture (default is `0`),
* `-b` or `--bg-value`: optional argument to specify the background value of the dataset (default is `0`).

Example of usage:

```python
python test.py \
  --dts-name 'dts_add_layer_aug' \
  --model 'attempt_1'
```

### Generate point clouds

As a final step, it is possible to generate point clouds from the depth images. To do so, the [point_cloud_gen.py](./point_cloud_gen.py) script is used. It takes as input the output of the test script and returns the corresponding point cloud for each sample. The script handles the following arguments:

* `-i` or `--input`: the path to the folder containing the `.h5` files of the raw dataset (e.g. `'./results/attempt_1'`),
* `-o` or `--output`: the path to the folder where to store the point clouds (e.g. `'./results/attempt_1/point_clouds'`).

Example of usage:

```python
python point_cloud_gen.py \
  --input './results/attempt_1' \
  --output './results/attempt_1/point_clouds'
```
