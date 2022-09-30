# PY README #

This folder contains all the Python code developed during this project, in particular there are two main folders:
* general_purposes_code: contains all the code that is not related to the neural network
  * `mitsuba_tester`: contains the code to test the *Mitsuba Renderer 2* (and all the related forks)
  * `transient_tools`: contains the code to performa all kinds of transient data manipulation and analysis
  * `dataset_generator`: contains the code to generate the dataset
  * `ground_truth_generator`: contains the code to generate the ground truth both for the *mirror trick* and the *Fermat flow* network
  * `mirror_scene_rebuilder`: contains the code to generate the point cloud from the output of the network (mainly for the *mirror trick*)
* neural_network_code: contains all the code related to the neural network, subdivided into four groups:
  * `dataset_creation`: contains the code to create the dataset for the neural network, starting from the *mirror dataset* it performs the split in train, validation and test set, and for the first two also performs the extraction of the patches
  * `training`: contains the code to train the neural network
  * `test`: contains the code to test the neural network (performs inference)
  * `utils`: contains some utility functions used by the other scripts