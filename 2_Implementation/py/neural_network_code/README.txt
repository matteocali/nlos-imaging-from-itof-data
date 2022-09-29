The following code performs the reconstruction of transient information given
an iToF input at multiple frequencies.

The folder structure is the following:

- dataset_creation  -->  Contains the code needed for creating the
  training, validation and test datasets

- train  -->  Contains all code for training the pipeline. 
              train.py is the main script, the rest of the code is in the src/
              folder

- test   -->  Contains the code for computing and showing the performance of
              the reconstruction. 
              The main file is test.py.
              The rest of the code is in the src/ folder.

- utils  -->  Contains some useful functions that will be used thorought the
              scripts. 



