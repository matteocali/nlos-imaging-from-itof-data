import sys
import getopt
import time
import glob
import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import Compose
from modules.dataset import (
    NlosTransientDatasetItofGt,
    NlosTransientDatasetItofReal,
    ItofNormalize,
    ItofNormalizeWithAddLayer,
    ChangeBgValue,
    RescaleRealData,
    dts_splitter,
)
from modules.utils.helpers import *


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
        param:
            - argv: system arguments
        return:
            - list containing the input and output path
    """

    arg_name = "dts"  # Argument containing the name of the dataset
    arg_data_path = ""  # Argument containing the path where the raw data are located
    arg_shuffle = True  # Argument containing the flag for shuffling the dataset
    arg_bg_value = 0  # Argument containing the background value
    arg_add_layer = True  # Argument defining if the iToF data will contains an additional 20MHz layer
    arg_multi_freq = (
        False  # Argument defining if the dts will use more than 3 frequencies
    )
    arg_augment_size = 0  # Argument defining if the dataset will be augmented
    arg_noise = False  # Argument defining if will be added noise to the dataset
    arg_real_dts = False  # Argument defining if the dataset is a real dataset
    arg_slurm = False  # Argument defining if the code will be run on slurm

    # Help string
    arg_help = "{0} -n <name> -i <input> -b <bg-value> -s <shuffle> -l <add-layer> -f <multi-freqs> -a <data-augment-size> -N <add-noise> -r <real-dts> -S <slurm>".format(
        argv[0]
    )

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(
            argv[1:],
            "hn:i:b:s:l:f:a:N:r:S:",
            [
                "help",
                "name=",
                "input=",
                "bg-value=",
                "shuffle=",
                "add-layer=",
                "multi_freqs=",
                "data-augment-size=",
                "add-noise=",
                "real-dts=",
                "slurm=",
            ],
        )
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # Print the help message
            sys.exit(2)
        elif opt in ("-n", "--name"):
            arg_name = arg  # Set the attempt name
        elif opt in ("-i", "--input"):
            arg_data_path = Path(arg)  # Set the path to the raw data
        elif opt in ("-b", "--bg-value"):
            arg_bg_value = int(arg)  # Set the background value
        elif opt in ("-s", "--shuffle"):
            if arg.lower() == "true":  # Set the shuffle flag
                arg_shuffle = True
            else:
                arg_shuffle = False
        elif opt in ("-l", "--add-layer"):
            if arg.lower() == "true":  # Set the add_layer flag
                arg_add_layer = True
            else:
                arg_add_layer = False
        elif opt in ("-f", "--multi-freqs"):
            if arg.lower() == "true":
                arg_multi_freq = True
            else:
                arg_multi_freq = False
        elif opt in ("-a", "--data-augment-size"):
            arg_augment_size = int(arg)  # Set the data augmentation batch value
        elif opt in ("-N", "--add-noise"):
            if arg.lower() == "true":
                arg_noise = True
            else:
                arg_noise = False
        elif opt in ("-r", "--real-dts"):
            if arg.lower() == "true":
                arg_real_dts = True
            else:
                arg_real_dts = False
        elif opt in ("-s", "--slurm"):
            if arg.lower() == "true":  # Check if the code is run on slurm
                arg_slurm = True  # Set the slurm flag
            else:
                arg_slurm = False

    print("Attempt name: ", arg_name)
    print("Input folder: ", arg_data_path)
    print("Background value: ", arg_bg_value)
    print("Shuffle: ", arg_shuffle)
    print("Add layer: ", arg_add_layer)
    print("Multi freqs: ", arg_multi_freq)
    print("Data augmentation batch size: ", arg_augment_size)
    print("Add noise: ", arg_noise)
    print("Real dataset: ", arg_real_dts)
    print("Slurm: ", arg_slurm)
    print()

    return [
        arg_name,
        arg_data_path,
        arg_bg_value,
        arg_shuffle,
        arg_add_layer,
        arg_multi_freq,
        arg_augment_size,
        arg_noise,
        arg_real_dts,
        arg_slurm,
    ]


if __name__ == "__main__":
    s_total_time = time.time()  # Start the timer for the total execution time
    torch.manual_seed(20797710)  # Set torch random seed
    args = arg_parser(sys.argv)  # Parse the input arguments
    bg_value = args[2]  # Get the background value
    add_layer = args[4]  # Get the add_layer flag
    multi_freqs = args[5]  # Get the multi_freqs flag
    data_augment = args[6]  # Get the data augmentation flag
    data_noise = args[7]  # Get the data noise flag
    real_dts = args[8]  # Get the real dataset flag
    slurm = args[9]  # Get the slurm flag

    if real_dts:
        out_path = (
            Path(__file__).parent.absolute() / "datasets" / args[0] / "processed_data"
        )
        if out_path.exists() and len(list(out_path.glob("*.pt"))) > 0:
            print("The dataset has already been processed - skipping...\n")
            sys.exit(0)
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            # Start the timer for the dataset creation
            s_dts_time = time.time()
            print("Creating the dataset...")
            # Define the frequencies vector
            freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)
            n_freqs = freqs.shape[0]
            # Define the transforms to apply to the dataset
            transforms_elm = []
            if not add_layer:
                transforms_elm.append(ItofNormalize(n_freq=n_freqs))
            else:
                transforms_elm.append(ItofNormalizeWithAddLayer(n_freq=n_freqs))
            if bg_value != 0:
                transforms_elm.append(ChangeBgValue(0, bg_value))
            # transforms_elm.append(MeanClipping())
            transforms_elm.append(RescaleRealData())
            transforms = Compose(transforms_elm)
            # Create the dataset
            dts = NlosTransientDatasetItofReal(
                Path(args[1]), frequencies=freqs, transform=transforms
            )
            # Save the dataset
            torch.save(dts, out_path / "processed_test_dts.pt")
            # End the timer for the dataset creation
            f_dts_time = time.time()
            print(
                f"The total computation time for generating the dataset was {format_time(s_dts_time, f_dts_time)}\n"
            )
            sys.exit(0)

    # Check if the dataset has alreay been splitted
    if not slurm:
        out_path = Path(__file__).parent.absolute() / "datasets" / args[0] / "csv_split"
    else:
        out_path = (
            Path(__file__).parent.parent.parent.absolute()
            / "datasets"
            / args[0]
            / "csv_split"
        )
    if out_path.exists() and len(list(out_path.glob("*.csv"))) > 0:
        print("The dataset has already been splitted - skipping...\n")
    else:
        print("Splitting the dataset...")
        # Create the output folder if it does not exist
        out_path.mkdir(parents=True, exist_ok=True)
        # Split the dataset
        dts_splitter(out_path, data_path=args[1], shuffle=args[3])
        print("Dataset splitted!\n")

    # Build the PyTorch dataset
    # Load the different csv files
    # Load the csv files
    csv_files = glob.glob1(str(out_path), "*.csv")
    # Get the type of the csv file (train, val, test)
    csv_types = [csv_name.split("_")[-1][:-4] for csv_name in csv_files]
    # Get the train csv file  # type: ignore
    train_csv = Path(out_path / csv_files[csv_types.index("train")])
    # Get the validation csv file  # type: ignore
    val_csv = Path(out_path / csv_files[csv_types.index("validation")])
    # Get the test csv file  # type: ignore
    test_csv = Path(out_path / csv_files[csv_types.index("test")])

    # Define the frequencies vector
    if not multi_freqs:
        freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)
        n_freqs = freqs.shape[0]
    else:
        freqs = np.array([i * 1e06 for i in range(10, 201, 10)], dtype=np.float32)
        n_freqs = freqs.shape[0]

    # Create the processed datset if not already created
    # Set the path to the processed datasets  # type: ignore
    processed_dts_path = Path(out_path.parent / "processed_data")
    # Check if the folder already exists and is not empty
    if processed_dts_path.exists() and len(list(processed_dts_path.iterdir())) > 0:
        print("The processed dataset already exists - skipping...\n")
    else:
        # Create the datasets folder
        processed_dts_path.mkdir(parents=True, exist_ok=True)
        # Start the timer for the dataset creation
        s_dts_time = time.time()

        # Define the transforms to apply to the dataset
        transforms_elm = []
        if not add_layer:
            transforms_elm.append(ItofNormalize(n_freq=n_freqs))
        else:
            transforms_elm.append(ItofNormalizeWithAddLayer(n_freq=n_freqs))
        if bg_value != 0:
            transforms_elm.append(ChangeBgValue(0, bg_value))
        transforms = Compose(transforms_elm)

        # Define bool to define if skip val and test
        skip_val = False
        skip_test = False

        # Printing the dimension of the dataset
        print("Dataset dimension (considering only pure data no augmented samples):")
        with open(train_csv, "r") as f:
            length = len(f.readlines())
        print("  - Train: ", length)
        with open(val_csv, "r") as f:
            length = len(f.readlines())
            if length == 0:
                skip_val = True
        print("  - Validation: ", length)
        with open(test_csv, "r") as f:
            length = len(f.readlines())
            if length == 0:
                skip_test = True
        print("  - Test: ", length, "\n")
        del length

        # Create and save the datasets
        print("Creating the training dataset...")
        train_dts = NlosTransientDatasetItofGt(
            Path(args[1]), train_csv, frequencies=freqs, transform=transforms
        )  # Create the train dataset
        # Save the train dataset
        if skip_val or skip_test:
            torch.save(train_dts, processed_dts_path / "processed_test_dts.pt")
        else:
            torch.save(train_dts, processed_dts_path / "processed_train_dts.pt")
        # Create the validation dataset
        if not skip_val:
            print("Creating the validation dataset...")
            val_dts = NlosTransientDatasetItofGt(
                Path(args[1]), val_csv, frequencies=freqs, transform=transforms
            )
            # Save the validation dataset
            torch.save(val_dts, processed_dts_path / "processed_validation_dts.pt")
        if not skip_test:
            # Create the test dataset
            print("Creating the test dataset...")
            test_dts = NlosTransientDatasetItofGt(
                Path(args[1]), test_csv, frequencies=freqs, transform=transforms
            )  # Create the test dataset
            # Save the test dataset
            torch.save(test_dts, processed_dts_path / "processed_test_dts.pt")

        f_dts_time = time.time()  # Stop the timer for the dataset creation
        print(
            f"The total computation time for generating the dataset was {format_time(s_dts_time, f_dts_time)}\n"
        )

    # Create the augmented dataset if required
    if data_augment > 0:
        # Defien the path to the augmented dataset
        augmented_data_path = processed_dts_path.parent.absolute() / "augmented_data"
        augment_data_name = (
            augmented_data_path / f"augmented_train_dts_{data_augment}.pt"
        )
        # Check if the folder already exists and is not empty
        if augment_data_name.exists():
            print("The augmented dataset already exists - skipping...\n")
        else:
            augmented_data_path.mkdir(parents=True, exist_ok=True)  # Create the folder
            # Start the timer for the dataset augmentation
            s_aug_time = time.time()
            print("Augmenting the training dataset...")

            # Load the training dataset if needed
            if "train_dts" not in locals():
                train_dts = torch.load(processed_dts_path / "processed_train_dts.pt")

            # Augment the training dataset
            train_dts.augment_dts(batch_size=data_augment)  # type: ignore
            # Save the augmented training dataset
            torch.save(train_dts, augment_data_name)  # type: ignore
            # End the timer for the dataset augmentation
            f_aug_time = time.time()
            print(
                f"The total computation time for generating the augmented dataset was {format_time(s_aug_time, f_aug_time)}\n"
            )

    # Create the noisy dataset if required
    if data_noise:
        # Defien the path to the noisy dataset
        noisy_data_path = processed_dts_path.parent.absolute() / "noisy_data"
        noisy_train_path = noisy_data_path / f"noisy_train_dts.pt"
        noisy_val_path = noisy_data_path / f"noisy_validation_dts.pt"
        # Check if the folder already exists and is not empty
        if noisy_train_path.exists():
            print("The noisy dataset already exists - skipping...\n")
        else:
            noisy_data_path.mkdir(parents=True, exist_ok=True)  # Create the folder
            # Start the timer for the dataset augmentation
            s_noise_time = time.time()
            print("Adding noise to the training dataset...")

            # Load the training dataset if needed
            if "train_dts" not in locals():
                train_dts = torch.load(processed_dts_path / "processed_train_dts.pt")
            if "val_dts" not in locals():
                val_dts = torch.load(processed_dts_path / "processed_validation_dts.pt")

            # Augment the training dataset avoiding the batch with gaussian noise
            train_dts.augment_dts(batch_size=data_augment, gaussian=False)  # type: ignore
            # Add the noise to the whole dataset
            train_dts.apply_noise(mean=0, std=0.03)  # type: ignore
            # Save the noisy training dataset
            torch.save(train_dts, noisy_train_path)  # type: ignore

            # Add the noise to the validation dataset
            val_dts.apply_noise(mean=0, std=0.03)  # type: ignore
            # Save the noisy validation dataset
            torch.save(val_dts, noisy_val_path)  # type: ignore

            # End the timer for the dataset augmentation
            f_noise_time = time.time()
            print(
                f"The total computation time for generating the noisy dataset was {format_time(s_noise_time, f_noise_time)}\n"
            )

    # Ending total timer
    f_total_time = time.time()

    # Send an email to notify the end of the training
    if not slurm:
        send_email(
            receiver_email="YOUR_EMAIL_ADDRESS",
            subject="Dataset creation completed",
            body=f'The "{args[0]}" dataset is fully processed (required time: {format_time(s_total_time, f_total_time)})',
        )
