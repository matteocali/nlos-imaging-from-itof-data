import sys
import getopt
import time
import glob
import torch
from pathlib import Path
from utils.dts_splitter import dts_splitter
from utils.NlosTransientDataset import NlosTransientDataset
from utils.CustomTransforms import ItofNormalize, ChangeBgValue
from utils.utils import format_time, send_email
from torchvision.transforms import Compose


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
        param:
            - argv: system arguments
        return: 
            - list containing the input and output path
    """

    arg_name = "dts"     # Argument containing the name of the dataset
    arg_data_path = ""   # Argument containing the path where the raw data are located
    arg_shuffle = True   # Argument containing the flag for shuffling the dataset
    arg_bg_value = 0     # Argument containing the background value
    arg_slurm = False    # Argument defining if the code will be run on slurm
    arg_augment = False  # Argument defining if the dataset will be augmented
    # Help string
    arg_help = "{0} -n <name> -i <input> -b <bg-value> -s <shuffle> -a <data-augment> -n <slurm>".format(
        argv[0])

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(argv[1:], "hn:i:b:s:a:n:", [
                                   "help", "name=", "input=", "bg-value=", "shuffle=", "data-augment=", "slurm="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-n", "--name"):
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
        elif opt in ("-a", "--data-augment"):
            if arg.lower() == "true":  # Check if the dataset will be augmented
                arg_augment = True  # Set the data augmentation flag
            else:
                arg_augment = False
        elif opt in ("-s", "--slurm"):
            if arg.lower() == "true":  # Check if the code is run on slurm
                arg_slurm = True  # Set the slurm flag
            else:
                arg_slurm = False

    print("Attempt name: ", arg_name)
    print("Input folder: ", arg_data_path)
    print("Background value: ", arg_bg_value)
    print("Shuffle: ", arg_shuffle)
    print("Data augmentation: ", arg_augment)
    print("Slurm: ", arg_slurm)
    print()

    return [arg_name, arg_data_path, arg_bg_value, arg_shuffle, arg_augment, arg_slurm]


if __name__ == '__main__':
    s_total_time = time.time()   # Start the timer for the total execution time
    torch.manual_seed(20797710)  # Set torch random seed
    args = arg_parser(sys.argv)  # Parse the input arguments
    bg_value = args[2]           # Get the background value
    data_augment = args[4]       # Get the data augmentation flag
    slurm = args[5]              # Get the slurm flag

    # Check if the dataset has alreay been splitted
    if not slurm:
        out_path = Path(__file__).parent.absolute() / \
            "datasets" / args[0] / "csv_split"
    else:
        out_path = Path(__file__).parent.parent.parent.absolute() / \
            "datasets" / args[0] / "csv_split"
    if out_path.exists() and len(list(out_path.glob("*.csv"))) > 0:
        print("The dataset has already been splitted - skipping...\n")
    else:
        print("Splitting the dataset...")
        # Create the output folder if it does not exist
        out_path.mkdir(parents=True, exist_ok=True)
        # Split the dataset
        dts_splitter(out_path, data_path=args[1], shuffle=args[3])
        print("Dataset splitted!\n")

    # Build the PyTorch dataset
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
        if bg_value != 0:
            transforms = Compose(
                [ItofNormalize(n_freq=3), ChangeBgValue(0, bg_value)])
        else:
            transforms = Compose([ItofNormalize(n_freq=3)])

        # Create and save the datasets
        print("Creating the training dataset...")
        train_dts = NlosTransientDataset(
            Path(args[1]), train_csv, transform=transforms)  # Create the train dataset
        # Save the train dataset
        torch.save(train_dts, processed_dts_path / "processed_train_dts.pt")
        print("Creating the validation dataset...")
        # Create the validation dataset
        val_dts = NlosTransientDataset(
            Path(args[1]), val_csv, transform=transforms)
        # Save the validation dataset
        torch.save(val_dts, processed_dts_path / "processed_validation_dts.pt")
        print("Creating the test dataset...")
        test_dts = NlosTransientDataset(
            Path(args[1]), test_csv, transform=transforms)    # Create the test dataset
        # Save the test dataset
        torch.save(test_dts, processed_dts_path / "processed_test_dts.pt")

        f_dts_time = time.time()  # Stop the timer for the dataset creation
        print(
            f"The total computation time for generating the dataset was {format_time(s_dts_time, f_dts_time)}\n")

    # Create the augmented dataset if required
    if data_augment:
        # Defien the path to the augmented dataset
        augmented_data_path = processed_dts_path.parent.absolute() / "augmented_data"
        # Check if the folder already exists and is not empty
        if augmented_data_path.exists() and len(list(augmented_data_path.iterdir())) > 0:
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
            train_dts.augment_dts(batch_size=80)  # type: ignore
            # Save the augmented training dataset
            torch.save(train_dts, augmented_data_path / "augmented_train_dts.pt")  # type: ignore
            # End the timer for the dataset augmentation
            f_aug_time = time.time()
            print(f"The total computation time for generating the augmented dataset was {format_time(s_aug_time, f_aug_time)}\n")

    # Ending total timer
    f_total_time = time.time()

    # Send an email to notify the end of the training
    if not slurm:
        send_email(receiver_email="matteocaly@gmail.com", subject="Dataset creation completed",
                   body=f"The \"{args[0]}\" dataset is fully processed (required time: {format_time(s_total_time, f_total_time)})")
