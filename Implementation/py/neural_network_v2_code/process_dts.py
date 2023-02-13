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

    arg_name = "dts"                      # Argument containing the name of the dataset
    arg_data_path = ""                    # Argument containing the path where the raw data are located
    arg_shuffle = True                    # Argument containing the flag for shuffling the dataset
    arg_help = "{0} -n <name> -i <input> -s <shuffle>".format(argv[0])  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(argv[1:], "hn:i:s:", ["help", "name=", "input=", "shuffle="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-n", "--name"):
            arg_name = arg  # Set the attempt name
        elif opt in ("-i", "--input"):
            arg_data_path = Path(arg)  # Set the path to the raw data
        elif opt in ("-s", "--shuffle"):
            if arg == "True":  # Set the shuffle flag
                arg_shuffle = True
            else:
                arg_shuffle = False
        
    print("Attempt name: ", arg_name)
    print("Input folder: ", arg_data_path)
    print("Shuffle: ", arg_shuffle)
    print()

    return [arg_name, arg_data_path, arg_shuffle]


if __name__ == '__main__':
    args = arg_parser(sys.argv)  # Parse the input arguments

    # Check if the dataset has alreay been splitted
    out_path = Path(__file__).parent.absolute() / "datasets" / args[0] / "csv_split"
    if out_path.exists() and len(list(out_path.glob("*.csv"))) > 0:
        print("The dataset has already been splitted - skipping...\n")
    else:
        print("Splitting the dataset...")
        # Create the output folder if it does not exist
        out_path.mkdir(parents=True, exist_ok=True)
        # Split the dataset
        dts_splitter(out_path, args[1], args[2])
        print("Dataset splitted!\n")

    # Build the PyTorch dataset
    # Load the different csv files
    csv_files = glob.glob1(str(out_path),"*.csv")                         # Load the csv files
    csv_types = [csv_name.split("_")[-1][:-4] for csv_name in csv_files]  # Get the type of the csv file (train, val, test)
    train_csv = Path(out_path / csv_files[csv_types.index("train")])      # Get the train csv file  # type: ignore
    val_csv = Path(out_path / csv_files[csv_types.index("validation")])   # Get the validation csv file  # type: ignore
    test_csv = Path(out_path / csv_files[csv_types.index("test")])        # Get the test csv file  # type: ignore

    # Create the processed datset if not already created
    processed_dts_path = Path(out_path.parent / "processed_data")                 # Set the path to the processed datasets  # type: ignore
    if processed_dts_path.exists() and len(list(processed_dts_path.iterdir())) > 0:  # Check if the folder already exists and is not empty
        print("The processed dataset already exists")
    else:
        processed_dts_path.mkdir(parents=True, exist_ok=True)  # Crreate the datasets folder
        s_dts_time = time.time()                               # Start the timer for the dataset creation

        # Define the transforms to apply to the dataset
        # transforms = Compose([ItofNormalize(n_freq=3), ChangeBgValue(0, -10)])
        transforms = Compose([ItofNormalize(n_freq=3)])

        # Create and save the datasets
        print("Creating the training dataset...")
        train_dts = NlosTransientDataset(Path(args[1]), train_csv, transform=transforms)  # Create the train dataset
        torch.save(train_dts, processed_dts_path / "processed_train_dts.pt")              # Save the train dataset
        print("Creating the validation dataset...")
        val_dts = NlosTransientDataset(Path(args[1]), val_csv, transform=transforms)      # Create the validation dataset
        torch.save(val_dts, processed_dts_path / "processed_validation_dts.pt")           # Save the validation dataset
        print("Creating the test dataset...")
        test_dts = NlosTransientDataset(Path(args[1]), test_csv, transform=transforms)    # Create the test dataset
        torch.save(test_dts, processed_dts_path / "processed_test_dts.pt")                # Save the test dataset

        f_dts_time = time.time()  # Stop the timer for the dataset creation
        print(f"The total computation time for generating the dataset was {format_time(s_dts_time, f_dts_time)}\n")

        # Send an email to notify the end of the training
        send_email(receiver_email="matteocaly@gmail.com", subject="Dataset creation completed", body=f"The \"{args[0]}\" dataset is fully processed (required time: {format_time(s_dts_time, f_dts_time)})")
