import sys
import getopt
import time
import torch
import glob
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss
from utils.NlosTransientDataset import NlosTransientDataset
from utils.ItofNormalize import ItofNormalize
from utils.NlosNet import NlosNet
from utils.train_functions import train
from pathlib import Path


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
        param:
            - argv: system arguments
        return: 
            - list containing the input and output path
    """

    # Argument containing the path where the raw data are located
    arg_data_path = ""
    # Argument containing the path where the csv data are located
    arg_csv_path = ""
    # Help string
    arg_help = "{0} -i <input> -c <output>".format(argv[0])

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(
            argv[1:], "hi:c:", ["help", "input=", "csv="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            arg_data_path = Path(arg)  # Set the path to the raw data
        elif opt in ("-c", "--csv"):
            arg_csv_path = Path(arg)   # Set the path to the csv foler

    print("Input folder: ", arg_data_path)
    print("CSV folder: ", arg_csv_path)
    print()

    return [arg_data_path, arg_csv_path]


if __name__ == '__main__':
    torch.manual_seed(2097710)         # Set the random seed
    start_time = time.time()           # Start the timer
    args = arg_parser(sys.argv)        # Parse the input arguments
    data_path = args[0]                # Set the path to the raw data
    csv_path = args[1]                 # Set the path to the csv folder

    # Chekc if the gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device, "\n")  # Print the device used

    # Load the different csv files
    csv_files = glob.glob1(str(csv_path),"*.csv")                         # Load the csv files
    csv_types = [csv_name.split("_")[-1][:-4] for csv_name in csv_files]  # Get the type of the csv file (train, val, test)
    train_csv = Path(csv_path / csv_files[csv_types.index("train")])      # Get the train csv file  # type: ignore
    val_csv = Path(csv_path / csv_files[csv_types.index("validation")])   # Get the validation csv file  # type: ignore
    test_csv = Path(csv_path / csv_files[csv_types.index("test")])        # Get the test csv file  # type: ignore

    # Create or load the processed datset
    processed_dts_path = Path(csv_path.parent / (str(csv_path.name).split("_")[0] + "_processed_datasets"))                # Set the path to the processed datasets  # type: ignore
    if processed_dts_path.exists() and len(list(processed_dts_path.iterdir())) > 0:  # Check if the folder already exists and is not empty
        train_dts = torch.load(processed_dts_path / "processed_train_dts.pt")     # Load the train dataset
        val_dts = torch.load(processed_dts_path / "processed_validation_dts.pt")  # Load the validation dataset
        test_dts = torch.load(processed_dts_path / "processed_test_dts.pt")       # Load the test dataset
    else:
        processed_dts_path.mkdir(parents=True, exist_ok=True)  # Crreate the datasets folder  # type: ignore
        s_dts_time = time.time()                               # Start the timer for the dataset creation

        # Create and save the datasets
        print("Creating the training dataset...")
        train_dts = NlosTransientDataset(Path(data_path), train_csv, transform=ItofNormalize(n_freq=3))  # Create the train dataset
        torch.save(train_dts, processed_dts_path / "processed_train_dts.pt")                       # Save the train dataset
        print("Creating the validation dataset...")
        val_dts = NlosTransientDataset(Path(data_path), val_csv, transform=ItofNormalize(n_freq=3))      # Create the validation dataset
        torch.save(val_dts, processed_dts_path / "processed_validation_dts.pt")                    # Save the validation dataset
        print("Creating the test dataset...")
        test_dts = NlosTransientDataset(Path(data_path), test_csv, transform=ItofNormalize(n_freq=3))    # Create the test dataset
        torch.save(test_dts, processed_dts_path / "processed_test_dts.pt")                         # Save the test dataset

        f_dts_time = time.time()  # Stop the timer for the dataset creation
        minutes, seconds = divmod(f_dts_time - s_dts_time, 60)
        hours, minutes = divmod(minutes, 60)
        print("The total computation time for generating the dataset is %d:%02d:%02d \n" % (hours, minutes, seconds))

    # Create the dataloaders
    train_loader = DataLoader(train_dts, batch_size=32, shuffle=True, num_workers=4)  # Create the train dataloader
    val_loader = DataLoader(val_dts, batch_size=32, shuffle=True, num_workers=4)      # Create the validation dataloader
    test_loader = DataLoader(test_dts, batch_size=32, shuffle=True, num_workers=4)    # Create the test dataloader

    # Create the network state folder 
    net_state_path = Path("neural_network_v2_code/net_state")  # Set the path to the network state folder  # type: ignore
    net_state_path.mkdir(parents=True, exist_ok=True)              # Create the network state folder

    # Create the model
    model = NlosNet().to(device)  # Create the model and move it to the device

    # Create the optimizer
    optimizer = Adam(model.parameters(), lr=0.0001)

    # Create the loss function
    loss_fn = L1Loss()

    # Train the model
    train_loss, val_loss =train(
                            net=model, 
                            train_loader=train_loader, 
                            val_loader=val_loader, 
                            optimizer=optimizer, 
                            loss_fn=loss_fn, 
                            device=device, 
                            n_epochs=100,
                            save_path=net_state_path)
