import sys
import getopt
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss
from torch.nn import BCEWithLogitsLoss
from utils.NlosNet import NlosNet
from utils.train_functions import train
from utils.utils import format_time
from pathlib import Path
from torchinfo import summary


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
        param:
            - argv: system arguments
        return: 
            - list containing the input and output path
    """

    # Argument containing the name of the dataset
    dts_name = ""
    # Argument containing the name of the model
    arg_model_name = ""
    # Help string
    arg_help = "{0} -d <dataset>, -n <name>".format(argv[0])

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(
            argv[1:], "hd:n:", ["help", "dataset=", "name="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            dts_name = arg              # Set the name of the dataset
        elif opt in ("-n", "--name"):
            arg_model_name = arg + "_"  # Set the name of the model

    print("Dataset name: ", dts_name)
    print("Model name: ", arg_model_name)
    print()

    return [dts_name, arg_model_name]


if __name__ == '__main__':
    torch.manual_seed(2097710)         # Set the random seed
    start_time = time.time()           # Start the timer
    args = arg_parser(sys.argv)        # Parse the input arguments
    dts_name = args[0]                 # Set the path to the csv folder
    batch_size = 16                    # Set the batch size
    n_epochs = 5000                    # Set the number of epochs
    lr = 1e-4                          # Set the learning rate

    # Chekc if the gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device, "\n")  # Print the device used

    # Load the processed datset
    processed_dts_path = Path(__file__).parent.absolute() / "datasets" / dts_name / "processed_data")  # Set the path to the processed datasets  # type: ignore
    train_dts = torch.load(processed_dts_path / "processed_train_dts.pt")                              # Load the train dataset
    val_dts = torch.load(processed_dts_path / "processed_validation_dts.pt")                           # Load the validation dataset

    # Create the dataloaders
    train_loader = DataLoader(train_dts, batch_size=batch_size, shuffle=True, num_workers=4)  # Create the train dataloader  # type: ignore
    val_loader = DataLoader(val_dts, batch_size=batch_size, shuffle=True, num_workers=4)      # Create the validation dataloader  # type: ignore

    # Create the network state folder 
    net_state_path = Path("neural_network_v2_code/net_state")  # Set the path to the network state folder  # type: ignore
    net_state_path.mkdir(parents=True, exist_ok=True)          # Create the network state folder

    # Create the model
    model = NlosNet(enc_channels=(6, 16, 32, 64, 128, 256), dec_channels=(256, 128, 64, 32, 16), num_class=8, n_final_layers=3).to(device)  # Create the model and move it to the device

    # Print the model summary
    summary(model, input_size=(batch_size, 6, 320, 240), device=str(device), mode="train")
    print("")

    # Create the optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # Create the loss function
    depth_loss_fn = L1Loss()
    mask_loss_fn = BCEWithLogitsLoss()

    # Train the model
    s_train_time = time.time()  # Start the timer for the training
    best_loss = train(
        net=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        depth_loss_fn=depth_loss_fn,
        mask_loss_fn=mask_loss_fn, 
        l = 0.6,
        device=device, 
        n_epochs=n_epochs, 
        save_path=(net_state_path / f"{args[1]}model.pt"))
    f_train_time = time.time()  # Stop the timer for the training
    print(f"The total computation time for training the model was {format_time(s_train_time, f_train_time)}\n")
