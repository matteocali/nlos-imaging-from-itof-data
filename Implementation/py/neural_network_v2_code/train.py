import sys
import getopt
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils.CustomLosses import BalancedMAELoss, BalancedBCELoss
from utils.NlosNet import NlosNet
from utils.train_functions import train
from utils.utils import format_time, send_email
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
    # Argument defining the learning rate
    arg_lr = 1e-4
    # Argument defining the lambda parameter
    arg_l = 0.2
    # Argument defining the number of the u-net output channels
    arg_n_out_channels = 8
    # Argument defining the number of epochs
    arg_n_epochs = 5000
    # Argument defining if the code will be run on slurm
    arg_slurm = False
    # Help string
    arg_help = "{0} -d <dataset>, -n <name>, -r <lr>, -l <lambda>, -c <n-out-channels>, -e <n-epochs>, -s <slurm>".format(
        argv[0])

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(
            argv[1:], "hd:n:r:l:c:e:s:", ["help", "dataset=", "name=", "lr=", "lambda=", "n-out-channels=", "n-epochs=", "slurm="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            dts_name = arg                 # Set the name of the dataset
        elif opt in ("-n", "--name"):
            arg_model_name = arg           # Set the name of the model
        elif opt in ("-r", "--lr"):
            arg_lr = float(arg)            # Set the learning rate
        elif opt in ("-l", "--lambda"):
            arg_l = float(arg)             # Set the lambda parameter
        elif opt in ("-c", "--n-out-channels"):
            # Set the number of the u-net output channels
            arg_n_out_channels = int(arg)
        elif opt in ("-e", "--n-epochs"):
            arg_n_epochs = int(arg)        # Set the number of epochs
        elif opt in ("-s", "--slurm"):
            if arg.lower() == "true":      # Check if the code is run on singularity
                arg_slurm = True           # Set the singularity flag
            else:
                arg_slurm = False

    print("Dataset name: ", dts_name)
    print("Model name: ", arg_model_name)
    print("Slurm: ", arg_slurm)
    print()

    return [dts_name, arg_model_name, arg_lr, arg_l, arg_n_out_channels, arg_n_epochs, arg_slurm]


if __name__ == '__main__':
    torch.manual_seed(2097710)   # Set the random seed
    start_time = time.time()     # Start the timer
    args = arg_parser(sys.argv)  # Parse the input arguments
    dts_name = args[0]           # Set the path to the csv folder
    batch_size = 32              # Set the batch size
    lr = args[2]                 # Set the learning rate
    l = args[3]                  # Set the lambda parameter
    n_out_channels = args[4]     # Set the number of the u-net output channels
    n_epochs = args[5]           # Set the number of epochs
    slurm = args[6]              # Set the slurm flag

    # Chekc if the gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device, "\n")  # Print the device used

    # Load the processed datset
    if not slurm:
        processed_dts_path = Path(__file__).parent.absolute(
        ) / "datasets" / dts_name / "processed_data"  # Set the path to the processed datasets
    else:
        # Set the path to the processed datasets
        processed_dts_path = Path(__file__).parent.parent.parent.absolute(
        ) / f"datasets/{dts_name}/processed_data"
    # Load the train dataset
    train_dts = torch.load(processed_dts_path / "processed_train_dts.pt")
    # Load the validation dataset
    val_dts = torch.load(processed_dts_path / "processed_validation_dts.pt")

    # Create the dataloaders
    train_loader = DataLoader(train_dts, batch_size=batch_size,
                              shuffle=True, num_workers=4)  # Create the train dataloader
    # Create the validation dataloader
    val_loader = DataLoader(val_dts, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    # Create the network state folder
    # Set the path to the network state folder  # type: ignore
    net_state_path = Path(__file__).parent.absolute() / "net_state"
    # Create the network state folder
    net_state_path.mkdir(parents=True, exist_ok=True)

    # Create the model
    model = NlosNet(enc_channels=(6, 16, 32, 64, 128, 256), dec_channels=(256, 128, 64, 32, 16),
                    num_class=n_out_channels).to(device)  # Create the model and move it to the device

    # Print the model summary
    summary(model, input_size=(batch_size, 6, 320, 240),
            device=str(device), mode="train")
    print("")

    # Create the optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # Create the loss function
    depth_loss_fn = BalancedMAELoss(reduction="weight_mean")
    mask_loss_fn = BalancedBCELoss(reduction="weight_mean")

    # Train the model
    s_train_time = time.time()  # Start the timer for the training
    train(
        attempt_name=args[1],
        net=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        depth_loss_fn=depth_loss_fn,
        mask_loss_fn=mask_loss_fn,
        l=l,
        device=device,
        n_epochs=n_epochs,
        save_path=(net_state_path / f"{args[1]}_model_lr_{lr}_l_{l}_ochannel_{n_out_channels}.pt"))
    f_train_time = time.time()  # Stop the timer for the training
    print(
        f"The total computation time for training the model was {format_time(s_train_time, f_train_time)}\n")

    # Send an email to notify the end of the training
    if not slurm:
        send_email(receiver_email="matteocaly@gmail.com", subject="Training completed",
                   body=f"The \"{args[1]}\" training is over (required time: {format_time(start_time, f_train_time)})")
