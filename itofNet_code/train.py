import sys
import getopt
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
from torchinfo import summary
from modules.net import BalancedMAELoss, NlosNetItof, train
from modules.utils.helpers import *


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
    # Argument defining the encoder channels
    arg_encoder_channels = (32, 64, 128, 256, 512)
    # Argument defining the number of the u-net output channels
    arg_n_out_channels = 16
    # Argument defining the number of additional CNN layers
    arg_additional_layers = 0
    # Argument defining the number of epochs
    arg_n_epochs = 500
    # Argument to set the flag for the data augmentation
    arg_augment_size = 0
    # Argument defining if to use the noisy dts
    arg_noisy = False
    # Argument defining the pretraining path
    arg_pre_train_path = None
    # Argument defining if the code will be run on slurm
    arg_slurm = False
    # Help string
    arg_help = "{0} -d <dataset>, -n <name>, -r <lr>, -i <encoder-channels>, -c <n-out-channels>, -p <additional-layers>, -e <n-epochs>, -P <pre-train>, -a <data-augment-size>, -N <noisy-dts>, -s <slurm>".format(
        argv[0]
    )

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(
            argv[1:],
            "hd:n:r:i:c:p:e:P:a:N:s:",
            [
                "help",
                "dataset=",
                "name=",
                "lr=",
                "encoder-channels=",
                "n-out-channels=",
                "additional-layers=",
                "n-epochs=",
                "pre-train=",
                "data-augment-size=",
                "noisy-dts=",
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
        elif opt in ("-d", "--dataset"):
            dts_name = arg  # Set the name of the dataset
        elif opt in ("-n", "--name"):
            arg_model_name = arg  # Set the name of the model
        elif opt in ("-r", "--lr"):
            arg_lr = float(arg)  # Set the learning rate
        elif opt in ("-i", "--encoder-channels"):
            arg_encoder_channels = tuple(
                int(x) for x in arg.split(", ")
            )  # Set the encoder channels
            if len(arg_encoder_channels) == 0:
                arg_encoder_channels = tuple(
                    int(x) for x in arg.split(",")
                )  # Set the encoder channels
            if len(arg_encoder_channels) != 5:
                print("The encoder channels must be a tuple of 6 integers")
                sys.exit(2)
        elif opt in ("-c", "--n-out-channels"):
            arg_n_out_channels = int(arg)  # Set the number of the u-net output channels
        elif opt in ("-p", "--additional-layers"):
            arg_additional_layers = int(arg)  # Set the number of additional CNN layers
        elif opt in ("-e", "--n-epochs"):
            arg_n_epochs = int(arg)  # Set the number of epochs
        elif opt in ("-P", "--pre-train"):
            arg_pre_train_path = str(arg)  # Set the pre-training path
        elif opt in ("-a", "--data-augment-size"):
            arg_augment_size = int(arg)  # Set the data augmentation batch size
        elif opt in ("-N", "--noisy-dts"):
            if arg.lower() == "true":
                arg_noisy = True  # Set the noisy dts flag
            else:
                arg_noisy = False
        elif opt in ("-s", "--slurm"):
            if arg.lower() == "true":  # Check if the code is run on singularity
                arg_slurm = True  # Set the singularity flag
            else:
                arg_slurm = False

    print("Dataset name: ", dts_name)
    print("Model name: ", arg_model_name)
    print("Learning rate: ", arg_lr)
    print("Encoder channels: ", arg_encoder_channels)
    print("Number of output channels: ", arg_n_out_channels)
    print("Number of additional layers: ", arg_additional_layers)
    print("Number of epochs: ", arg_n_epochs)
    print("Pre-training model: ", arg_pre_train_path)
    print("Data augmentation batch size: ", arg_augment_size)
    print("Noisy dts: ", arg_noisy)
    print("Slurm: ", arg_slurm)
    print()

    return [
        dts_name,
        arg_model_name,
        arg_lr,
        arg_encoder_channels,
        arg_n_out_channels,
        arg_additional_layers,
        arg_n_epochs,
        arg_pre_train_path,
        arg_augment_size,
        arg_noisy,
        arg_slurm,
    ]


if __name__ == "__main__":
    torch.manual_seed(2097710)  # Set the random seed
    np.random.seed(2097710)  # Set the random seed
    start_time = time.time()  # Start the timer
    args = arg_parser(sys.argv)  # Parse the input arguments
    dts_name = args[0]  # Set the path to the csv folder
    batch_size = 32  # Set the batch size
    lr = args[2]  # Set the learning rate
    encoder_channels = args[3]  # Set the encoder channels
    n_out_channels = args[4]  # Set the number of the u-net output channels
    additional_layers = args[5]  # Set the number of additional CNN layers
    n_epochs = args[6]  # Set the number of epochs
    pre_train_path = args[7]  # Set the path to the pre-trained model
    augment = args[8]  # Set the data augmentation flag
    noisy = args[9]  # Set the noisy dts flag
    slurm = args[10]  # Set the slurm flag

    # Chekc if the gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device, "\n")  # Print the device used

    # Load the processed datset
    if not slurm:
        processed_dts_path = (
            Path(__file__).parent.absolute() / "datasets" / dts_name / "processed_data"
        )  # Set the path to the processed datasets
    else:
        # Set the path to the processed datasets
        processed_dts_path = (
            Path(__file__).parent.parent.parent.absolute()
            / f"datasets/{dts_name}/processed_data"
        )
    # Load the augmented dataset
    if augment > 0 and not noisy:
        augment_dts_path = (
            processed_dts_path.parent.absolute() / "augmented_data"
        )  # Set the path to the augmented dataset
        train_dts = torch.load(augment_dts_path / f"augmented_train_dts_{augment}.pt")
    elif noisy:
        noisy_dts_path = processed_dts_path.parent.absolute() / "noisy_data"
        train_dts = torch.load(noisy_dts_path / "noisy_train_dts.pt")
        val_dts = torch.load(noisy_dts_path / "noisy_validation_dts.pt")
    else:
        train_dts = torch.load(processed_dts_path / "processed_train_dts.pt")
    # Load the validation dataset
    if not noisy:
        val_dts = torch.load(processed_dts_path / "processed_validation_dts.pt")

    # Create the dataloaders
    train_loader = DataLoader(
        train_dts, batch_size=batch_size, shuffle=True, num_workers=4
    )  # Create the train dataloader
    # Create the validation dataloader
    val_loader = DataLoader(val_dts, batch_size=batch_size, shuffle=True, num_workers=4)

    # Compute the ratio between the number of background and object pixels
    bg_obj_ratio = train_dts.get_bg_obj_ratio()

    # Create the network state folder
    # Set the path to the network state folder
    net_state_path = Path(__file__).parent.absolute() / "net_state"
    # Create the network state folder
    net_state_path.mkdir(parents=True, exist_ok=True)

    # Get input dimensions
    dims = [
        train_dts[0]["itof_data"].shape[0],
        train_dts[0]["itof_data"].shape[1],
        train_dts[0]["itof_data"].shape[2],
    ]

    # Set the decoder channels as the reversed encoder channels
    decoder_channels = encoder_channels[::-1]

    # Add the input channels to the encoder channels
    encoder_channels = (dims[0], *encoder_channels)

    # Create the model
    model = NlosNetItof(
        enc_channels=encoder_channels,
        dec_channels=decoder_channels,
        num_class=n_out_channels,
        additional_cnn_layers=additional_layers,
    ).to(
        device
    )  # Create the model and move it to the device

    # Load the pre-trained model if required
    if pre_train_path is not None:
        pre_train_path = (
            Path(__file__).parent.absolute() / f"net_state/{pre_train_path}.pt"
        )
        model.load_state_dict(torch.load(pre_train_path, map_location=torch.device(device)))  # type: ignore

    # Print the model summary
    net_summary = summary(
        model,
        input_size=(batch_size, dims[0], dims[1], dims[2]),
        device=str(device),
        mode="train",
    )
    print()

    # Print the info on the dataset
    print("\nTrain dataset size: ", len(train_dts))
    print("Validation dataset size: ", len(val_dts))
    print("Ratio between bg and obj pixels: ", round(bg_obj_ratio, 2))
    print()

    # Save the info on the training on a log file
    with open(net_state_path / f"{args[1]}_log.txt", "w") as f:
        f.write("--- ATTEMPT AND NETWORK SETUP DATA ---\n")
        f.write("Dataset: " + dts_name + "\n")
        f.write("Model name: " + args[1] + "\n")
        f.write("Learning rate: " + str(lr) + "\n")
        f.write("Encoder channels: " + str(encoder_channels) + "\n")
        f.write("Decoder channels: " + str(decoder_channels) + "\n")
        f.write("Number of output channels: " + str(n_out_channels) + "\n")
        f.write("Number of additional CNN layers: " + str(additional_layers) + "\n")
        f.write("Number of epochs: " + str(n_epochs) + "\n")
        f.write("Pre-train model: " + str(pre_train_path) + "\n")
        f.write("Data augmentation: " + str(augment) + "\n")
        f.write("Device: " + str(device) + "\n\n\n")
        f.write("--- NETWORK SUMMARY ---\n")
        f.write(str(net_summary) + "\n\n\n")
        f.write("--- DATASET SUMMARY ---\n")
        f.write("Train dataset size: " + str(len(train_dts)) + "\n")
        f.write("Validation dataset size: " + str(len(val_dts)) + "\n")
        f.write(
            "Ratio between bg and obj pixels: " + str(round(bg_obj_ratio, 2)) + "\n\n\n"
        )
        f.write("--- TRAINING SUMMARY ---\n")

    # Create the optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # Create the loss function
    scale = 1/7
    loss_fn = BalancedMAELoss(
        reduction="mean", pos_weight=torch.Tensor([scale * bg_obj_ratio]).to(device)
    ).to(device)
    # loss_fn = BalancedMAELoss(reduction="mean")

    # Train the model
    s_train_time = time.time()  # Start the timer for the training
    train(
        attempt_name=args[1],
        net=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        n_epochs=n_epochs,
        save_path=(
            net_state_path
            / f"{args[1]}_model_lr_{lr}_ochannel_{n_out_channels}_addlayers_{additional_layers}_aug_{str(augment)}.pt"
        ),
    )
    f_train_time = time.time()  # Stop the timer for the training
    print(
        f"The total computation time for training the model was {format_time(s_train_time, f_train_time)}\n"
    )

    # Send an email to notify the end of the training
    if not slurm:
        send_email(
            receiver_email="YOUR_EMAIL_ADDRESS",
            subject="Training completed",
            body=f'The "{args[1]}" training is over (required time: {format_time(start_time, f_train_time)})',
        )
