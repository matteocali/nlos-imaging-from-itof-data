import sys
import getopt
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from modules.net import NlosNetItof, test
from modules.utils.helpers import format_time


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
        param:
            - argv: system arguments
        return:
            - list containing the input and output path
    """

    # Argument containing the name of the used dataset
    arg_dts_name = ""
    # Argument containing the name of the network to load
    arg_net_name = ""
    # Argument defining the encoder channels
    arg_encoder_channels = (32, 64, 128, 256, 512)
    # Argument defining the number of the u-net output channels
    arg_n_out_channels = 16
    # Argument defining the number of additional CNN layers
    arg_additional_layers = 0
    # Argument containing the bg value
    arg_bg = 0
    # Help string
    arg_help = "{0} -d <dts-name> -m <model> -i <encoder-channels> -c <n-out-channels> -p <additional-layers> -b <bg-value>".format(
        argv[0]
    )

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(
            argv[1:],
            "hd:m:i:c:p:b:",
            [
                "help",
                "dts-name=",
                "model=",
                "encoder-channels=",
                "n-out-channels=",
                "additional-layers=",
                "bg-value=",
            ],
        )
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # Print the help message
            sys.exit(2)
        elif opt in ("-d", "--dts-name"):
            arg_dts_name = arg  # Set thename of the dataset toi use
        elif opt in ("-m", "--model"):
            arg_net_name = arg  # Set the name of the model to load
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
            # Set the number of the u-net output channels
            arg_n_out_channels = int(arg)
        elif opt in ("-p", "--additional-layers"):
            # Set the number of additional CNN layers
            arg_additional_layers = int(arg)
        elif opt in ("-b", "--bg-value"):
            arg_bg = int(arg)  # Set the bg value

    print("Dataset name: ", arg_dts_name)
    print("Model name: ", arg_net_name)
    print("Encoder channels: ", arg_encoder_channels)
    print("Number of output channels: ", arg_n_out_channels)
    print("Number of additional layers: ", arg_additional_layers)
    print("Bg value: ", arg_bg)
    print()

    return [
        arg_dts_name,
        arg_net_name,
        arg_encoder_channels,
        arg_n_out_channels,
        arg_additional_layers,
        arg_bg,
    ]


if __name__ == "__main__":
    torch.manual_seed(2097710)  # Set the random seed
    args = arg_parser(sys.argv)  # Parse the input arguments
    enc_channels = args[2]  # Get the encoder channels
    n_out_channels = args[3]  # Get the number of the u-net output channels
    additional_layers = args[4]  # Get the number of additional CNN layers
    bg = args[5]  # Get the bg value

    # Chekc if the gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device, "\n")  # Print the device used

    # Load the dataset
    dts_path = (
        Path(__file__).parent.absolute()
        / f"datasets/{args[0]}/processed_data/processed_test_dts.pt"
    )  # Get the path to the dataset
    # Load the test dataset
    test_dts = torch.load(dts_path)

    # Create the test dataloader
    test_loader = DataLoader(test_dts, batch_size=1, shuffle=True, num_workers=4)

    # Get input dimensions
    dims = [
        test_dts[0]["itof_data"].shape[0],
        test_dts[0]["itof_data"].shape[1],
        test_dts[0]["itof_data"].shape[2],
    ]

    # Set the decoder channels as the reversed encoder channels
    dec_channels = enc_channels[::-1]

    # Add the input channels to the encoder channels
    enc_channels = (dims[0], *enc_channels)

    # Load the model
    # Get the path to the model state dict
    state_dict_path = Path(__file__).parent.absolute() / f"net_state/{args[1]}.pt"
    model = NlosNetItof(
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        num_class=n_out_channels,
        additional_cnn_layers=additional_layers,
    ).to(
        device
    )  # Create the model and move it to the device
    # Load the model
    model.load_state_dict(
        torch.load(state_dict_path, map_location=torch.device(device))
    )

    # Compute the ratio between the number of background and object pixels
    if "real" not in args[0]:
        bg_obj_ratio = test_dts.get_bg_obj_ratio()
    else:
        bg_obj_ratio = 35.48

    # Create the loss function
    loss_fn = torch.nn.L1Loss(reduction="mean")

    # Define the output path
    out_folder = Path(__file__).parent.absolute() / "results"
    out_path = out_folder / f"{args[0]}__{args[1]}"
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    # Test the model
    s_test_time = time.time()  # Start the test time
    test(
        net=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        bg=bg,
        out_path=out_path,
    )
    e_test_time = time.time()  # End the test time

    # Print the time spent
    print("Test time: ", format_time(s_test_time, e_test_time))
