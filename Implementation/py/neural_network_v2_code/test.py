import sys
import getopt
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils.NlosNet import NlosNet
from utils.test_function import test
from utils.utils import format_time
from utils import CustomLosses as CL


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
    # Argument containing the bg value
    arg_bg = 0
    # Help string
    arg_help = "{0} -d <dts-name> -m <model> -b <bg-value>".format(argv[0])

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(
            argv[1:], "hd:m:b:", ["help", "dts-name=", "model=", "bg-value="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--dts-name"):
            arg_dts_name = arg           # Set thename of the dataset toi use
        elif opt in ("-m", "--model"):
            arg_net_name = arg           # Set the name of the model to load
        elif opt in ("-b", "--bg-value"):
            arg_bg = int(arg)            # Set the bg value

    print("Dataset name: ", arg_dts_name)
    print("Model name: ", arg_net_name)
    print("Bg value: ", arg_bg)
    print()

    return [arg_dts_name, arg_net_name, arg_bg]


if __name__ == '__main__':
    torch.manual_seed(2097710)   # Set the random seed
    args = arg_parser(sys.argv)  # Parse the input arguments
    
    # Chekc if the gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device, "\n")  # Print the device used

    # Load the dataset
    dts_path = Path(__file__).parent.absolute() / f"datasets/{args[0]}/processed_data/processed_test_dts.pt"  # Get the path to the dataset
    test_dts = torch.load(dts_path)                                                                          # Load the test dataset
    test_loader = DataLoader(test_dts, batch_size=1, shuffle=True, num_workers=4)                          # Create the test dataloader

    # Load the model
    state_dict_path = Path(__file__).parent.absolute() / f"net_state/{args[1]}.pt"                                    # Get the path to the model state dict
    model = NlosNet(enc_channels=(6, 16, 32, 64, 128, 256), dec_channels=(256, 128, 64, 32, 16), num_class=8).to(device)  # Create the model and move it to the device
    model.load_state_dict(torch.load(state_dict_path))                                                                            # Load the model

    # Define the loss function
    loss_fn = CL.BalancedMAELoss(reduction="weight_mean")

    # Define the output path
    out_folder = Path(__file__).parent.absolute() / "results"
    out_path = out_folder / f"{args[0]}__{args[1]}"
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    #out_path = out_folder / f"{str(args[0].stem)[10:]}_results.npy"

    # Test the model
    s_test_time = time.time()  # Start the test time
    test(
        net=model, 
        data_loader=test_loader, 
        loss_fn=loss_fn,
        device=device,
        bg=args[2],
        out_path=out_path)
    e_test_time = time.time()  # End the test time

    # Print the time spent 
    print("Test time: ", format_time(s_test_time, e_test_time))
