import sys
import getopt
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils.NlosNet import NlosNet
from utils.test_function import test
from utils.utils import format_time


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
    # Argument containing the name of the network to load
    arg_net_name = ""
    # Argument containing the path where to save the output
    arg_output_path = "results"
    # Help string
    arg_help = "{0} -i <input> -m <model> -o <output>".format(argv[0])

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(
            argv[1:], "hi:m:o:", ["help", "input=", "model=", "output="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            arg_data_path = Path(arg)    # Set the path to the raw data
        elif opt in ("-m", "--model"):
            arg_net_name = arg           # Set the name of the model to load
        elif opt in ("-o", "--output"):
            arg_output_path = Path(arg)  # Set the path to the output file

    print("Input folder: ", arg_data_path)
    print("Model name: ", arg_net_name)
    print("Output folder: ", arg_output_path)
    print()

    return [arg_data_path, arg_net_name, arg_output_path]


if __name__ == '__main__':
    torch.manual_seed(2097710)   # Set the random seed
    args = arg_parser(sys.argv)  # Parse the input arguments
    
    # Chekc if the gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device, "\n")  # Print the device used

    # If not alreaty present create the output folder
    args[2].mkdir(parents=True, exist_ok=True)  # type: ignore

    # Load the dataset
    test_dts = torch.load(args[0])  # Load the test dataset
    test_loader = DataLoader(test_dts, batch_size=32, shuffle=True, num_workers=4)  # Create the test dataloader  # type: ignore

    # Load the model
    state_dict_path = Path(__file__).parent.absolute() / "net_state" / args[1] + ".pt"                                                      # Get the path to the model state dict
    model = NlosNet(enc_channels=(6, 16, 32, 64, 128, 256), dec_channels=(256, 128, 64, 32, 16), num_class=8, n_final_layers=3).to(device)  # Create the model and move it to the device
    model.load_state_dict(torch.load(args[1]))                                                                                              # Load the model

    # Define the loss function
    depth_loss_fn = torch.nn.L1Loss()
    mask_loss_fn = torch.nn.BCEWithLogitsLoss()

    # Test the model
    s_test_time = time.time()  # Start the test time
    test_loss, out = test(
        net=model, 
        data_loader=test_loader, 
        depth_loss_fn=depth_loss_fn, 
        mask_loss_fn=mask_loss_fn,
        l = 0.6,
        device=device,
        out_path=(args[2] / f"{str(args[0].stem)[10:]}_results.npy"))  # type: ignore
    e_test_time = time.time()  # End the test time

    # Print the results
    print("Test loss: ", test_loss)
    print("Test time: ", format_time(s_test_time, e_test_time))
