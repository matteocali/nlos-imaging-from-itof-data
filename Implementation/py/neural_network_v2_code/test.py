import sys
import getopt
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils.NlosNet import NlosNet
from utils.test_function import test


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
    # Argument containing the path where the model are located
    arg_model_path = ""
    # Argument containing the path where to save the output
    arg_output_path = ""
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
            arg_data_path = Path(arg)  # Set the path to the raw data
        elif opt in ("-m", "--model"):
            arg_model_path = Path(arg)   # Set the path to the model file
        elif opt in ("-o", "--output"):
            arg_output_path = Path(arg)  # Set the path to the output file

    print("Input folder: ", arg_data_path)
    print("Model file: ", arg_model_path)
    print("Output folder: ", arg_output_path)
    print()

    return [arg_data_path, arg_model_path, arg_output_path]


if __name__ == '__main__':
    torch.manual_seed(2097710)   # Set the random seed
    args = arg_parser(sys.argv)  # Parse the input arguments
    
    # Chekc if the gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device, "\n")  # Print the device used

    # Load the dataset
    test_dts = torch.load(args[0])  # Load the test dataset
    test_loader = DataLoader(test_dts, batch_size=32, shuffle=True, num_workers=4)  # Create the test dataloader  # type: ignore

    # Load the model
    model = NlosNet(retain_dim=True, out_size=(320, 240)).to(device)  # Create the model and move it to the device
    model.load_state_dict(torch.load(args[1]))                        # Load the model

    # Define the loss function
    loss_fn = torch.nn.L1Loss()

    # Test the model
    test_loss, out = test(
        net=model, 
        data_loader=test_loader, 
        loss_fn=loss_fn, 
        device=device,
        out_path=(args[2] / f"{args[0][11:]}_results.npy"))  # type: ignore