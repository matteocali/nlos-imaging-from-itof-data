import sys
import time
import getopt
from pathlib import Path
import glob
import random


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
    arg_out_path = Path("dts_csv_split")  # Argument containing the path of the output folder
    arg_shuffle = True                    # Argument containing the flag for shuffling the dataset
    arg_help = "{0} -n <name> -i <input> -o <output> -s <shuffle>".format(argv[0])  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(argv[1:], "hn:i:o:s:", ["help", "name=", "input=", "output=", "shuffle="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-n", "--name"):
            arg_name = arg  # Set the attempt name
        elif opt in ("-i", "--input"):
            arg_data_path = Path(arg)  # Set the path to the raw data
        elif opt in ("-o", "--output"):
            arg_out_path = Path(arg)  # Set the output path
        elif opt in ("-s", "--shuffle"):
            if arg == "True":  # Set the shuffle flag
                arg_shuffle = True
            else:
                arg_shuffle = False
        
    print("Attempt name: ", arg_name)
    print("Input folder: ", arg_data_path)
    print("Output folder: ", arg_out_path)
    print("Shuffle: ", arg_shuffle)
    print()

    return [arg_name, arg_data_path, arg_out_path, arg_shuffle]


if __name__ == '__main__':
    random.seed(2097710)         # Set the random seed
    start_time = time.time()     # Start the timer
    args = arg_parser(sys.argv)  # Parse the input arguments

    out_path = args[2] / f"{args[0]}_csv_split"

    # Create the output folder if it does not exist
    out_path.mkdir(parents=True, exist_ok=True)

    # Load the dataset folderù
    elements = glob.glob1(args[1],"*.h5")
    n_of_elements = len(elements)
    
    # Shuffle the dataset if the flag is set
    if args[3]:
        random.shuffle(elements)

    # Split the dataset in train, validation and test (60, 20, 20)
    train = elements[:int(round(n_of_elements*0.6, 0))]
    validation = elements[int(round(n_of_elements*0.6, 0)) : int(round(n_of_elements*0.8, 0))]
    test = elements[int(round(n_of_elements*0.8, 0)):]

    # Create the csv files
    with open(out_path / (args[0] + "_train.csv"), "w") as f:
        for element in train:
            f.write(element + "\n")
    with open(out_path / (args[0] + "_validation.csv"), "w") as f:
        for element in validation:
            f.write(element + "\n")
    with open(out_path / (args[0] + "_test.csv"), "w") as f:
        for element in test:
            f.write(element + "\n")

    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    hours, minutes = divmod(minutes, 60)
    print("Total time to split the dataset is %d:%02d:%02d" % (hours, minutes, seconds))