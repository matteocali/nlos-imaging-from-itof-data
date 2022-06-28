import getopt
import sys
from pathlib import Path
from modules import utilities as ut


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
    (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_permutation_list_path = None  # Argument containing the permutation list directory
    arg_dataset_file_path = None  # Argument containing the dataset recap file directory
    arg_batches_path = None  # Argument containing the batches (final xml files) directory
    arg_template_path = None  # Argument containing the xml template file directory
    arg_rnd_seed = None  # Argument containing the random seed value
    arg_help = "{0} -p <perm> -d <dataset> -b <batch> -t <template> -s <seed>".format(argv[0])  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(argv[1:],
                                   "hp:d:b:t:s:",
                                   ["help", "perm=", "dataset=", "batch=", "template=", "seed="])
    except:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # Print the help message
            sys.exit(2)
        elif opt in ("-p", "--perm"):
            arg_permutation_list_path = Path(arg)  # Set the permutation lists directory
        elif opt in ("-d", "--dataset"):
            arg_dataset_file_path = Path(arg)  # Set the dataset recap directory
        elif opt in ("-b", "--batch"):
            arg_batches_path = Path(arg)  # Set the batches directory
        elif opt in ("-t", "--template"):
            arg_template_path = Path(arg)  # Set the template file directory
        elif opt in ("-s", "--seed"):
            arg_rnd_seed = int(arg)  # Set the template file directory

    print("Permutation list path: ", arg_permutation_list_path)
    print("Dataset path: ", arg_dataset_file_path)
    print("Batches path: ", arg_batches_path)
    print("Template path: ", arg_template_path)
    print("Random seed: ", arg_rnd_seed)
    print()

    return [arg_permutation_list_path, arg_dataset_file_path, arg_batches_path, arg_template_path, arg_rnd_seed]


if __name__ == '__main__':
    arg_permutation_list_path, arg_dataset_file_path, arg_batches_path, arg_template_path, arg_rnd_seed = arg_parser(
        sys.argv)  # Recover the input and output folder from the console args

    # CONSTANTS #
    N_BATCH = 8  # Number of different batches that will be generated
    DEF_CAM_POS = (1.5, -1, 1.65)  # Default camera position (x, y, z)
    DEF_CAM_ROT = (90, 0, 50)  # Default camera orientation (x, y, z)
    N_TR_OBJ = [9, 9, 8, 8, 8, 8, 7, 7]  # List that contains the number of different translations that I want for each
                                         # object (sphere excluded) for each batch (len(list) = n_batches)
    N_TR_SPHERE = [16, 16, 14, 14, 14, 14, 12, 12]  # List that contains the number of different translations that I
                                                    # want for sphere for each batch (len(list) = n_batches)
    N_ROT_OBJ = [10, 10, 9, 9, 9, 9, 8, 8]  # List that contains the number of different rotations that I want for each
                                            # object (sphere excluded) for each batch (len(list) = n_batches)
    N_TR_ROT_CAM = 18  # Number of different position and/or rotations that I require for the camera
    OBJ_POS = {"Cube": [1.5, 1, 1.65],
               "Cone": [1.5, 1, 1.55],
               "Cylinder": [1.5, 1, 1.65],
               "Parallelepiped": [1.5, 1, 1.65],
               "Sphere": [1.5, 1, 1.65],
               "Concave plane": [1.5, 1, 1.65],
               "Cube + sphere": [1.5, 1, 1.65],
               "Cylinder + cone": [1.5, 1, 1.65],
               "Sphere + cone": [1.5, 1, 1.65]}  # Dictionary that contains all the default position of the objects

    # PARAMETERS LIST #
    # List that contains all the possible translations that is granted to an object
    # [[possible x translations], [possible y translations], [possible z translations]]
    obj_tr_list = [[i / 10 for i in range(0, 21, 5)],
                   [i / 10 for i in range(-4, 13, 2)],
                   [i / 10 for i in range(-10, 11, 5)]]
    # List that contains all the possible rotations that is granted to an object
    # [[possible x rotations], [possible y rotations], [possible z rotations]]
    obj_full_rot_list = [[i for i in range(-90, 91, 45)],
                         [i for i in range(-90, 91, 45)],
                         [i for i in range(-90, 91, 45)]]
    # List that contains all the possible rotations that is granted to an object with no
    # rotations over the z axis [[possible x rotations], [possible y rotations], 0]
    obj_partial_rot_list = [[i for i in range(-90, 91, 45)],
                            [i for i in range(-90, 91, 45)],
                            [0]]
    # List that contains all the possible rotations that is granted to the camera
    # [[possible x rotations], [possible y rotations], [possible z rotations]]
    cam_rot_list = [[i for i in range(90, 101, 5)],
                    [i for i in range(-30, 31, 11)],
                    [i for i in range(50, 101, 10)]]
    # List that contains all the possible positions that is granted to the camera
    # [[possible x positions], [possible y positions], [possible z positions]]
    cam_pos_list = [[i / 10 for i in range(10, 31, 5)],
                    [i / 10 for i in range(-30, -11, 5)],
                    [i / 10 for i in
                     range(10, 31, 5)]]

    # BUILD THE FINAL SET OF ROTATIONS/TRANSLATIONS/POSITIONS #
    print("Compute all the permutations of the camera and objects locations and rotations (batch by batch):")
    tr_rot_list = ut.generate_dataset_list(obj_tr_list=obj_tr_list,
                                           obj_full_rot_list=obj_full_rot_list,
                                           obj_partial_rot_list=obj_partial_rot_list,
                                           cam_rot_list=cam_rot_list,
                                           cam_pos_list=cam_pos_list,
                                           def_cam_pos=DEF_CAM_POS,
                                           def_cam_rot=DEF_CAM_ROT,
                                           n_batches=N_BATCH,
                                           obj_names=list(OBJ_POS.keys()),
                                           n_tr_rot_cam=N_TR_ROT_CAM,
                                           n_tr_obj=N_TR_OBJ,
                                           n_rot_obj=N_ROT_OBJ,
                                           n_tr_sphere=N_TR_SPHERE,
                                           folder_path=arg_permutation_list_path,
                                           seed=arg_rnd_seed)

    ut.generate_dataset_file(tx_rt_list=tr_rot_list,
                             folder_path=arg_dataset_file_path,
                             objs=OBJ_POS)  # Export a .txt file containing the information of the dataset
    # actually used after all the random permutations

    # BUILD THE XML FILE FOR EACH SCENE #
    print("\nGenerate all the .xml files:")
    ut.generate_dataset_xml(tr_rot_list=tr_rot_list,
                            template=arg_template_path,
                            folder_path=arg_batches_path,
                            objs=OBJ_POS)
