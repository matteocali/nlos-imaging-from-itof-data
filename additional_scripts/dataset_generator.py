import getopt
import sys
import modules.dataset_utils as dts
from pathlib import Path


"""
Generate al the .xml files required to render the dataset
"""


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
    (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_permutation_list_path = (
        None  # Argument containing the permutation list directory
    )
    arg_dataset_file_path = None  # Argument containing the dataset recap file directory
    arg_batches_path = (
        None  # Argument containing the batches (final xml files) directory
    )
    arg_template_path = None  # Argument containing the xml template file directory
    arg_rnd_seed = None  # Argument containing the random seed value
    arg_img_shape = None  # Image shape
    arg_grid = None  # Pattern to use for the mask
    arg_help = "{0} -p <perm> -d <dataset> -b <batch> -t <template> -s <seed> -i <img> -g <grid>".format(
        argv[0]
    )  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, _ = getopt.getopt(
            argv[1:],
            "hp:d:b:t:s:i:g:",
            [
                "help",
                "perm=",
                "dataset=",
                "batch=",
                "template=",
                "seed=",
                "img=",
                "grid=",
            ],
        )
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
            arg_rnd_seed = int(arg)  # Set the random seed
        elif opt in ("-i", "--img"):
            arg_img_shape = tuple(map(int, arg.split(",")))  # Set the image shape
        elif opt in ("-g", "--grid"):
            arg_grid = tuple(map(int, arg.split(",")))  # Set the grid pattern

    print("Permutation list path: ", arg_permutation_list_path)
    print("Dataset path: ", arg_dataset_file_path)
    print("Batches path: ", arg_batches_path)
    print("Template path: ", arg_template_path)
    print("Random seed: ", arg_rnd_seed)
    print("Image shape: ", arg_img_shape)
    print("Grid pattern: ", arg_grid)
    print()

    return [
        arg_permutation_list_path,
        arg_dataset_file_path,
        arg_batches_path,
        arg_template_path,
        arg_rnd_seed,
        arg_img_shape,
        arg_grid,
    ]


if __name__ == "__main__":
    (
        perm_list_path,
        dts_file_path,
        batches_path,
        template_path,
        rnd_seed,
        img_shape,
        pattern,
    ) = arg_parser(
        sys.argv
    )  # Recover the input and output folder from the console args

    # CONSTANTS #
    N_CLASSES = 4  # Number of classes
    N_ELM_PER_CLASS = (
        336  # Number of elements per class (must be divisible by 2, 7 and 8)
    )
    N_BATCH_PER_CLASS = 2  # Number of different batches that will be generated
    DEF_CAM_POS = (1.0, -1.0, 1.65)  # Default camera position (x, y, z)
    DEF_CAM_ROT = (90, 0, 50)  # Default camera orientation (x, y, z)
    DEF_OBJ_POS = (0.9, 1.0, 1.65)  # Default object position (x, y, z)
    ROUGHNESS = tuple(
        i / 100 for i in range(30, 101, 5)
    )  # Roughness of the front wall material
    OBJ_NAMES = (
        "Cube",
        "Cone",
        "Cylinder",
        "Parallelepiped",
        "Sphere",
        "Concave plane",
        "Random",
    )  # List that contains the name of the different objects

    # PARAMETERS LIST #
    # List that contains all the possible translations that is granted to an object
    # [[possible x translations], [possible y translations], [possible z translations]]
    obj_tr_list = [
        [i / 10 for i in range(0, 4)],
        [i / 10 for i in range(-5, 4)],
        [i / 10 for i in range(-4, 4)],
    ]
    # List that contains all the possible rotations that is granted to an object
    # [[possible x rotations], [possible y rotations], [possible z rotations]]
    obj_full_rot_list = [
        [i for i in range(-90, 91)],
        [i for i in range(-90, 91)],
        [i for i in range(-90, 91)],
    ]
    # List that contains all the possible rotations that is granted to an object with no
    # rotations over the z axis [[possible x rotations], [possible y rotations], 0]
    obj_partial_rot_list = [
        [i for i in range(-90, 91)],
        [i for i in range(-90, 91)],
        [0],
    ]
    # List that contains all the possible rotations that is granted to the camera
    # [[possible x rotations], [possible y rotations], [possible z rotations]]
    cam_rot_list = [
        [i for i in range(85, 96)],
        [i for i in range(-5, 6)],
        [i for i in range(50, 91)],
    ]
    # List that contains all the possible positions that is granted to the camera
    # [[possible x positions], [possible y positions], [possible z positions]]
    cam_pos_list = [
        [i / 10 for i in range(10, 14)],
        [i / 10 for i in range(-13, -9)],
        [i / 10 for i in range(15, 18)],
    ]

    # BUILD THE FINAL SET OF ROTATIONS/TRANSLATIONS/POSITIONS #

    print(
        "Compute all the permutations of the camera and objects locations and rotations (batch by batch):"
    )
    tr_rot_list, roughness_list = dts.log.generate_dataset_list(
        obj_tr_list=obj_tr_list,
        obj_full_rot_list=obj_full_rot_list,
        obj_partial_rot_list=obj_partial_rot_list,
        cam_rot_list=cam_rot_list,
        cam_pos_list=cam_pos_list,
        def_cam_pos=DEF_CAM_POS,
        def_cam_rot=DEF_CAM_ROT,
        n_classes=N_CLASSES,
        n_elm_per_class=N_ELM_PER_CLASS,
        n_batches=N_BATCH_PER_CLASS,
        obj_names=OBJ_NAMES,
        roughness_values=ROUGHNESS,
        folder_path=perm_list_path,
        seed=rnd_seed,
    )
    dts.log.generate_dataset_file(
        tx_rt_list=tr_rot_list,
        folder_path=dts_file_path,
        obj_names=OBJ_NAMES,
        obj_base_pos=DEF_OBJ_POS,
        wall_roughness=roughness_list,
    )  # Export a .txt file containing the information of the dataset
    # actually used after all the random permutations

    # BUILD THE XML FILE FOR EACH SCENE #
    print("\nGenerate all the .xml files:")
    dts.build.generate_dataset_xml(
        tr_rot_list=tr_rot_list,
        n_classes=N_CLASSES,
        templates_folder=template_path,
        folder_path=batches_path,
        obj_names=OBJ_NAMES,
        obj_base_pos=DEF_OBJ_POS,
        roughness=roughness_list,
        img_shape=img_shape,
        pattern=pattern,
    )
