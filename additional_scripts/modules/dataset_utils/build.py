import numpy as np
from pathlib import Path
from random import sample as rnd_sample
from lxml import etree as et
from tqdm import tqdm
from typing import Union
from ..utilities import permute_list, blender2mitsuba_coord_mapping, spot_bitmap_gen


def rnd_obj_data_generator(
    tr_rot_list: list[list[list]], obj_names: list
) -> list[list[list]]:
    """
    Function that define the rotation, translation and parent objects of each randomly generated one
    :param tr_rot_list: list of all the rotations and translation
    :param obj_names: list of the names of the possible object to combine
    :return: Updated input list [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], ..., obj_rand[(camera_pos, camera_rot), ((obj1_tr, obj1_rot), (obj2_tr, obj2_rot), (obj1_name, obj2_name))]], batch2[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], ..., obj_rand[(camera_pos, camera_rot), ((obj1_tr, obj1_rot), (obj2_tr, obj2_rot), (obj1_name, obj2_name))], ...]
    """
    delta_tr_permute = permute_list(
        [
            [i / 100 for i in range(-12, 16)],
            [i / 100 for i in range(-12, 16)],
            [i / 100 for i in range(-12, 16)],
        ]
    )
    delta_rot_permute = permute_list(
        [
            [i for i in range(-45, 45)],
            [i for i in range(-45, 45)],
            [i for i in range(-45, 45)],
        ]
    )

    for b_index, batch in enumerate(tr_rot_list):
        for e_index, elm in enumerate(batch[-1]):
            delta_tr = rnd_sample(delta_tr_permute, 1)[0]
            delta_rot = rnd_sample(delta_rot_permute, 1)[0]
            objs = tuple(rnd_sample(obj_names, 2))
            second_obj_tr = (
                round(elm[1][0][0] + delta_tr[0], 1),
                round(elm[1][0][1] + delta_tr[1], 1),
                round(elm[1][0][2] + delta_tr[2], 1),
            )
            second_obj_rot = (
                round(elm[1][1][0] + delta_rot[0], 0),
                round(elm[1][1][1] + delta_rot[1], 0),
                round(elm[1][1][2] + delta_rot[2], 0),
            )
            second_obj_tr_rot = (second_obj_tr, second_obj_rot)
            tr_rot_list[b_index][-1][e_index] = (
                elm[0],
                (elm[1], second_obj_tr_rot, objs),
            )

    return tr_rot_list


def save_dts_xml(
    template_path: Union[str, Path],
    file_path: Union[str, Path],
    cam_pos: list,
    cam_rot: list,
    roughness: str,
    obj_file_name: str = None,
    obj_file_name_1: str = None,
    obj_file_name_2: str = None,
    col: int = None,
    row: int = None,
) -> None:
    """
    Function that save the dts xml file
    :param template_path: path to the template file
    :param file_path: path to the file to save
    :param obj_file_name: name of the single object
    :param obj_file_name_1: name of the first object in the random object
    :param obj_file_name_2: name of the second object in the random object
    :param cam_pos: list of the camera position
    :param cam_rot: list of the camera rotation
    :param roughness: roughness value of the wall
    :param col: column index
    :param row: row index
    """

    with open(str(template_path), encoding="utf8") as f:  # Open the template file
        tree = et.parse(f)
        root = tree.getroot()

        for elem in root.getiterator():  # For each element in the template file
            try:  # Try to change the value of the element if present
                if elem.attrib["value"] == "obj_name":
                    # noinspection PyUnboundLocalVariable
                    elem.attrib["value"] = str(obj_file_name)
                elif elem.attrib["value"] == "obj_name_1":
                    # noinspection PyUnboundLocalVariable
                    elem.attrib["value"] = str(obj_file_name_1)
                elif elem.attrib["value"] == "obj_name_2":
                    # noinspection PyUnboundLocalVariable
                    elem.attrib["value"] = str(obj_file_name_2)
                elif elem.attrib["value"] == "t_cam_x":
                    elem.attrib["value"] = str(cam_pos[0])
                elif elem.attrib["value"] == "t_cam_y":
                    elem.attrib["value"] = str(cam_pos[1])
                elif elem.attrib["value"] == "t_cam_z":
                    elem.attrib["value"] = str(cam_pos[2])
                elif elem.attrib["value"] == "r_cam_x":
                    elem.attrib["value"] = str(cam_rot[0])
                elif elem.attrib["value"] == "r_cam_y":
                    elem.attrib["value"] = str(cam_rot[1])
                elif elem.attrib["value"] == "r_cam_z":
                    elem.attrib["value"] = str(cam_rot[2])
                elif elem.attrib["value"] == "a_value":
                    elem.attrib["value"] = str(roughness)
                elif elem.attrib["value"] == "o_x":
                    elem.attrib["value"] = str(col)
                elif elem.attrib["value"] == "o_y":
                    elem.attrib["value"] = str(row)
                elif elem.attrib["value"] == "t_name":
                    elem.attrib["value"] = f"bitmap_r{row}_c{col}"
            except KeyError:
                pass
    tree.write(str(file_path), method="xml", encoding="utf8")


def generate_dataset_xml(
    tr_rot_list: list,
    n_classes: int,
    templates_folder: Path,
    folder_path: Path,
    obj_names: tuple,
    obj_base_pos: tuple,
    roughness: list,
    img_shape: tuple = None,
    pattern: tuple = None,
) -> None:
    """
    Function that given the template.xml file and all the chosen position?translation?rotation combinations generate the correspondent .xml files
    :param tr_rot_list: list of all the final combination of position/translation/rotations of each object and of the camera [batch1[obj1[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], batch2[(camera_pos, camera_rot), (obj_tr, obj_rot)], obj2[(camera_pos, camera_rot), (obj_tr, obj_rot)], ...], ...]
    :param n_classes: number of classes
    :param templates_folder: template folder
    :param folder_path: path of the output folder
    :param obj_names: tuple that contains the name of every object that will be considered
    :param obj_base_pos: tuple that contains the base position of every object that will be considered
    :param roughness: tuple that contains the value of alpha (roughness) for the front wall material
    :param img_shape: tuple that contains the shape of the image
    :param pattern: tuple that contains the pattern of the emitter grid
    """

    if not folder_path.exists():  # Create the output folder if not already present
        folder_path.mkdir(parents=True)

    for b_index, batch in tqdm(
        enumerate(tr_rot_list),
        desc="Generate all the xml files batch by batch",
        leave=True,
    ):  # Cycle through each batch
        batch_path = Path(folder_path / f"batch0{b_index + 1}")
        if not batch_path.exists():  # Create the output folder if not already present
            batch_path.mkdir(parents=True)

        if (
            b_index >= n_classes
        ):  # If the current batch is in the last two classes (the ones with different walls)
            rough_wall = True
        else:
            rough_wall = False

        for o_index, obj in tqdm(
            enumerate(batch), desc="Objects", leave=False
        ):  # Cycle through each object
            name = obj_names[o_index]  # Extract the object name
            for e_index, elm in tqdm(
                enumerate(obj), desc="File", leave=False
            ):  # Cycle through each position/translation/rotation combination
                cam_pos = [
                    elm[0][0][0],
                    elm[0][0][1],
                    elm[0][0][2],
                ]  # Extract the camera position
                cam_rot = [
                    elm[0][1][0],
                    elm[0][1][1],
                    elm[0][1][2],
                ]  # Extract the camera rotation
                if name != "Random":
                    obj_tr = [
                        elm[1][0][0],
                        elm[1][0][1],
                        elm[1][0][2],
                    ]  # Extract the object translation
                    obj_rot = [
                        elm[1][1][0],
                        elm[1][1][1],
                        elm[1][1][2],
                    ]  # Extract the object rotation
                else:
                    obj_tr_1 = [
                        elm[1][0][0][0],
                        elm[1][0][0][1],
                        elm[1][0][0][2],
                    ]  # Extract the object translation
                    obj_tr_2 = [
                        elm[1][1][0][0],
                        elm[1][1][0][1],
                        elm[1][1][0][2],
                    ]  # Extract the object translation
                    obj_rot_1 = [
                        elm[1][0][1][0],
                        elm[1][0][1][1],
                        elm[1][0][1][2],
                    ]  # Extract the object rotation
                    obj_rot_2 = [
                        elm[1][1][1][0],
                        elm[1][1][1][1],
                        elm[1][1][1][2],
                    ]  # Extract the object rotation

                if (
                    len(roughness[b_index][0]) == 0
                ):  # If the current batch doesn't require roughness
                    roughness_str = "diffuse"  # Set the roughness to diffuse
                else:  # If the current batch requires roughness
                    roughness_str = str(
                        roughness[b_index][o_index][e_index]
                    )  # Set the roughness to the one inside the roughness list

                if name != "Random":
                    # noinspection PyUnboundLocalVariable
                    obj_file_name = f"{name}_batch0{b_index + 1}_tr({obj_tr[0]}_{obj_tr[1]}_{obj_tr[2]})_rot({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    file_name = (
                        f"transient_nlos_{name.lower()}_"
                        f"cam_pos_({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_"
                        f"cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_"
                        f"obj_pos_({round(obj_base_pos[0] + obj_tr[0], 2)}_{round(obj_base_pos[1] + obj_tr[1], 2)}_{round(obj_base_pos[2] + obj_tr[2], 2)})_"
                        f"obj_rot_({obj_rot[0]}_{obj_rot[1]}_{obj_rot[2]})_"
                        f"wall_{roughness_str}.xml"
                    )  # Set the output file name in a way that contains all the relevant info
                else:
                    # noinspection PyUnboundLocalVariable
                    obj_file_name_1 = f"{elm[1][2][0]}_batch0{b_index + 1}_tr({obj_tr_1[0]}_{obj_tr_1[1]}_{obj_tr_1[2]})_rot({obj_rot_1[0]}_{obj_rot_1[1]}_{obj_rot_1[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    # noinspection PyUnboundLocalVariable
                    obj_file_name_2 = f"{elm[1][2][1]}_batch0{b_index + 1}_tr({obj_tr_2[0]}_{obj_tr_2[1]}_{obj_tr_2[2]})_rot({obj_rot_2[0]}_{obj_rot_2[1]}_{obj_rot_2[2]})".lower()  # Find the correct file name of the object given the translation and rotation value
                    file_name = (
                        f"transient_nlos_{elm[1][2][0].lower()}+{elm[1][2][1].lower()}_"
                        f"cam_pos_({cam_pos[0]}_{cam_pos[1]}_{cam_pos[2]})_cam_rot_({cam_rot[0]}_{cam_rot[1]}_{cam_rot[2]})_"
                        f"obj_pos_({round(obj_base_pos[0] + obj_tr_1[0], 2)}_{round(obj_base_pos[1] + obj_tr_1[1], 2)}_{round(obj_base_pos[2] + obj_tr_1[2], 2)})_({round(obj_base_pos[0] + obj_tr_2[0], 2)}_{round(obj_base_pos[1] + obj_tr_2[1], 2)}_{round(obj_base_pos[2] + obj_tr_2[2], 2)})_"
                        f"obj_rot_({obj_rot_1[0]}_{obj_rot_1[1]}_{obj_rot_1[2]})_({obj_rot_2[0]}_{obj_rot_2[1]}_{obj_rot_2[2]})_"
                        f"wall_{roughness_str}.xml"
                    )  # Set the output file name in a way that contains all the relevant info

                # Convert camera position and rotation from blender to mitsuba coordinates system
                cam_pos, cam_rot = blender2mitsuba_coord_mapping(
                    cam_pos[0],
                    cam_pos[1],
                    cam_pos[2],
                    cam_rot[0],
                    cam_rot[1],
                    cam_rot[2],
                )

                # Modify the template inserting the desired data
                # (code from: https://stackoverflow.com/questions/37868881/how-to-search-and-replace-text-in-an-xml-file-using-python)
                if name != "Random" and not rough_wall:
                    template_path = (
                        templates_folder / "xml_template_std_obj_std_wall.xml"
                    )
                elif name != "Random" and rough_wall:
                    template_path = (
                        templates_folder / "xml_template_std_obj_rough_wall.xml"
                    )
                elif name == "Random" and not rough_wall:
                    template_path = (
                        templates_folder / "xml_template_rnd_obj_std_wall.xml"
                    )
                else:
                    template_path = (
                        templates_folder / "xml_template_rnd_obj_rough_wall.xml"
                    )

                if img_shape is None and pattern is None:
                    if name != "Random":
                        # noinspection PyUnboundLocalVariable
                        save_dts_xml(
                            template_path=template_path,
                            file_path=batch_path / file_name.replace(" ", "_"),
                            obj_file_name=obj_file_name,
                            cam_pos=cam_pos,
                            cam_rot=cam_rot,
                            roughness=roughness_str,
                        )
                    else:
                        # noinspection PyUnboundLocalVariable
                        save_dts_xml(
                            template_path=template_path,
                            file_path=batch_path / file_name.replace(" ", "_"),
                            obj_file_name_1=obj_file_name_1,
                            obj_file_name_2=obj_file_name_2,
                            cam_pos=cam_pos,
                            cam_rot=cam_rot,
                            roughness=roughness_str,
                        )
                else:
                    mask = spot_bitmap_gen(img_size=list(img_shape), pattern=pattern)
                    non_zero_pos = np.nonzero(mask)

                    elm_path = batch_path / file_name.replace(" ", "_")[:-4]
                    if (
                        not elm_path.exists()
                    ):  # Create the output folder if not already present
                        elm_path.mkdir(parents=True)
                    for row in np.unique(non_zero_pos[0]):
                        for col in np.unique(non_zero_pos[1]):
                            if name != "Random":
                                save_dts_xml(
                                    template_path=template_path,
                                    file_path=elm_path
                                    / (file_name[:-4] + f"_r{row}_c{col}.xml").replace(
                                        " ", "_"
                                    ),
                                    # noinspection PyUnboundLocalVariable
                                    obj_file_name=obj_file_name,
                                    cam_pos=cam_pos,
                                    cam_rot=cam_rot,
                                    roughness=roughness_str,
                                    row=row,
                                    col=col,
                                )
                            else:
                                save_dts_xml(
                                    template_path=template_path,
                                    file_path=elm_path
                                    / (file_name[:-4] + f"_r{row}_c{col}.xml").replace(
                                        " ", "_"
                                    ),
                                    # noinspection PyUnboundLocalVariable
                                    obj_file_name_1=obj_file_name_1,
                                    # noinspection PyUnboundLocalVariable
                                    obj_file_name_2=obj_file_name_2,
                                    cam_pos=cam_pos,
                                    cam_rot=cam_rot,
                                    roughness=roughness_str,
                                    row=row,
                                    col=col,
                                )
