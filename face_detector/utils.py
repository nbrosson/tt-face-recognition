import os
from os.path import isfile, join, isdir


def prepare_objects(path=None):
    """
    For each person of images/raw_images, we will create a key value
    key: name; value: embedded vector

    returns: Dict such as {"name_1": [], "name_2": []...}
    """
    objects = {}
    if not path:
        path = os.getcwd() + "\\data\\images\\raw_images"
    for name in os.listdir(path):
        objects[name] = []
    return objects


def get_images_arrays(final_name_objects, path=None):
    """
    For each person, get all the arrays from images_arrays, corresponding to the face of the person.
    :param final_name_objects:
    :param path:
    :return:
    """
    names_objects = final_name_objects.copy()
    if not path:
        path = os.getcwd() + "\\data\\images"

    for name in names_objects.keys():
        temp_path_images_arrays = path + f"\\images_arrays\\{name}"
        if not isdir(temp_path_images_arrays):
            os.mkdir(temp_path_images_arrays)
        for array_path in os.listdir(temp_path_images_arrays):
            names_objects[name].append(np.load(temp_path_images_arrays + "\\" + array_path))
    return names_objects