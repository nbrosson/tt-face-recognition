import numpy as np
import os
from os.path import isdir


def prepare_benchmark_names_to_empty_lists_mapping(path=os.getcwd() + "\\data\\images\\raw_images"):
    """
    For each person of images/raw_images, we will create a key value
    key: name; value: embedded vector

    :param path: Path where to find the raw_images. If None, we use the default path.
    :return: Dict such as {"name_1": [], "name_2": []...}
    """
    benchmark_mapping = {}
    for name in os.listdir(path):
        benchmark_mapping[name] = []
    return benchmark_mapping


def get_face_arrays_from_file_system():
    """
    For each person, get all the arrays from images_arrays, corresponding to the face of the person.

    :param benchmark_names_mapping: Dict such as {"name_1": [], "name_2": []...}
    :param path: Path where to find the detected faces array.
    :return: Dict such as {"name_1": [face_array_1, face_array_2, ...], "name_2": [face_array_1, ...]...}
    """
    path = os.getcwd() + "\\data\\images\\images_arrays"
    benchmark_names_to_face_arrays_mapping = dict()
    for name in os.listdir(path):
        benchmark_names_to_face_arrays_mapping[name] = []
        temp_path_face_arrays_one_person = path + f"\\{name}"
        if not isdir(temp_path_face_arrays_one_person):
            os.mkdir(temp_path_face_arrays_one_person)
        for array_path in os.listdir(temp_path_face_arrays_one_person):
            benchmark_names_to_face_arrays_mapping[name].append(np.load(temp_path_face_arrays_one_person + "\\" + array_path))
    return benchmark_names_to_face_arrays_mapping
