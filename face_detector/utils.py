import os


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
