from keras_vggface.vggface import VGGFace
from .utils import prepare_objects
from .face_detector import get_images_arrays, extract_input_image
import numpy as np
from scipy.spatial.distance import euclidean
from . import MAPPING_FOLDER_TO_NAMES


resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


def predict_input_identity(input=None):
    """
    1. Prepare embeddings for training data and input data
    2. Compute Euclidean Distance to make the face recognition

    :param input: Image on which we want to run face recognition
    :return: Identity of the input image
    """
    # get all the embeddings from the training data and the input image
    final_embeddings, input_embedding = images_embedding(input)
    min_distance = None
    associated_person = None

    # Run face recognition using Euclidean distance
    for name in final_embeddings.keys():
        if final_embeddings[name] != []:
            distance = euclidean(final_embeddings[name], input_embedding)
            print(f"Distance between {name} and the input person: {distance}")
            if not min_distance or distance < min_distance:
                min_distance = distance
                associated_person = name
        else:
            print(f"No available image for {name}")
    print(f"We believe the input target's name is {associated_person}")
    if MAPPING_FOLDER_TO_NAMES.get(associated_person):
        associated_person = MAPPING_FOLDER_TO_NAMES[associated_person]
    return {"input_identity": associated_person}


def images_embedding(input=None):
    """
    1. Retrieve face detection arrays for the training data. Compute face detection for the input.
    2. Compute the embeddings.

    :param input: Input image on which we run face recognition
    :return:
    """
    final_name_objects = prepare_objects()
    # Retrieve face detection results of all the training data
    names_objects = get_images_arrays(final_name_objects)

    # Compute face detection on the input image
    input_processed_image = extract_input_image(input=input)

    # Compute image embedding on all images arrays (if there are more than one array for a person, we use the mean)
    for name in [f for f in final_name_objects.keys() if names_objects[f]]:
        for counter, image_array in enumerate(names_objects[name]):
            names_objects[name][counter] = resnet50_features.predict(names_objects[name][counter].reshape(1, 224, 224, 3))
        final_name_objects[name] = np.mean(np.concatenate(names_objects[name], axis=0), axis=0)
    return final_name_objects, resnet50_features.predict(input_processed_image.reshape(1,224,224,3))

