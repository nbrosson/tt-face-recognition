import numpy as np

from keras_vggface.vggface import VGGFace
from .utils import prepare_benchmark_names_to_empty_lists_mapping, get_face_arrays_from_file_system
from .face_detector import extract_input_face
from scipy.spatial import distance

resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


def predict_input_identity(uploaded_image=None):
    """
    1. Prepare embeddings for training data and input data
    2. Compute Euclidean Distance to make the face recognition

    :param uploaded_image: Image on which we want to run face recognition
    :return: Identity of the input image
    """
    # get embeddings arrays for both Benchmark and Input (uploaded) data
    final_benchmark_embeddings, uploaded_image_embedding = apply_embedding_to_all_faces_arrays(uploaded_image)
    minimum_euclidean_distance = None
    predicted_person = None

    # Run face recognition using Euclidean distance
    for name in final_benchmark_embeddings.keys():
        if final_benchmark_embeddings[name] != []:
            euclidean_distance_for_this_benchmark_person = distance.euclidean(final_benchmark_embeddings[name],
                                                                              uploaded_image_embedding)
            print(f"Distance between {name} and the uploaded person: {euclidean_distance_for_this_benchmark_person}")
            if not minimum_euclidean_distance or euclidean_distance_for_this_benchmark_person < minimum_euclidean_distance:
                minimum_euclidean_distance = euclidean_distance_for_this_benchmark_person
                predicted_person = name
        else:
            print(f"No available embedding for {name}")
    print(f"We believe the input target's name is {predicted_person}")
    return {"input_identity": predicted_person}


def apply_embedding_to_all_faces_arrays(uploaded_image):
    """
    1. Retrieve face detection arrays for the benchmark data. Compute face detection for the input.
    2. Compute the embeddings for all benchmark faces arrays   for the uploaded image face array.

    :param uploaded_image: Input image on which we run face recognition
    :return: Dict that maps benchmark people to their embeddings.
    :return: Array which represents the embedding for the uploaded image
    """
    benchmark_names_to_embeddings_mapping = dict()

    # Retrieve face detection arrays of all the benchmark data in the filesystem
    benchmark_names_to_face_arrays_mapping = get_face_arrays_from_file_system()

    # make a list of people that have at least one face array
    people_with_at_least_one_face_array = [
        person
        for person in benchmark_names_to_face_arrays_mapping
        if benchmark_names_to_face_arrays_mapping[person]
    ]

    # Compute image embedding on all images arrays (if there are more than one array for a person, we use the mean)
    for name in people_with_at_least_one_face_array:
        benchmark_names_to_embeddings_mapping[name] = np.mean(
            np.concatenate(
                [
                    resnet50_features.predict(face_array)
                    for face_array in benchmark_names_to_face_arrays_mapping[name]
                ],
                axis=0,
            ),
            axis=0,
        )

    # Compute face detection on the input image
    face_aray_of_uploaded_image = extract_input_face(input=uploaded_image)

    # return 1d array embedding for both benchmark people and uploaded image
    return benchmark_names_to_embeddings_mapping, resnet50_features.predict(
        face_aray_of_uploaded_image
    )



