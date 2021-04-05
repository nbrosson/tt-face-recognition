from PIL import Image
from mtcnn.mtcnn import MTCNN
from .utils import prepare_benchmark_names_to_empty_lists_mapping
import numpy as np
import cv2
import os
import sys
from os.path import isdir


def extract_input_face(input=None):
	"""
	Get only the face from the input image.
	:param input: Image uploaded from the web application
	:return: Array of the face of the image
	"""
	if input:  # request coming from the web application
		image = np.asarray(bytearray(input), dtype="uint8")
		input_processed_image = extract_face_from_image(img=image)
	else:  # the input is already in data and is called input.jpg
		path = os.getcwd() + "\\data\\images"
		input_processed_image = extract_face_from_image(filename=path + "\\input.jpg")
	return input_processed_image


def raw_images_processing(path=None):
	"""
	The goal is to run the face detection algorithm to all benchmark images, and store the results in the folder
	data/images/images_array/

	1. Make an images_arrays folder if not exist
	2. For each image of raw_images/ :
		a. get the name of the corresponding person
		b. create a folder in images_array for this person if it does not exist
		c. Run and store the face detection result in images_array/person_name

	:param path: Path of the images folder
	:return: None
	"""
	benchmark_names = prepare_benchmark_names_to_empty_lists_mapping()
	if not path:
		if not isdir(os.getcwd() + "\\data\\images"):
			sys.exit(
				"You need to create a data folder in the root of the project, "
				"+ data/images and data/images/raw_images folders"
			)
		path = os.getcwd() + "\\data\\images"

	for name in benchmark_names.keys():
		folder_path_of_raw_images_for_one_benchmark_person = path + f"\\raw_images\\{name}"
		folder_path_of_faces_arrays_for_one_benchmark_person = path + f"\\images_arrays\\{name}"
		if not isdir(folder_path_of_faces_arrays_for_one_benchmark_person):
			try:
				os.mkdir(folder_path_of_faces_arrays_for_one_benchmark_person)
			except FileNotFoundError:
				print("You should check that the folder images/images_arrays is created")

		# already_processed_images: list of already existing faces arrays
		already_processed_images = [os.path.splitext(face_array_file)[0] for face_array_file in
									os.listdir(folder_path_of_faces_arrays_for_one_benchmark_person)]
		for img_path in os.listdir(folder_path_of_raw_images_for_one_benchmark_person):
			pic_name = os.path.splitext(img_path)[0]
			if pic_name not in already_processed_images:
				try:
					face_array = extract_face_from_image(filename=folder_path_of_raw_images_for_one_benchmark_person + "\\" + img_path)
					np.save(folder_path_of_faces_arrays_for_one_benchmark_person + "\\" + f"{pic_name}.npy", face_array)
				except:
					pass


def extract_face_from_image(filename=None, img=None, required_size=(224, 224)):
	"""
	1. Load an image
	2. Detect the face from this image as a box
	3. Extract the box from the image as an array

	Note that if several faces are detected, only the most probable (face_detection_results[0]), is selected. Please
	provide an image with only one person to avoid confusions.

	:param filename: Url of the image
	:param img: Numpy Array
	:return: Array of the detected face
	"""
	# load image from file or from numpy image
	if img is not None:
		uploaded_raw_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
	else:
		uploaded_raw_image = cv2.imread(filename)
	mctnn_detector = MTCNN()

	# detect faces in the image
	face_detection_results = mctnn_detector.detect_faces(uploaded_raw_image)

	# extract the bounding box from the detected face
	x1, y1, width, height = face_detection_results[0]['box']
	x2, y2 = x1 + width, y1 + height

	# extract the face
	face = uploaded_raw_image[y1:y2, x1:x2]
	# resize pixels to the model size
	face_only_image = Image.fromarray(face)
	face_only_image = face_only_image.resize(required_size)
	face_array = np.asarray(face_only_image)
	return face_array.reshape(1, required_size[0], required_size[1], 3)
