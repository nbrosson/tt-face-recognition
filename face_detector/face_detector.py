from numpy import asarray
from PIL import Image
from mtcnn.mtcnn import MTCNN
from .utils import prepare_objects
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
	The goal is to run the face detection algorithm to all training images, and store the results in the folder
	images_array/

	1. Make an images_arrays folder if not exist
	2. For each image of raw_images/ :
		a. get the name of the corresponding person
		b. create a folder in images_array for this person if it does not exist
		c. Run and store the face detection result in images_array/person_name

	:param path: Path of the images folder
	"""
	names_objects = prepare_objects()
	if not path:
		if not isdir(os.getcwd() + "\\data\\images"):
			sys.exit(
				"You need to create a data folder in the root of the project, "
				"+ data/images and data/images/raw_images folders"
			)
		path = os.getcwd() + "\\data\\images"

	for name in names_objects.keys():
		temp_path_raw_image = path + f"\\raw_images\\{name}"
		temp_path_images_arrays = path + f"\\images_arrays\\{name}"
		if not isdir(temp_path_images_arrays):
			os.mkdir(temp_path_images_arrays)
		# processed_files: list of already existing faces arrays
		processed_files = [os.path.splitext(file)[0] for file in os.listdir(temp_path_images_arrays)]
		for img_path in os.listdir(temp_path_raw_image):
			pic_name = os.path.splitext(img_path)[0]
			if pic_name not in processed_files:
				try:
					face_array = extract_face_from_image(temp_path_raw_image + "\\" + img_path)
					np.save(temp_path_images_arrays + "\\" + f"{pic_name}.npy", face_array)
				except:
					pass


def extract_face_from_image(filename=None, img=None, required_size=(224, 224)):
	# load image from file or from numpy image
	if img is not None:
		pixels = cv2.imdecode(img, cv2.IMREAD_COLOR)
	else:
		pixels = cv2.imread(filename)
	mctnn_detector = MTCNN()

	# detect faces in the image
	results = mctnn_detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height

	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
