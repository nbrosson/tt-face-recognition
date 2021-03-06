# The project

The goal of the project is to build a face recognition algorithm, in a simple web application. How does it work?

1. Define the list of people you want to run the model on. (One or more images per person). Put these images in 
data/images/raw_images/person_name. One folder per person (you can follow data_template example).

2. **Run Face Detection:** Face Detection is an heavy calculation. We will run face detection on all the training data,
and store the results in data/images/images_arrays.

3. **Image embedding:** Images are transformed to 1-dimension vectors using the VGGFace model (based on the ResNet50
architecture). It is a famous CNN which performed very well on famous people, and can be used for image embedding. 
Image embedding is done on all the training data, and on the image to predict. 

4. **Prediction**: We run Euclidean Distances to find the final result.  
 
# Start the project

## Create and run a virtual environment (Windows)

First, OpenCV is required in your Computer. You also need Python (This project is made with Python 3.7.6).
Then:
```
$python -m venv .venv
$C:\Users\...\tt-face-recognition\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## All the steps to run the web application

1. Go to data/images/raw_images
2. Create one folder per person in raw_images/ (Follow data_template example)
3. Run this command to run the face detection algorithm and store the results: `python main.py prepare-faces-arrays-for-benchmark-images`.
It is a long calculation, and this is why we split the face detection from the face recognition. 
4. Run the web application that will perform face recognition:
```
set FLASK_APP=app.py 
flask run --port=5000
```

6. test your application in 127.0.0.1:5000