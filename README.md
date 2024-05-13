# Driver Drowsiness Detection (in progress)

## Project overview
This Driver Drowsiness Detection system is designed to identify whether a driver's eyes are open or closed, using machine learning techniques. The project utilizes facial landmark detection and convolutional neural networks to analyze eye states from images, helping to increase road safety by alerting when signs of drowsiness are detected.

## Repository Structure
- /model: Contains a Jupyter Notebook that details the creation and evaluation of machine learning models using a dataset from Kaggle. The best-performing model is selected here.
- open_or_closed.py: Defines functions for facial detection and drowsiness detection using the trained model.
- photos_capture.py: Script to capture frames from video at a set interval to create a dataset for testing or further training.
- test.py: Evaluates the model against a new set of images and provides a classification report, detailing performance metrics.

## Used libraries:
- OpenCV
- dlib
- NumPy
- imutils
- Keras
