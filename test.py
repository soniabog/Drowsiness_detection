import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
from sklearn.metrics import classification_report
from open_or_closed import classify_eye_state, add_margin_to_eye, predictor, eye_state_model

def process_folder(folder_path, expected_label, model):
    predictions = []
    filenames = []
    detector = dlib.get_frontal_face_detector()

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceRects = detector(gray, 1)

        for (i, faceRect) in enumerate(faceRects):
            shape = predictor(gray, faceRect)
            shape_np = face_utils.shape_to_np(shape)

            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                if name in ["left_eye", "right_eye"]:
                    (x, y, w, h) = cv2.boundingRect(np.array([shape_np[i:j]]))
                    (x, y, w, h) = add_margin_to_eye(x, y, w, h, img.shape[1], img.shape[0])

                    roi = img[y:y + h, x:x + w]
                    eye_state_prediction = classify_eye_state(roi, model)

                    eye_state = 1 if eye_state_prediction[0] > 0.5 else 0
                    predictions.append(eye_state)
                    filenames.append(image_name)

    return predictions, filenames



open_folder = r'output\captured_frames\open'
closed_folder = r'output\captured_frames\closed'

open_predictions, open_filenames = process_folder(open_folder, 0, eye_state_model)
closed_predictions, closed_filenames = process_folder(closed_folder, 1, eye_state_model)

predictions = open_predictions + closed_predictions
true_labels = [0] * len(open_predictions) + [1] * len(closed_predictions)
filenames = open_filenames + closed_filenames

print(classification_report(true_labels, predictions, target_names=['Open', 'Closed']))

misclassified = [(filename, pred, true) for filename, pred, true in zip(filenames, predictions, true_labels) if pred != true]
for file, predicted_label, true_label in misclassified:
    print(f"Filename: {file}, Predicted: {'Closed' if predicted_label else 'Open'}, True: {'Closed' if true_label else 'Open'}")