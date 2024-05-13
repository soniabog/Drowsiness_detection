import cv2
import dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model

def add_margin_to_eye(x, y, w, h, width, height, margin_w=0.6, upper_margin_h=2.5, lower_margin_h=0.7):

    x_margin = int(w * margin_w)
    upper_y_margin = int(h * upper_margin_h)
    lower_y_margin = int(h * lower_margin_h)

    x_new = max(x - x_margin, 0)
    y_new = max(y - upper_y_margin, 0)
    x_end = min(x + w + x_margin, width)
    y_end = min(y + h + lower_y_margin, height)

    return (x_new, y_new, x_end - x_new, y_end - y_new)

def add_margin_to_mouth(x, y, w, h, width, height, margin_w=0.3, margin_h=0.3):

    x_margin = int(w * margin_w)
    y_margin = int(h * margin_h)

    x_new = max(x - x_margin, 0)
    y_new = max(y - y_margin, 0)
    x_end = min(x + w + x_margin, width)
    y_end = min(y + h + y_margin, height)

    return (x_new, y_new, x_end - x_new, y_end - y_new)

def classify_eye_state(roi, model):
    roi_resized = cv2.resize(roi, (145, 145))
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
    roi_normalized = roi_rgb / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)
    prediction = model.predict(roi_expanded)
    return prediction[0]

eye_state_model = load_model(r'\model\model_2.h5')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image_path = 'image_path'
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
            eye_state_prediction = classify_eye_state(roi, eye_state_model)

            if isinstance(eye_state_prediction, np.ndarray):
                eye_state_probability = eye_state_prediction[0]
            else:
                eye_state_probability = eye_state_prediction

            eye_state = "Open" if eye_state_probability < 0.5 else "Closed"
            print(f"{name} state: {eye_state}")