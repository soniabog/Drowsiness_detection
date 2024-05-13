import cv2
import os
from PIL import Image

def capture_frames(video_path, output_folder, interval_seconds):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Can't open video stream or file.")
        return

    # Frequency
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    frame_count = 0
    saved_count = 1016

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Taking screenshots
        if frame_count % frame_interval == 0:
            img_path = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            # Color format from BGR (OpenCV) to RGB (Pillow)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            pil_image.save(img_path)
            saved_count += 1

        frame_count += 1



    cap.release()
    print(f"{saved_count} photos saved.")


video_path = 'videopath'
output_folder = 'outputfolderpath'
interval_seconds = 1

capture_frames(video_path, output_folder, interval_seconds)
