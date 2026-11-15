import os
import time
from urllib.parse import quote

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

CAMERA_HOST = os.environ.get("CAMERA_HOST", "127.0.0.1")
CAMERA_PORT = os.environ.get("CAMERA_PORT", "5001")
CAMERA_TOKEN = os.environ.get("CAMERA_TOKEN", "")
stream_url = f"http://{CAMERA_HOST}:{CAMERA_PORT}/video?token={quote(CAMERA_TOKEN)}"
MODEL_PATH = "hand_landmarker.task"

#landmarks for the hands
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]

options = vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

#function for drawing the hand frames and mapping
def draw_hands(frame, result):
    h, w = frame.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
        for p in points:
            cv2.circle(frame, p, 4, (0, 200, 0), -1)
        # MediaPipe's handedness is reversed for this feed, so swap it
        side = "Right" if handedness[0].category_name == "Left" else "Left"
        label = f"{side} {handedness[0].score:.2f}"
        cv2.putText(frame, label, (points[0][0] - 30, points[0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("GeekSplooge Connection Fail")
    exit()

print("Connected click 'q' to quit")
start = time.monotonic()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Dropped frame/stream ended")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.monotonic() - start) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    draw_hands(frame, result)

    #result.hand_landmarks[i][j] -> landmark j (0-20) of hand i, with .x .y (0-1) and .z
   

    cv2.imshow("Macbook Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
