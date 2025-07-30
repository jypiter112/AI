import numpy as np
import cv2
import time
from ultralytics import YOLO

prev_frame_time = 0
new_frame_time = 0

model = YOLO("drone.pt")
cap = cv2.VideoCapture("modeltest1.mp4")

def get_location(result):
    result = result[0].boxes
    if len(result.xywh) == 0:
        return None
    xywh = result.xywh[0]
    xywh = xywh.tolist()
    
    x = int(xywh[0])
    y = int(xywh[1])
    w = int(xywh[2])
    h = int(xywh[3])
    return (x, y, w, h)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(frame, conf=0.4)
    location = get_location(results)
    if location:
        x, y, w, h = location
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
        cv2.putText(frame, f"Enemy", (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # calc fps
    new_frame_time = time.time()
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(frame, f"FPS: {fps}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
