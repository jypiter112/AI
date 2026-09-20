AI — YOLOv11 Real-Time Screen Object Detection

A computer-vision project built on YOLOv11 (via the ultralytics package) and OpenCV, fine-tuned on a small custom dataset to detect enemy players in the game Arma 3. The model captures the screen, runs local inference on the frame, and renders a live, updating bounding box around detected enemies through a transparent always-on-top overlay window.

This repo was built as a personal computer-vision / real-time-inference learning project. See Disclaimer before using it with any online multiplayer game.

How it works
Capture — the screen is grabbed with PIL.ImageGrab (or a video file, for offline testing with OpenCV's VideoCapture).
Crop / preprocess — the raw screenshot is saved and passed to the model.
Inference — a YOLOv11 model (drone.pt), fine-tuned from pretrained COCO weights on a small custom-labeled dataset, predicts bounding boxes (x, y, w, h) for detected targets.
Overlay — a transparent, click-through tkinter window is layered on top of the game window. Detected boxes are redrawn on a fast loop (~1 ms tick) so the overlay tracks targets in near real time, independent of the (slower) detection loop, which runs on its own background thread.
FPS counter — the OpenCV demo script also overlays a live FPS readout for benchmarking inference speed.
Repo contents
File	Description
yolov11_cv2_img_detection.py	Standalone OpenCV demo. Runs YOLOv11 inference on a video file (modeltest1.mp4) frame-by-frame, draws a bounding box + "Enemy" label directly on the frame with cv2.rectangle / cv2.putText, and displays it in an cv2.imshow window with an FPS counter. Good for testing the model against pre-recorded footage without touching the live game.
yolov11_detection_overlay.py	The real-time version. Grabs the live desktop, runs detection in a background thread, and draws detections as a transparent tkinter overlay on top of the game window so boxes update independently of the detection loop's frame rate.
Requirements
Windows 11 (the overlay uses Windows-style multi-monitor screen grabbing and transparent-color window tricks; it was developed and tested on this host)
Python 3.9+
A CUDA-capable GPU is recommended for real-time inference speed, but not required

Install dependencies:

bash
pip install ultralytics opencv-python numpy pillow pyautogui

tkinter and threading ship with the Python standard library.

Model
Architecture: YOLOv11 (Ultralytics)
Base weights: pretrained (COCO) weights, fine-tuned via transfer learning
Training data: a relatively small, custom-labeled dataset of in-game screenshots, annotated in YOLO format (see labelformat.com's YOLOv11 guide for the annotation spec used)
Weights file: the scripts expect a trained weights file named drone.pt in the working directory. This file is not included in the repo — you'll need to train your own on a labeled dataset of the target class(es), or point the scripts at your own .pt file.

To train your own weights with Ultralytics:

bash
yolo detect train data=your_dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
Usage

Offline / video test:

bash
python yolov11_cv2_img_detection.py

Runs detection against modeltest1.mp4 and shows an OpenCV window with boxes + FPS. Press Esc to quit.

Live overlay:

bash
python yolov11_detection_overlay.py

Launches the transparent overlay on top of your primary display and starts the background detection thread. Boxes are drawn in green with an "Enemy" label. Press Ctrl+C in the terminal to stop.

Screen dimensions (right_screen_dimensions, left_screen_dimensions) are currently hardcoded for a specific dual-monitor setup — update these to match your own display layout before running.

Limitations
Trained on a small dataset, so generalization outside the training distribution (different maps, lighting, character skins, distances) is limited.
No object tracking between frames — each detection is independent, so boxes can flicker or jump if the model misses a frame.
Overlay screen coordinates and monitor dimensions are hardcoded rather than auto-detected.
Single-detection assumption in the video-demo script (yolov11_cv2_img_detection.py only reads the first detected box per frame).
Disclaimer

This project was built for learning purposes around real-time object detection and screen-based inference pipelines. Using detection/overlay tools of this kind in online multiplayer games (including Arma 3) will generally violate the game's terms of service / anti-cheat policy and can result in a ban. Use at your own risk, and only against local/offline content (bots, replays, recorded footage) if you want to stay on the safe side.
