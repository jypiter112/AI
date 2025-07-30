import time
import numpy as np
import pyautogui
import threading
from ultralytics import YOLO
from PIL import ImageGrab
from tkinter import Tk, Canvas

# https://labelformat.com/formats/object-detection/yolov11/#

model = YOLO("drone.pt")
right_screen_dimensions = (2560, 1440)
left_screen_dimensions = (1920, 1080)

# Calculating fps
prev_frame_time = 0
new_frame_time = 0

# Overlay window setup
root = Tk()
root.title("Overlay Example")
x="0"
y="0"
root.geometry(f"{right_screen_dimensions[0]}x{right_screen_dimensions[1]}+{x}+{y}")
root.overrideredirect(True)  # Remove window decorations
root.attributes("-transparentcolor", "red")  # Set transparency
root.config(bg="red")  # Set background color to match transparency

canvas = Canvas(root, width=right_screen_dimensions[0], height=right_screen_dimensions[1], bg="red", highlightthickness=0)
canvas.pack()
root.wm_attributes("-topmost", 1)  # Keep the window on top
locations = []

def move_mouse(x, y):
    x = int(x)
    y = int(y)
    # pyautogui.moveTo(x, y, 0.2)
    pyautogui.moveTo(x, y, duration=0.1)
    print(f"Mouse moved to enemy: ({x}, {y})")

def calc_location(x, y, w, h):
    x -= left_screen_dimensions[0]
    x = int(x)
    y = int(y)
    
    return (x, y, w, h)

def read_location(xywh):
    location = calc_location(xywh[0], xywh[1], xywh[2], xywh[3])
    return location

def detect_enemy():
    screenshot = ImageGrab.grab(all_screens=True)
    screenshot.save("screenshot.png", "PNG")

    result = model.predict(source="screenshot.png", conf=0.45)
    
    if result is None:
        return None
    return result

def draw_overlay_rectangle(x, y, w, h):
    # READ RAW XYWH
    if canvas is None:
        return
    x = int(x - w // 2) # Possible fix for centering
    y = int(y - h // 2) # Possible fix for centering
    nx = int(x + w)
    ny = int(y + h)
    if y - 30 < 0:
        y = 30
    if x - 10 < 0 and x - w // 2 < 0:
        x = 10 + w // 2
    canvas.create_rectangle(x, y, nx, ny, outline="green")
    canvas.create_text(x, y - 10, text="Enemy", font=("Arial", 8))

def update_overlay():
    global locations, canvas, overlay_created
    global new_frame_time, prev_frame_time
    canvas.delete("all")  # Clear previous drawings
    if locations is not [] and canvas is not None:
        for loc in locations:
            x, y, w, h = loc
            draw_overlay_rectangle(x, y, w, h)
    else:
        if canvas is not None:
            canvas.delete("all")
    root.after(1, update_overlay)  # Update every 1 ms

def run_detection():
    global locations, canvas
    while True:
            try:
                results = detect_enemy()
                locations = []
                if results is None:
                    continue
                for res in results:
                    boxes = res.boxes
                    if boxes is None:
                        return None
                    for xywh in boxes.xywh:
                        loc = read_location(xywh.tolist())
                        if loc is not None:
                            locations.append(loc)
                # move_mouse([0], [1])
                # pyautogui.click()
            except KeyboardInterrupt:
                print("Detection stopped by user.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
if __name__ == "__main__":
    detection_thread = threading.Thread(target=run_detection).start()
    root.after(1, update_overlay)  # Start the overlay update loop
    root.mainloop()
    print("\n exiting...")
