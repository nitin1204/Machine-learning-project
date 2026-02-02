# person_count_single.py
from ultralytics import YOLO
import cv2
import numpy as np

# --- Load YOLOv8 nano model (auto-download, official weights) ---
model = YOLO("yolov8n")  # Do not manually place .pt file

# --- Image path ---
img_path = "images/test.jpg"

# --- Read image ---
img = cv2.imread(img_path)
if img is None:
    print("Image not found! Please check path:", img_path)
    exit()

# --- Run detection ---
results = model(img)

# --- Initialize count ---
person_count = 0

# --- Draw boxes & count persons ---
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# --- Show count on image ---
cv2.putText(img, f"Persons: {person_count}", (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# --- Display image ---
cv2.imshow("Person Count Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Print count in CMD ---
print("Total Persons Detected:", person_count)
