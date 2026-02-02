import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

st.title("👥 Person Counting App")

uploaded = st.file_uploader("Upload Image", type=["jpg","png"])

if uploaded:
    img = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    results = model(img)
    count = 0

    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])] == "person":
                count += 1
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

    st.image(img, channels="BGR")
    st.success(f"👥 Total Persons Detected: {count}")
