import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8m model
model = YOLO("yolov8m.pt")

# Set page
st.set_page_config(page_title="Object Analyzer", layout="wide")
st.title("🔍 Object Analyzer (max 15-second videos)")
st.write("Upload a short video (up to 15 seconds). This app will detect living creatures, classify vehicle types, and analyze directions of humans.")

# Upload video
uploaded_file = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if duration > 15:
        st.error("🚫 Only videos up to 15 seconds are allowed.")
    else:
        st.success(f"✅ Video accepted ({round(duration, 2)} seconds). Processing...")

        object_counts = defaultdict(int)
        direction_summary = defaultdict(set)
        vehicle_types_count = defaultdict(int)
        living_creatures_count = 0

        # Define categories
        animals = {"cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
        vehicles = {"car", "truck", "bus", "train", "motorcycle", "bicycle", "boat"}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    if box.conf < 0.5:
                        continue
                    cls_id = int(box.cls)
                    cls_name = model.names[cls_id]
                    object_counts[cls_name] += 1

                    # Count living creatures
                    if cls_name == "person" or cls_name in animals:
                        living_creatures_count += 1

                    # Count vehicle types
                    if cls_name in vehicles:
                        vehicle_types_count[cls_name] += 1

                    # Estimate direction of people
                    if cls_name == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = x2 - x1
                        h = y2 - y1
                        aspect_ratio = h / w if w != 0 else 0
                        direction = "Front" if aspect_ratio > 1.3 else "Back"
                        direction_summary[cls_name].add(direction)

        cap.release()
        os.remove(temp_video_path)  # Clean up the temporary file

        st.subheader("🧍 Living Creatures Detected")
        st.write(f"**Total Humans and Animals:** {living_creatures_count}")

        st.subheader("🚗 Vehicles Detected")
        if vehicle_types_count:
            for vehicle, count in vehicle_types_count.items():
                st.write(f"• **{vehicle.capitalize()}** — Count: {count}")
        else:
            st.write("No vehicles detected.")