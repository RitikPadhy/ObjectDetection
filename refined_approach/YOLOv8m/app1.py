import cv2
import time
from ultralytics import YOLO
import streamlit as st
from collections import defaultdict

# Load YOLO model
model = YOLO('yolov8m.pt')

def filter_and_count(results):
    counts = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        counts[label] = counts.get(label, 0) + 1
    return counts

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = int(frame_count / fps)

    frames_list = []
    cumulative_counts = defaultdict(int)

    for sec in range(duration_sec):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 360))
        results = model(resized_frame, verbose=False)[0]
        counts = filter_and_count(results)

        # Update cumulative counts
        for label, count in counts.items():
            cumulative_counts[label] += count

        # Draw bounding boxes and labels
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        frames_list.append((resized_frame, counts))

    cap.release()
    return frames_list, cumulative_counts

# Streamlit UI
st.title("Object Detection and Counting Over Entire Video")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing video... Please wait.")
    frames_data, cumulative_counts = process_video("temp_video.mp4")

    st.write("Displaying processed frames:")

    for i, (frame, counts) in enumerate(frames_data):
        st.image(frame, channels="BGR", caption=f"Second {i + 1}", width=1000)
        
        with st.expander(f"Object count for second {i + 1}"):
            for label, count in counts.items():
                st.write(f"**{label}**: {count}")
        
        time.sleep(0.2)

    # Display cumulative counts
    st.subheader("Cumulative Object Count Across Entire Video")
    for label, total in cumulative_counts.items():
        st.write(f"**{label}**: {total}")