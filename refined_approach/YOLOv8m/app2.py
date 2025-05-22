import cv2
import time
from ultralytics import YOLO
import streamlit as st
from collections import defaultdict
import numpy as np

# Load YOLO model
model = YOLO('yolov8m.pt')

# Tracks object appearances by sec and position
previous_objects = {}

def get_box_center_and_area(box):
    x1, y1, x2, y2 = box
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    area = (x2 - x1) * (y2 - y1)
    return center, area

def is_object_moving_toward(last_area, current_area):
    return current_area > last_area  # coming closer

def filter_and_count(results, sec, motion_track):
    current_objects = []
    counts = {}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center, area = get_box_center_and_area((x1, y1, x2, y2))

        # Identify new object based on proximity and label
        matched = False
        for prev_obj in previous_objects.get(sec - 1, []):
            if prev_obj["label"] == label:
                prev_center = prev_obj["center"]
                dist = np.linalg.norm(np.array(center) - np.array(prev_center))
                if dist < 50:  # Threshold for spatial closeness
                    if is_object_moving_toward(prev_obj["area"], area):
                        counts[label] = counts.get(label, 0) + 1
                        motion_track.append({
                            "label": label, "status": "towards", "center": center
                        })
                    else:
                        motion_track.append({
                            "label": label, "status": "away", "center": center
                        })
                    matched = True
                    break
        if not matched and sec == 0:  # First second, no past data
            counts[label] = counts.get(label, 0) + 1
            motion_track.append({
                "label": label, "status": "towards", "center": center
            })
        current_objects.append({
            "label": label,
            "center": center,
            "area": area
        })

    previous_objects[sec] = current_objects
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

        motion_track = []
        counts = filter_and_count(results, sec, motion_track)

        # Update cumulative counts
        for label, count in counts.items():
            cumulative_counts[label] += count

        # Draw bounding boxes
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            status = ""
            for m in motion_track:
                if m["label"] == label and abs(m["center"][0] - ((x1 + x2)//2)) < 5:
                    status = m["status"]
                    break

            color = (0, 255, 0) if status == "towards" else (0, 0, 255)
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resized_frame, f"{label} {conf:.2f} {status}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw count at the bottom
        y = 30
        for label, count in counts.items():
            cv2.putText(resized_frame, f"{label}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            y += 30
        cv2.putText(resized_frame, f"Second: {sec + 1}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        frames_list.append(resized_frame)

    cap.release()
    return frames_list, cumulative_counts

# Streamlit UI
st.title("Object Counting (Only Moving Towards Camera)")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing video... Please wait.")
    frames, cumulative_counts = process_video("temp_video.mp4")

    st.write("Showing analyzed frames:")
    for i, frame in enumerate(frames):
        st.image(frame, channels="BGR", caption=f"Second {i + 1}", width=1000)
        time.sleep(0.2)

    st.markdown("### 📊 Cumulative Object Count (Only Moving Toward Camera)")
    for label, count in cumulative_counts.items():
        st.write(f"- **{label}**: {count}")