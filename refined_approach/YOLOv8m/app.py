import streamlit as st
import cv2
import tempfile
from detect_count import model, filter_and_count, reset_counters

st.title("Vehicle and Person Counting (1 Frame Per Second)")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Reset tracker and counters
    reset_counters()

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = int(total_frames / fps)

    st.write(f"Video FPS: {fps}")
    st.write(f"Duration: {duration_sec} seconds")

    counts_per_second = []

    st.write("### Counts every 1 second:")

    for sec in range(duration_sec):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)  # Go to frame at that second
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance (optional)
        resized_frame = cv2.resize(frame, (640, 360))

        # Run YOLO detection
        results = model(resized_frame, verbose=False)[0]

        # Count filtered objects
        counts = filter_and_count(results)
        counts_per_second.append((sec + 1, counts))

        st.write(f"Second {sec + 1}: {counts}")

    cap.release()

    st.write("### Summary Table:")
    st.table([{"Second": sec, **cnts} for sec, cnts in counts_per_second])
