Object Detection

To understand traffic patterns and audience engagement potential for ad boards, possibly to inform ad targeting, placement, or billing.

- Setup a vitual env with python -m venv venv and activate it using source venv/bin/activate OR venv\Scripts\activate on Windows. pip install -r requirements.txt to install the dependecies on the requirements.txt. The interpreter will be the one in the venv folder. streamlit run streamlit_app.py to run the application on streamlit.

- Different ways of trying out the models are:
1. Cloud based tools: When you want good ui experience, and no GPU locally
2. Edge and Real time deployment using NVIDIA DeepStream maybe from Jetson: 24X7 live completely
3. Local Inference: More control over your model
4. Model Benchmarking Frameworks using MMDetection: For comparing multiple models on our own dataset
5. Fine Tuned models: When we do not get right results from our pretrained models, we fine tune our models to give us better results

- The different models going to be used during this application:
One stage detectors
YOLOv1 – YOLOv4
YOLOv5
YOLOv6
YOLOv7
YOLOv8
PP-YOLO / PP-YOLOE
YOLOX
EfficientDet
RetinaNet
SSD (Single Shot Detector)
CenterNet
CenterTrack
CornerNet
FCOS

Two stage detectors
Faster R-CNN
Mask R-CNN
Cascade R-CNN
Libra R-CNN
R-FCN

Transformer-Based Detectors
DETR
Deformable DETR
DINO DETR
YOLOS
Sparse R-CNN
QueryInst

Pose / Keypoint-Based Detectors (For humans)
CenterNet
PoseNet
HRNet
OpenPose
YOLOv7-pose
YOLOv8-pose

Multi-Task / Frameworks (Support many models)
Detectron2
MMDetection
TensorFlow Object Detection API
