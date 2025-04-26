# Autonomous Transformation Drone - Victim Detection using Computer Vision
 
**Pushpak Viman** is an initiative focused on autonomous rescue missions using aerial vehicles. A core part of this system involves using computer vision to detect and classify features from aerial and land imagery, helping in navigation and victim detection tasks.

![WhatsApp Image 2024-09-30 at 12 50 51_89777784](https://github.com/user-attachments/assets/05abb4bc-e08b-40fa-9d03-33572dc93349)


 
In this repository, we work with a satellite imagery dataset sourced from:
 
- **MDPI Remote Sensing Journal**: [Land Cover Classification from Satellite Imagery](https://www.mdpi.com/2072-4292/14/13/2977)

---

## Aim

This repository provides a complete pipeline for:
- Preparing and splitting a victim-under-debris dataset
- Converting and organizing annotations
- Training and running object detection models
- Real-time inference and visualization
- Laying the groundwork for drone integration in disaster response scenarios

---

## Features

- **Automated dataset splitting** for robust machine learning workflows
- **Annotation conversion** from Pascal VOC XML to CSV and TFRecord formats
- **Object detection model support** (TensorFlow Lite, YOLOv5)
- **Real-time video and image inference** with bounding box visualization
- **Hand landmark tracking** for gesture-based controls or victim identification
- **Visualization tools** for rapid result assessment

---

## Installation

1. **Clone the repository:**
git clone https://github.com/fai05/Project-Pushpak-Viman-SIH.git


2. cd Project-Pushpak-Viman-SIH


3. **Install dependencies:**
pip install opencv-python mediapipe tensorflow torch pandas

Dependencies
- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- TensorFlow / TensorFlow Lite
- PyTorch (for YOLOv5)
- Pandas
- xml.etree.ElementTree (standard library)
- os, pathlib, shutil, random (standard library)


---

## Usage

### 1. Dataset Preparation

Split your dataset (images and VOC XML annotations) into training, validation, and test sets:
python victimdataset.py

- Organizes your data into `train/`, `validation/`, and `test/` folders for streamlined training and evaluation.

---

### 2. Annotation Conversion

Convert XML annotations to CSV for easier processing:
python create_csv.py

- Generates `train_labels.csv` and `validation_labels.csv` for model training.

---

### 3. TFRecord Generation

Create TFRecords and label maps for TensorFlow Object Detection API:
python create_tfrecord.py

- Outputs `.tfrecord` files and `labelmap.pbtxt`.

---

### 4. Real-Time Hand Tracking

Demo for real-time hand landmark detection (can be adapted for gesture-based drone control):
python signlanguagemediapipe.py

- Uses your webcam to display detected hand landmarks.

---

### 5. Object Detection Inference
python mayura.py

#### TensorFlow Lite (Webcam):

- Runs object detection on live video using a TFLite model.

#### YOLOv5 (Image):
- Ensure YOLOv5 weights and an input image are available.
- The script will load the model and display detection results.

---

### 6. Visualization Example

Draw rectangles (e.g., bounding boxes) on images for quick visualization:
python trialopencv.py

- Loads, annotates, displays, and saves images with rectangles.
 ![output2](https://github.com/user-attachments/assets/e6af8c5d-fc5f-43e9-a29b-567c3cbbaa2b)

---
 
## Contribution
 
Contributions are welcome! Please open an issue to discuss changes before submitting a pull request.
 
---
