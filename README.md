# Autonomous Transformation Drone - Victim Detection using YOLOv5
 
**Pushpak Viman** is an initiative focused on autonomous rescue missions using aerial vehicles. A core part of this system involves using computer vision to detect and classify features from satellite and aerial imagery, helping in navigation and victim detection tasks.
 
In this repository, we leverage the YOLOv5 object detection framework and a satellite imagery dataset sourced from:
 
- **MDPI Remote Sensing Journal**: [Land Cover Classification from Satellite Imagery](https://www.mdpi.com/2072-4292/14/13/2977)
 
---
 
## 📈 Dataset Details
 
- **Source**: MDPI Remote Sensing
- **Content**: Labeled satellite images for land cover and features.
- **Format**: YOLO format with `.txt` annotation files.
 
Directory structure:
 
```
/dataset
  /images
    /train
    /val
  /labels
    /train
    /val
```
 
---
 
## 🚀 Getting Started
 
### Installation
 
```bash
# Clone the repository
$ git clone https://github.com/your-username/pushpak-viman-yolov5.git
$ cd pushpak-viman-yolov5
 
# Install dependencies
$ pip install -r requirements.txt
```
 
Ensure Python 3.8+ and PyTorch are installed.
 
---
 
### Training
 
```bash
python train.py --img 640 --batch 16 --epochs 100 --data data/my_dataset.yaml --weights yolov5s.pt --name pushpakviman
```
 
### Detection
 
```bash
python detect.py --weights runs/train/pushpakviman/weights/best.pt --img 640 --conf 0.25 --source dataset/images/val
```
 
### Validation
 
```bash
python val.py --weights runs/train/pushpakviman/weights/best.pt --data data/my_dataset.yaml --img 640
```
 
---
 
## 📈 Results

 ![output2](https://github.com/user-attachments/assets/e6af8c5d-fc5f-43e9-a29b-567c3cbbaa2b)
 
---
 
## 📊 Model Information
 
- Starting Model: `yolov5s.pt`
- Custom fine-tuning on satellite images
- Adjustable architectures available in `models/`
 
---
 
## ✨ Acknowledgements
 
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [MDPI Remote Sensing Dataset](https://www.mdpi.com/2072-4292/14/13/2977)
 
---
 
## 🙌 Contributing
 
Contributions are welcome! Please open an issue to discuss changes before submitting a pull request.
 
---
 
