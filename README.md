# Video Object Detection with YOLOv8

A Python-based solution for detecting objects in video streams using Ultralytics YOLOv8. Detects 80+ common objects (people, vehicles, animals, etc.) and displays them with bounding boxes and labels.

## 📺 Demo 

[![YOLOv8 Video Object Detection Demo](https://img.youtube.com/vi/DSTCFva7BrY/0.jpg)](https://youtu.be/DSTCFva7BrY)

*Click the image above to watch the demo video*


## Technologies Used
- **Python 3.8+**
- **Ultralytics YOLOv8** - Object detection model
- **OpenCV** - Video processing and visualization
- **PyTorch** - Deep learning backend
- **COCO Dataset** - 80-class object detection dataset


## How It Works
1. **Video Input**: Reads video frames using OpenCV
2. **Object Detection**:
   - Uses YOLOv8n (nano version) for fast inference
   - Processes each frame through the neural network
3. **Visualization**:
   - Draws bounding boxes around detected objects
   - Labels objects with text
4. **Termination**:
   - Press `ESC` to exit anytime
   - Automatic cleanup on exit


## Setup Instructions

### 1. Prerequisites
- Python 3.8 or later
- pip package manager

### 2. Create Virtual Environment
```bash
python -m venv objdetect_env
source objdetect_env/bin/activate  # Linux/Mac
objdetect_env\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install torch ultralytics opencv-python
```

## Usage

### 1. Configure Video Input
- Replace `traffic.mp4` with your video path:
  ```python
  # In main() function:
  VIDEO_PATH = "your-video.mp4"  # Update this line
  ```

### 2. Run the Detector
```bash
python object_detector.py
```

### 3. Controls
- Press `ESC` to exit
- Video will automatically stop at end


## Requirements File
Create `requirements.txt`:
```
torch>=2.0.0
ultralytics>=8.0.0
opencv-python>=4.7.0
```


## Acknowledgments
- YOLOv8 by Ultralytics
- COCO Dataset by Microsoft
- OpenCV for video processing
