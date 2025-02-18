# Video Object Detection with YOLOv8

A Python-based solution for detecting objects in video streams using Ultralytics YOLOv8. Detects 80+ common objects (people, vehicles, animals, etc.) and displays them with bounding boxes and labels.

## 📺 Demo 

[![YOLOv8 Video Object Detection Demo](https://img.youtube.com/vi/DSTCFva7BrY/0.jpg)](https://youtu.be/DSTCFva7BrY)

*Click the image above to watch the demo video*


## Technologies

**Backend:**
- Python 3.8+
- Flask (Web Framework)
- YOLOv8 (Object Detection)
- OpenCV (Video Processing)
- PyTorch (Deep Learning Backend)

**Frontend:**
- HTML5/CSS3
- JavaScript
- Multipart JPEG Streaming

**Dataset:**
- COCO (Common Objects in Context) 80-class


## How It Works
1. **Video Input**: Reads video frames using OpenCV
2. **Object Detection**:
   - Uses YOLOv8n (nano version) for fast inference
   - Processes each frame through the neural network
3. **Visualization**:
   - Draws bounding boxes around detected objects
   - Labels objects with text


### Installation

1. **Clone Repository**

2. **Set Up Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
pip install ultralytics Flask
```


### Usage

1. **Start Development Server**
```bash
python app.py
```

2. **Access Web Interface**  
Open `http://localhost:5001` in your browser

3. **Upload & Process Video**
1. Click "Choose File" and select an MP4 video
2. Click "Upload & Process"
3. View real-time detection results
4. Reload page to process new video


## Acknowledgments
- YOLOv8 by Ultralytics
- COCO Dataset by Microsoft
- OpenCV for video processing
