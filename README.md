# Video Object Detection with YOLOv8

A Python-based solution for detecting objects in video streams using Ultralytics YOLOv8. Detects 80+ common objects (people, vehicles, animals, etc.) and displays them with bounding boxes and labels.

## 📺 Demo 

[![YOLOv8 Video Object Detection Demo](https://img.youtube.com/vi/t55-aXkhzfQ/0.jpg)](https://youtu.be/t55-aXkhzfQ)

*Click the image above to watch the demo video*


## Technologies

**Backend:**

* Python 3.8+
* Flask (Web Framework)
* YOLOv8 (Object Detection)
* OpenCV (Video Processing)
* PyTorch (Deep Learning Backend)

**Frontend:**

* HTML5/CSS3
* JavaScript

**Dataset:**

* COCO (Common Objects in Context) 80-class

**Deployment:**

* Docker 
* Kubernetes
* Virtual Environment


## How It Works
1. **Video Input**: Reads video frames using OpenCV
2. **Object Detection**:
   - Uses YOLOv8n (nano version) for fast inference
   - Processes each frame through the neural network
3. **Visualization**:
   - Draws bounding boxes around detected objects
   - Labels objects with text


## Installation

1. **Clone Repository**

2. **Choose a Deployment Method:**

   **Option 1: Using Kubernetes**

   * **Pull Docker Image:**
   ```bash
   docker run ghcr.io/aryanp123/myapp:1.0.0
   ```
   (Can ^C after, just needs to retrieve the image)

   * **Apply Deployment and Service:**
     ```bash
     kubectl apply -f deployment.yaml
     kubectl apply -f service.yaml
     ```

   * **Access Web Interface:** Open `http://localhost`


   **Option 2: Using Docker Compose**

   * **Start Application:**
     ```bash
     docker-compose up -d
     ```

   * **Access Web Interface:** Open `http://localhost:3000` in your browser.

   **Option 3: Using a Virtual Environment (without Docker)**

   * **Set Up Virtual Environment:**
     ```bash
     python -m venv venv
     source venv/bin/activate  # Linux/MacOS
     venv\Scripts\activate     # Windows
     ```

   * **Install Dependencies:**
     ```bash
     pip install -r requirements.txt
     ```

   * **Start Development Server:**
     ```bash
     python app.py
     ```

   * **Access Web Interface:** Open `http://localhost:3000` in your browser.

### Usage

1. **Upload & Process Video**

   1. Click "Choose File" and select an MP4 video.
   2. Click "Upload & Process".
   3. View real-time detection results.


## Acknowledgments
- YOLOv8 by Ultralytics
- COCO Dataset by Microsoft
- OpenCV for video processing
