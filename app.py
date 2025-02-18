from flask import Flask, render_template, request, Response
from object_detector import VideoObjectDetector
import cv2
import os
import time
import threading
from queue import Queue

app = Flask(__name__)

# Configuration
MODEL_PATH = "yolo-Weights/yolov8n.pt"
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer",
    "toothbrush"
]

# Initialize detector
detector = VideoObjectDetector(MODEL_PATH, COCO_CLASSES)

# Global variables for video processing
frame_queue = Queue(maxsize=10)
stop_event = threading.Event()

def video_processing_task():
    global frame_queue, stop_event
    
    temp_path = 'temp_video.mp4'
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1/fps if fps > 0 else 1/30
    
    while cap.isOpened() and not stop_event.is_set():
        start_time = time.time()
        
        success, frame = cap.read()
        if not success:
            break
        
        # Process frame
        processed_frame = detector.process_frame(frame.copy())
        
        # Convert frames to JPEG
        _, orig_buffer = cv2.imencode('.jpg', frame)
        _, proc_buffer = cv2.imencode('.jpg', processed_frame)
        
        # Put frames in queue
        if not frame_queue.full():
            frame_queue.put((orig_buffer.tobytes(), proc_buffer.tobytes()))
        
        # Maintain original video speed
        elapsed = time.time() - start_time
        time.sleep(max(frame_delay - elapsed, 0))
    
    cap.release()
    stop_event.set()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global stop_event, frame_queue
    
    if 'video' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400
    
    # Clear previous processing
    stop_event.set()
    with frame_queue.mutex:
        frame_queue.queue.clear()
    
    # Save uploaded file
    temp_path = 'temp_video.mp4'
    file.save(temp_path)
    
    # Start new processing thread
    stop_event = threading.Event()
    threading.Thread(target=video_processing_task).start()
    
    return '', 204

def generate_feed(feed_type: str):
    while not stop_event.is_set():
        try:
            orig_frame, proc_frame = frame_queue.get(timeout=1)
            if feed_type == 'original':
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + orig_frame + b'\r\n')
            else:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + proc_frame + b'\r\n')
        except:
            break

@app.route('/original_feed')
def original_feed():
    return Response(generate_feed('original'),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def processed_feed():
    return Response(generate_feed('processed'),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)