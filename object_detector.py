from ultralytics import YOLO
import cv2
from typing import List, Tuple

class VideoObjectDetector:
    def __init__(self, model_path: str, class_labels: List[str]):
        self.model = YOLO(model_path)
        self.classes = class_labels
        self.box_color = (0, 255, 0)
        self.text_color = (255, 255, 0)
        self.font = cv2.FONT_HERSHEY_DUPLEX

    def process_frame(self, frame):
        detections = self.model(frame, stream=True)
        for result in detections:
            self._draw_predictions(frame, result.boxes)
        return frame

    def _draw_predictions(self, frame, boxes):
        for box in boxes:
            self._draw_box(frame, box)
            self._draw_class(frame, box)

    def _get_box_coordinates(self, box) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        return x1, y1, x2, y2

    def _draw_box(self, frame, box):
        x1, y1, x2, y2 = self._get_box_coordinates(box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)

    def _draw_class(self, frame, box):
        x1, y1, _, _ = self._get_box_coordinates(box)
        class_id = int(box.cls[0])
        if 0 <= class_id < len(self.classes):
            label = f"{self.classes[class_id].upper()}"
            cv2.putText(frame, label, (x1, y1-5), self.font, 0.7, 
                       self.text_color, 1, cv2.LINE_AA)

def main():
    # Configuration
    MODEL_PATH = "yolo-Weights/yolov8n.pt"
    VIDEO_PATH = "traffic.mp4"
    
    # COCO classes (corrected - removed "cheetah")
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

    # Initialize video capture
    video = cv2.VideoCapture(VIDEO_PATH)
    if not video.isOpened():
        raise IOError("Cannot open video file")

    # Create detector
    detector = VideoObjectDetector(MODEL_PATH, COCO_CLASSES)

    # Processing loop
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        processed_frame = detector.process_frame(frame)
        cv2.imshow('Object Analysis', processed_frame)
        
        if cv2.waitKey(1) == 27:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()