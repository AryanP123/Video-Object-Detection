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