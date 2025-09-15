import cv2
import torch
from ultralytics import YOLO

class YOLOStream:
    def __init__(self, model_path="yolov8n.pt"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)

    def predict_objects(self, frame):
        """ Run object detection on an input frame and return annotated frame """
        # Convert BGR to RGB as YOLO expects RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model(rgb_frame)

        # Annotate the frame and convert back to BGR for OpenCV display
        annotated_frame = results[0].plot()
        return cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
