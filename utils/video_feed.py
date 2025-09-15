import cv2
import threading
import numpy as np
from models.yolo import YOLOStream
from models.depth import DepthStream
from utils.proximity_analyzer import ProximityAnalyzer

CAM = 1

class VideoFeed:
    def __init__(self):
        self.yolo_stream = YOLOStream()
        self.depth_stream = DepthStream()
        self.proximity_analyzer = ProximityAnalyzer()
        self.cap = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.current_proximity = 'SAFE'
        self.proximity_lock = threading.Lock()

    def start_capture(self):
        """Initialize and start the video capture in a separate thread"""
        self.cap = cv2.VideoCapture(CAM)
        self.running = True
        thread = threading.Thread(target=self._update_frame, daemon=True)
        thread.start()

    def _update_frame(self):
        """Continuously update the frame from the camera"""
        while self.running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.frame_lock:
                self.frame = frame

    def stop_capture(self):
        """Stop the video capture"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_current_frame(self):
        """Get the most recent frame safely"""
        with self.frame_lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_proximity_status(self):
        with self.proximity_lock:
            return self.current_proximity

    def _process_frame(self, frame):
        """Process frame through both models and update proximity status"""
        # Get YOLO detections
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo_stream.model(rgb_frame)
        
        # Get depth map
        depth_map = self.depth_stream.predict_depth(frame)
        
        # Get bounding boxes from YOLO results
        boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes.data:
                x1, y1, x2, y2 = map(int, box[:4])
                boxes.append((x1, y1, x2, y2))
        
        # Update proximity status
        with self.proximity_lock:
            self.current_proximity = self.proximity_analyzer.get_proximity_status(depth_map, boxes)

        # Convert YOLO result back to BGR for OpenCV
        yolo_frame = results[0].plot()
        yolo_frame = cv2.cvtColor(yolo_frame, cv2.COLOR_RGB2BGR)
        
        return yolo_frame, depth_map

    def generate_yolo_feed(self):
        if not self.running:
            self.start_capture()
            
        while self.running:
            frame = self.get_current_frame()
            if frame is None:
                continue
                
            yolo_frame, _ = self._process_frame(frame)  # Fixed unpacking
            success, buffer = cv2.imencode('.jpg', yolo_frame)
            if not success:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def generate_depth_feed(self):
        if not self.running:
            self.start_capture()
            
        while self.running:
            frame = self.get_current_frame()
            if frame is None:
                continue
                
            _, depth_map = self._process_frame(frame)  # Fixed unpacking
            success, buffer = cv2.imencode('.jpg', depth_map)
            if not success:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')