import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np

class DepthStream:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to('cuda' if torch.cuda.is_available() else 'cpu')

    def predict_depth(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        depth = self.processor.post_process_depth_estimation(outputs, target_sizes=[(frame.shape[0], frame.shape[1])])
        predicted_depth = depth[0]["predicted_depth"].cpu().numpy()
        
        depth_normalized = (predicted_depth * 255 / np.max(predicted_depth)).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        return depth_colored
