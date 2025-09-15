import numpy as np
from typing import List, Dict, Tuple

class ProximityAnalyzer:
    def __init__(self):
        # Thresholds for normalized depth values (0-255)
        self.DANGER_THRESHOLD = 60
        self.WARNING_THRESHOLD = 120

    def get_proximity_status(self, depth_map: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> str:
        """
        Get overall proximity status based on objects' depths
        
        Args:
            depth_map: Normalized depth map (0-255)
            boxes: List of bounding boxes (x1, y1, x2, y2)
        
        Returns:
            'DANGER', 'WARNING', or 'SAFE'
        """
        if not boxes:
            return 'SAFE'
            
        avg_depths = []
        for x1, y1, x2, y2 in boxes:
            region = depth_map[y1:y2, x1:x2]
            if region.size > 0:
                avg_depths.append(np.mean(region))
        
        if not avg_depths:
            return 'SAFE'
            
        min_depth = min(avg_depths)
        if min_depth >= self.DANGER_THRESHOLD:
            return 'DANGER'
        elif min_depth >= self.WARNING_THRESHOLD:
            return 'WARNING'
        return 'SAFE'