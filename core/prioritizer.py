"""
prioritizer.py - Determines the main/active object in a scene
"""

from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np

from core.interfaces import ObjectPrioritizer, Detection, ClassificationResult


class DefaultPrioritizer(ObjectPrioritizer):
    """Determines the main/active object in a scene based on position, size, and class"""
    
    def __init__(self):
        """Initialize prioritizer with weights and parameters"""
        # Weights for scoring
        self.center_weight = 2.0     # Weight for object proximity to center
        self.size_weight = 1.1       # Weight for object size
        self.person_weight = 0.9     # Extra weight for persons
        self.emergency_weight = 1.5  # Extra weight for emergency vehicles
        
        # Thresholds
        self.min_main_object_score = 0.5  # Minimum score to be considered "main"
    
    def is_likely_stationary(self, box: np.ndarray, image_shape: Tuple[int, int, int], 
                            motion_region: Optional[Tuple[float, float, float, float]] = None) -> bool:
        """
        Determine if an object is likely parked/stationary based on position
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            image_shape: Image shape (height, width, channels)
            motion_region: Optional region where objects are considered in motion
            
        Returns:
            Boolean indicating if object is likely stationary
        """
        img_height, img_width = image_shape[:2]
        x1, y1, x2, y2 = box
        
        if motion_region is None:
            # Central region (middle 60% of image)
            center_x1 = img_width * 0.2
            center_y1 = img_height * 0.2
            center_x2 = img_width * 0.8
            center_y2 = img_height * 0.8
            motion_region = (center_x1, center_y1, center_x2, center_y2)
        
        # Calculate center of box
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        
        # Check if center is in motion region
        if (motion_region[0] <= box_center_x <= motion_region[2] and
            motion_region[1] <= box_center_y <= motion_region[3]):
            return False  # Not likely stationary
        
        # Check if close to image edges
        edge_margin = min(img_width, img_height) * 0.1
        if (x1 < edge_margin or 
            y1 < edge_margin or 
            x2 > img_width - edge_margin or 
            y2 > img_height - edge_margin):
            return True  # Likely stationary
        
        return False
    
    def find_main_object(self, 
                         detections: List[Detection],
                         classifications: List[ClassificationResult],
                         image_shape: Tuple[int, int, int]) -> Tuple[Optional[int], float]:
        """
        Find the main/active object in the scene
        
        Args:
            detections: List of Detection objects
            classifications: List of ClassificationResult objects
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Tuple of (index of main object, score) or (None, 0.0) if no main object
        """
        if not detections or not classifications:
            return None, 0.0
            
        img_center_x = image_shape[1] / 2
        img_center_y = image_shape[0] / 2
        object_scores = []
        
        for i, (detection, classification) in enumerate(zip(detections, classifications)):
            box = detection.box
            is_person = detection.is_person
            class_name = classification.class_name
            
            box_center_x = (box[0] + box[2]) / 2
            box_center_y = (box[1] + box[3]) / 2
            
            # Distance from image center (normalized by image dimensions)
            distance = np.sqrt(
                ((box_center_x - img_center_x) / image_shape[1])**2 + 
                ((box_center_y - img_center_y) / image_shape[0])**2
            )
            
            # Centrality score (1 at center, 0 at corners)
            centrality = 1 - min(distance, 1.0)
            
            # Size score (normalized by image area)
            area = (box[2] - box[0]) * (box[3] - box[1]) / (image_shape[0] * image_shape[1])
            
            # Apply penalties and bonuses
            stationary_penalty = 0.7 if self.is_likely_stationary(box, image_shape) else 1.0
            person_bonus = self.person_weight if is_person or class_name == "person" else 1.0
            emergency_bonus = self.emergency_weight if class_name == "emergency vehicle" else 1.0
            
            # Skip non-vehicle/non-person items
            if class_name == "no-vehicle" and not is_person:
                score = 0
            else:
                # Combined score (higher is better)
                score = (
                    (centrality * self.center_weight) + 
                    (area * self.size_weight)
                ) * stationary_penalty * person_bonus * emergency_bonus
            
            object_scores.append(score)
        
        # Find object with highest score above threshold
        if object_scores and max(object_scores) >= self.min_main_object_score:
            main_idx = np.argmax(object_scores)
            return main_idx, object_scores[main_idx]
        
        return None, 0.0