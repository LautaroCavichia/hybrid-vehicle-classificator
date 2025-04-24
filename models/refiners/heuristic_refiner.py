"""
heuristic_refiner.py - Refines vehicle classifications based on size and context
"""

from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np

from core.interfaces import ClassificationRefiner, ClassificationResult, Detection


class HeuristicRefiner(ClassificationRefiner):
    """Refines vehicle classifications based on size and other heuristics"""
    
    def __init__(self):
        """Initialize the refiner"""
        # Size thresholds for vehicle classification (pixels²)
        self.car_max_area = 42500    # Maximum area for a car
        self.van_max_area = 135000   # Maximum area for a van
        self.bus_max_area = 250000   # Maximum area for a bus
        # Above bus_max_area is considered a heavy truck
        
        # Aspect ratio ranges (width/height)
        self.car_aspect_ratio_range = (1.0, 2.2)
        self.van_aspect_ratio_range = (1.2, 2.5)
        self.bus_aspect_ratio_range = (1.5, 4.0)
        self.truck_aspect_ratio_range = (1.5, 3.5)
    
    def refine(self, 
               image: np.ndarray,
               detection: Detection, 
               classification: ClassificationResult,
               image_shape: Tuple[int, int, int]) -> ClassificationResult:
        """
        Refine the classification based on additional heuristics
        
        Args:
            image: Full input image
            detection: Detection object
            classification: Initial classification result
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Refined ClassificationResult
        """
        # Handle special cases that don't need refinement
        if classification.class_name == "person" or detection.is_person:
            return ClassificationResult(class_name="person", confidence=0.99)
            
        # Get the current class and confidence
        current_class = classification.class_name
        current_conf = classification.confidence
        
        # Extract size and shape information
        box = detection.box
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        aspect_ratio = width / max(height, 1)
        
        # Get the class from YOLO detection (useful for coarse categories)
        class_id = detection.class_id
        
        # REFINEMENT RULE 1: Class-based rules
        if class_id == 7:  # YOLO truck class
            # If model is highly confident it's not a truck, keep its decision
            if current_class != "truck" and current_conf > 0.9:
                return classification
            # Otherwise trust YOLO's truck detection
            return ClassificationResult(class_name="truck", confidence=0.85)
            
        elif class_id == 5:  # YOLO bus class
            # If model is highly confident it's not a bus, keep its decision
            if current_class != "bus" and current_conf > 0.9:
                return classification
            # Otherwise trust YOLO's bus detection
            return ClassificationResult(class_name="bus", confidence=0.85)
            
        elif class_id == 3 and area < 10000:  # Small motorcycle
            return ClassificationResult(class_name="no-vehicle", confidence=0.8)
        
        # REFINEMENT RULE 2: Size-based rules
        if area < self.car_max_area:
            # Small vehicle - likely a car
            # But if classifier is very confident, trust it
            if current_conf > 0.85:
                return classification
                
            # Check if aspect ratio matches car profile
            if self.car_aspect_ratio_range[0] <= aspect_ratio <= self.car_aspect_ratio_range[1]:
                return ClassificationResult(class_name="car", confidence=0.75)
                
        elif area < self.van_max_area:
            # Medium vehicle - likely a van or car
            # But if classifier is very confident, trust it
            if current_conf > 0.85:
                return classification
                
            # Check aspect ratio
            if self.van_aspect_ratio_range[0] <= aspect_ratio <= self.van_aspect_ratio_range[1]:
                return ClassificationResult(class_name="van", confidence=0.75)
            elif self.car_aspect_ratio_range[0] <= aspect_ratio <= self.car_aspect_ratio_range[1]:
                return ClassificationResult(class_name="car", confidence=0.7)
                
        elif area < self.bus_max_area:
            # Large vehicle - likely a bus or truck
            # But if classifier is very confident, trust it
            if current_conf > 0.85:
                return classification
                
            # Check aspect ratio
            if self.bus_aspect_ratio_range[0] <= aspect_ratio <= self.bus_aspect_ratio_range[1]:
                return ClassificationResult(class_name="bus", confidence=0.75)
            elif self.truck_aspect_ratio_range[0] <= aspect_ratio <= self.truck_aspect_ratio_range[1]:
                return ClassificationResult(class_name="truck", confidence=0.75)
                
        elif area >= self.bus_max_area:
            # Very large vehicle - likely a truck
            # But if classifier is very confident, trust it
            if current_conf > 0.85:
                return classification
                
            return ClassificationResult(class_name="truck", confidence=0.8)
        
        # REFINEMENT RULE 3: Emergency vehicle detection
        # This would typically use color analysis to detect emergency vehicle patterns
        # For simplicity, we're relying on the classifier result
        
        # If no specific rule was triggered, return the original classification
        return classification