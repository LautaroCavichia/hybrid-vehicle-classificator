"""
yolo_world_wrapper.py - Wrapper for integrating YOLO-World into the benchmark framework
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any

# Import base interfaces
from core.interfaces import Detection, ClassificationResult

class YOLOWorldWrapper:
    """
    Wrapper for YOLO-World that adapts it to the benchmark framework
    This is an end-to-end system that performs both detection and classification
    """
    
    def __init__(self, model_size="medium", confidence_threshold=0.25, custom_model_path=None):
        """
        Initialize the YOLO-World wrapper
        
        Args:
            model_size: Size of the model (small, medium, large)
            confidence_threshold: Detection confidence threshold
            custom_model_path: Path to custom model weights
        """
        # Import the YOLOWorldVehicleSystem here to avoid circular imports
        from models.yolo_world_standalone import YOLOWorldVehicleSystem
        
        self.system = YOLOWorldVehicleSystem(
            model_size=model_size,
            confidence_threshold=confidence_threshold,
            custom_model_path=custom_model_path
        )
        
        self.name = "yolo_world"
        self.model_size = model_size
        self._last_image = None
        self._last_detections = None
        self._last_results = None
    
    def process_image(self, image: np.ndarray) -> Tuple[List[Detection], List[ClassificationResult]]:
        """
        Process an image to detect and classify vehicles in a single pass
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (detections list, classifications list)
        """
        # Cache the image
        self._last_image = image
        
        # Run YOLO-World inference
        _, main_object, yolo_world_detections = self.system.process_image(image)
        
        # Convert to the formats expected by the benchmark
        detections = []
        classifications = []
        
        for i, det in enumerate(yolo_world_detections):
            # Extract coordinates and convert if needed
            # YOLO-World uses [x1, y1, x2, y2] format which matches the benchmark
            box = det['box']
            
            # Determine if this is a vehicle
            is_vehicle = det['is_vehicle']
            
            # Create Detection object - FIXED: removed object_id parameter
            detection = Detection(
                box=box,
                confidence=det['confidence'],
                class_id=det['class_id'] if is_vehicle else 0,
                is_vehicle=is_vehicle,
                is_person=det['is_person']
            )
            detections.append(detection)
            
            # Create ClassificationResult object
            classification = ClassificationResult(
                class_name=det['class_name'],
                confidence=det['confidence']
            )
            classifications.append(classification)
        
        # Cache the results
        self._last_detections = detections
        self._last_results = classifications
        
        return detections, classifications