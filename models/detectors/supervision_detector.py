"""
supervision_detector.py - Supervision-based object detector implementation (corrected)
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO

from core.interfaces import ObjectDetector, Detection


class SupervisionDetector(ObjectDetector):
    """Supervision-based object detector using Ultralytics YOLO backend"""
    
    # COCO dataset class IDs for vehicles and persons
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    PERSON_CLASS_ID = 0  # person
    
    MODEL_SIZES = {
        "small": "yolov8s.pt",
        "medium": "yolov8m.pt",
        "large": "yolov8l.pt",
    }
    
    def __init__(self, 
                 model_size: str = "medium", 
                 confidence_threshold: float = 0.25,
                 custom_model_path: Optional[str] = None):
        """
        Initialize Supervision detector
        
        Args:
            model_size: Size of the model (small, medium, large)
            confidence_threshold: Detection confidence threshold
            custom_model_path: Path to custom model weights (overrides model_size if provided)
        """
        self._model_size = model_size.lower()
        self.confidence_threshold = confidence_threshold
        
        # Set up model path
        if custom_model_path and os.path.exists(custom_model_path):
            self.model_path = custom_model_path
        else:
            if self._model_size not in self.MODEL_SIZES:
                raise ValueError(f"Invalid model size: {model_size}. "
                              f"Choose from {list(self.MODEL_SIZES.keys())}")
            self.model_path = self.MODEL_SIZES[self._model_size]
        
        # Initialize YOLO model
        self.model = YOLO(self.model_path)
        
        # Create Supervision detection annotator for visualization (optional)
        self.box_annotator = sv.BoxAnnotator()
        
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in the image using Supervision with YOLO
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        # Run YOLO detection
        yolo_results = self.model(image, conf=self.confidence_threshold)[0]
        
        # Convert to supervision Detections format
        sv_detections = sv.Detections.from_ultralytics(yolo_results)
        
        # Convert to our Detection format
        result_detections = []
        
        if len(sv_detections):
            for i in range(len(sv_detections.xyxy)):
                box = sv_detections.xyxy[i]
                class_id = int(sv_detections.class_id[i])
                confidence = float(sv_detections.confidence[i])
                
                # Filter for vehicle classes and person class
                if class_id in self.VEHICLE_CLASS_IDS or class_id == self.PERSON_CLASS_ID:
                    result_detections.append(Detection(
                        box=box,
                        class_id=class_id,
                        confidence=confidence,
                        is_person=class_id == self.PERSON_CLASS_ID,
                        is_vehicle=class_id in self.VEHICLE_CLASS_IDS
                    ))
        
        return result_detections
    
    @property
    def name(self) -> str:
        """Return the name/type of the detector"""
        return "Supervision"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size