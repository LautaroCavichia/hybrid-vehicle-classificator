"""
yolo_detector.py - YOLO-based object detector implementations
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO

from core.interfaces import ObjectDetector, Detection


class YOLODetector(ObjectDetector):
    """YOLO-based object detector"""
    
    # COCO dataset class IDs for vehicles and persons
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    PERSON_CLASS_ID = 0  # person
    
    MODEL_SIZES = {
        "nano": "yolov8n.pt",
        "small": "yolov8s.pt",
        "medium": "yolov8m.pt",
        "large": "yolov8l.pt",
        "xlarge": "yolov8x.pt"
    }
    
    def __init__(self, 
                 model_size: str = "medium", 
                 confidence_threshold: float = 0.25,
                 custom_model_path: Optional[str] = None):
        """
        Initialize YOLO detector
        
        Args:
            model_size: Size of the model (nano, small, medium, large, xlarge)
            confidence_threshold: Detection confidence threshold
            custom_model_path: Path to custom model weights (overrides model_size if provided)
        """
        self._model_size = model_size.lower()
        self.confidence_threshold = confidence_threshold
        
        # Load model
        if custom_model_path and os.path.exists(custom_model_path):
            self.model_path = custom_model_path
        else:
            if self._model_size not in self.MODEL_SIZES:
                raise ValueError(f"Invalid model size: {model_size}. "
                              f"Choose from {list(self.MODEL_SIZES.keys())}")
            self.model_path = self.MODEL_SIZES[self._model_size]
        
        self.model = YOLO(self.model_path)
        
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in the image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        results = self.model(image, conf=self.confidence_threshold)
        
        detections = []
        
        if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes[i].xyxy.cpu().numpy()[0]
                
                class_id = int(boxes[i].cls.cpu().numpy()[0])
                
                confidence = float(boxes[i].conf.cpu().numpy()[0])
                
                # Filter for vehicle classes and person class
                if class_id in self.VEHICLE_CLASS_IDS or class_id == self.PERSON_CLASS_ID:
                    detections.append(Detection(
                        box=box,
                        class_id=class_id,
                        confidence=confidence,
                        is_person=class_id == self.PERSON_CLASS_ID,
                        is_vehicle=class_id in self.VEHICLE_CLASS_IDS
                    ))
        
        return detections
    
    @property
    def name(self) -> str:
        """Return the name/type of the detector"""
        return "YOLO"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size


class YOLOv5Detector(YOLODetector):
    """YOLOv5-based object detector"""
    
    MODEL_SIZES = {
        "nano": "yolov5n.pt",
        "small": "yolov5s.pt",
        "medium": "yolov5m.pt",
        "large": "yolov5l.pt", 
        "xlarge": "yolov5x.pt"
    }
    
    @property
    def name(self) -> str:
        """Return the name/type of the detector"""
        return "YOLOv5"