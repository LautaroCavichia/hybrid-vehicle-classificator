"""
retinanet_detector.py - RetinaNet-based object detector implementation
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
import torchvision
import cv2

from core.interfaces import ObjectDetector, Detection


class RetinaNetDetector(ObjectDetector):
    """RetinaNet-based object detector"""
    
    MODEL_SIZES = {
        "small": "retinanet_resnet50_fpn",
        "medium": "retinanet_resnet101_fpn",
    }
    
    def __init__(self, 
                 model_size: str = "small", 
                 confidence_threshold: float = 0.25,
                 custom_model_path: Optional[str] = None):
        """
        Initialize RetinaNet detector
        
        Args:
            model_size: Size of the model (resnet50 or resnet101)
            confidence_threshold: Detection confidence threshold
            custom_model_path: Path to custom model weights (overrides model_size if provided)
        """
        self._model_size = model_size.lower()
        self.confidence_threshold = confidence_threshold
        
        # Load model
        if custom_model_path and os.path.exists(custom_model_path):
            # Load custom model if provided
            self.model = torch.load(custom_model_path)
        else:
            if self._model_size not in self.MODEL_SIZES:
                raise ValueError(f"Invalid model size: {model_size}. "
                              f"Choose from {list(self.MODEL_SIZES.keys())}")
            
            # Load pretrained model from torchvision
            print(f"Loading RetinaNet model: {self._model_size}")
            
            if self._model_size == "small":
                self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
            else:  # resnet101
                self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
                # Note: torchvision doesn't provide resnet101 pretrained, so we use resnet50
            
        # Set to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set class IDs for filtering (COCO dataset)
        self.person_class_id = 0  # Person
        self.vehicle_class_ids = [2, 3, 5, 7]  # Car, motorcycle, bus, truck
        
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in the image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        image_tensor = torchvision.transforms.functional.to_tensor(image_rgb)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Run inference
            predictions = self.model(image_tensor)
            
        # Extract results from first image in batch
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        detections = []
        
        for i in range(len(boxes)):
            # Filter by confidence
            if scores[i] < self.confidence_threshold:
                continue
                
            # Get class ID (RetinaNet returns COCO class IDs)
            class_id = int(labels[i]) - 1  # RetinaNet class IDs are 1-indexed, COCO is 0-indexed
            
            # Filter for person and vehicle classes
            if class_id == self.person_class_id or class_id in self.vehicle_class_ids:
                detections.append(Detection(
                    box=boxes[i],
                    class_id=class_id,
                    confidence=float(scores[i]),
                    is_person=class_id == self.person_class_id,
                    is_vehicle=class_id in self.vehicle_class_ids
                ))
        
        return detections
    
    @property
    def name(self) -> str:
        """Return the name/type of the detector"""
        return "RetinaNet"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size