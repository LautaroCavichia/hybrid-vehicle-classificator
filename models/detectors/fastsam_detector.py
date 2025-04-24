"""
fastsam_detector.py - FastSAM (Segment Anything Model) detector implementation
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
import cv2
from ultralytics import FastSAM

from core.interfaces import ObjectDetector, Detection


class FastSAMDetector(ObjectDetector):
    """FastSAM detector for object detection based on segmentation"""
    
    MODEL_SIZES = {
        "small": "FastSAM-s.pt",
        "medium": "FastSAM-x.pt",
    }
    
    def __init__(self, 
                 model_size: str = "small", 
                 confidence_threshold: float = 0.25,
                 custom_model_path: Optional[str] = None,
                 iou_threshold: float = 0.7):
        """
        Initialize FastSAM detector
        
        Args:
            model_size: Size of the model (small or medium)
            confidence_threshold: Detection confidence threshold
            custom_model_path: Path to custom model weights (overrides model_size if provided)
            iou_threshold: IoU threshold for NMS
        """
        self._model_size = model_size.lower()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Set up model path
        if custom_model_path and os.path.exists(custom_model_path):
            self.model_path = custom_model_path
        else:
            if self._model_size not in self.MODEL_SIZES:
                raise ValueError(f"Invalid model size: {model_size}. "
                              f"Choose from {list(self.MODEL_SIZES.keys())}")
            self.model_path = self.MODEL_SIZES[self._model_size]
        
        # Initialize model
        print(f"Loading FastSAM model: {self.model_path}")
        self.model = FastSAM(self.model_path)
        
        # Set class IDs for filtering (COCO dataset)
        self.person_class_id = 0  # Person
        self.vehicle_class_ids = [2, 3, 5, 7]  # Car, motorcycle, bus, truck
        
        # Define clear class names for consistent identification
        self.class_names = {
            0: "person",
            2: "car", 
            3: "motorcycle", 
            5: "bus", 
            7: "truck"
        }
        
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in the image using FastSAM
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        # Convert to RGB for model input (FastSAM expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 1: Run detection with FastSAM for segmentation
        results = self.model(
            image_rgb,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            retina_masks=True,
            verbose=False
        )
        
        detections = []
        
        # Extract detections from results
        if not results or len(results) == 0:
            return []
            
        # Get segmentation masks
        masks = results[0].masks
        
        # If no masks were found, return empty list
        if masks is None or len(masks.data) == 0:
            return []
            
        # Step 2: Use the original FastSAM model for class prediction
        # Run a standard predict to get class information
        cls_results = self.model.predict(
            image_rgb, 
            conf=self.confidence_threshold,
            verbose=False
        )[0]
        
        # If no results, return empty list
        if not hasattr(cls_results, 'boxes') or len(cls_results.boxes) == 0:
            return []
        
        # Get boxes for class information
        orig_boxes = None
        if hasattr(cls_results, 'boxes') and len(cls_results.boxes) > 0:
            orig_boxes = cls_results.boxes
            
        # Step 3: Extract bounding boxes from masks and match with classes
        for i in range(len(masks.data)):
            # Get segmentation mask
            mask = masks.data[i].cpu().numpy()
            
            # Find contours
            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Skip if no contours found
            if not contours:
                continue
                
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box from contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Skip tiny detections that are likely noise
            if w < 10 or h < 10 or w * h < 100:
                continue
            
            # Create bounding box
            box = np.array([x, y, x + w, y + h])
            
            # Step 4: Try to classify this region by comparing with original results
            matched_class_id = None
            matched_confidence = 0.0
            
            if orig_boxes is not None:
                # Convert box to format [x1, y1, x2, y2]
                mask_box = np.array([x, y, x + w, y + h])
                
                # Find matching box with highest IoU
                for j in range(len(orig_boxes)):
                    det_box = orig_boxes.xyxy[j].cpu().numpy()
                    class_id = int(orig_boxes.cls[j].item())
                    confidence = float(orig_boxes.conf[j].item())
                    
                    # Only consider person and vehicle classes
                    if class_id != self.person_class_id and class_id not in self.vehicle_class_ids:
                        continue
                    
                    # Calculate IoU between mask box and detection box
                    iou = self._calculate_iou(mask_box, det_box)
                    
                    # If IoU is high enough, this is likely the same object
                    if iou > 0.3 and confidence > matched_confidence:
                        matched_class_id = class_id
                        matched_confidence = confidence
            
            # If we found a match with good confidence
            if matched_class_id is not None and matched_confidence >= self.confidence_threshold:
                detections.append(Detection(
                    box=box,
                    class_id=matched_class_id,
                    confidence=matched_confidence,
                    is_person=matched_class_id == self.person_class_id,
                    is_vehicle=matched_class_id in self.vehicle_class_ids
                ))
            # If no match but this is a strong mask, assign it as a vehicle by default (adjustable)
            elif mask.mean() > 0.5:  # Strength of the mask
                # Default to car for strong masks without class match
                default_class_id = 2  # Car
                default_confidence = 0.6  # Reasonable confidence
                
                detections.append(Detection(
                    box=box,
                    class_id=default_class_id,
                    confidence=default_confidence,
                    is_person=False,
                    is_vehicle=True
                ))
        
        return detections
        
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Get coordinates of intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Area of intersection
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        intersection = w * h
        
        # Area of union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    @property
    def name(self) -> str:
        """Return the name/type of the detector"""
        return "FastSAM"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size