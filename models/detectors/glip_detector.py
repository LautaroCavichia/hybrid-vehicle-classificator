"""
glip_detector.py - GLIP-based object detector implementation (using local GLIP repo)
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
import cv2
import sys

# Add GLIP repo to path
GLIP_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "GLIP")
if GLIP_PATH not in sys.path:
    sys.path.append(GLIP_PATH)

# Import GLIP dependencies
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from core.interfaces import ObjectDetector, Detection


class GLIPDetector(ObjectDetector):
    """GLIP-based object detector that supports text prompting"""
    
    MODEL_CONFIGS = {
        "small": {
            "config": os.path.join(GLIP_PATH, "configs/pretrain/glip_A_Swin_T_O365.yaml"),
            "weights": os.path.join("MODEL", "glip_a_tiny.pth")  # Use tiny for small until you add the model
        },
        "medium": {
            "config": os.path.join(GLIP_PATH, "configs/pretrain/glip_Swin_L.yaml"),
            "weights": os.path.join("MODEL", "glip_a_medium.pth")
        },
        "large": {
            "config": os.path.join(GLIP_PATH, "configs/pretrain/glip_Swin_L.yaml"),
            "weights": os.path.join("MODEL", "glip_a_large.pth")
        }
    }
    
    def __init__(self, 
                 model_size: str = "small", 
                 confidence_threshold: float = 0.4,
                 custom_model_path: Optional[str] = None):
        """
        Initialize GLIP detector
        
        Args:
            model_size: Size of the model (tiny, small, medium, large)
            confidence_threshold: Detection confidence threshold
            custom_model_path: Path to custom model (overrides model_size if provided)
        """
        self._model_size = model_size.lower()
        self.confidence_threshold = confidence_threshold
        
        # Make sure the model size is valid
        if self._model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid model size: {model_size}. "
                          f"Choose from {list(self.MODEL_CONFIGS.keys())}")

        # Use model config based on size or custom path
        model_config = self.MODEL_CONFIGS[self._model_size]
        config_file = model_config["config"]
        weights_file = custom_model_path if custom_model_path else model_config["weights"]
        
        print(f"Loading GLIP model: {weights_file}")
        print(f"Using config: {config_file}")
        
        # Try to load the GLIP model
        try:
            # Update configuration
            cfg.merge_from_file(config_file)
            cfg.MODEL.WEIGHT = weights_file
            cfg.freeze()
            
            # Initialize GLIP Demo
            self.glip_demo = GLIPDemo(
                cfg,
                confidence_threshold=self.confidence_threshold,
                min_image_size=800,
                show_mask_heatmaps=False
            )
            
            # Define text prompt for vehicles and persons
            self.prompt = "person . car . motorcycle . bus . truck"
            
            self.use_fallback = False
            
        except Exception as e:
            print(f"Error loading GLIP model: {e}")
            print("GLIP model not available. Using fallback to YOLO.")
            # Fallback to YOLO
            from models.detectors.yolo_detector import YOLODetector
            self.fallback_detector = YOLODetector(model_size="small")
            self.use_fallback = True
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in the image using GLIP
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        # Use fallback if GLIP model failed to load
        if self.use_fallback:
            return self.fallback_detector.detect(image)
            
        # Convert to RGB for GLIP
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run GLIP detection
        predictions = self.glip_demo.run_on_image(image_rgb, self.prompt)
        
        # Extract detection boxes, scores and labels
        boxes = predictions.bbox.tolist()
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        
        # Map label names using the GLIP demo object
        label_names = [self.glip_demo.entities[label - 1] for label in labels]
        
        # Convert to our Detection format
        detections = []
        
        for box, score, label_name in zip(boxes, scores, label_names):
            # Skip low confidence detections
            if score < self.confidence_threshold:
                continue
                
            # Map to our internal class IDs
            class_id = -1
            is_person = False
            is_vehicle = False
            
            if "person" in label_name:
                class_id = 0
                is_person = True
            elif "car" in label_name:
                class_id = 2
                is_vehicle = True
            elif "motorcycle" in label_name:
                class_id = 3
                is_vehicle = True
            elif "bus" in label_name:
                class_id = 5
                is_vehicle = True
            elif "truck" in label_name:
                class_id = 7
                is_vehicle = True
            
            if class_id >= 0:
                detections.append(Detection(
                    box=box,
                    class_id=class_id,
                    confidence=score,
                    is_person=is_person,
                    is_vehicle=is_vehicle
                ))
        
        return detections
    
    @property
    def name(self) -> str:
        """Return the name/type of the detector"""
        return "GLIP"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size