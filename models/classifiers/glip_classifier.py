"""
glip_classifier.py - GLIP-based fine-grained vehicle classifier (using local GLIP repo)
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
import cv2
import sys

# Add GLIP repo to path
GLIP_PATH = os.path.join(os.path.dirname(__file__), "..", "GLIP")
if GLIP_PATH not in sys.path:
    sys.path.append(GLIP_PATH)

# Import GLIP dependencies
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from core.interfaces import Classifier, ClassificationResult, Detection


class GLIPClassifier(Classifier):
    """GLIP classifier that performs fine-grained classification with text prompts"""
    
    MODEL_CONFIGS = {
        "small": {
            "config": os.path.join(GLIP_PATH, "configs/pretrain/glip_A_Swin_T_O365.yaml"),
            "weights": os.path.join("MODEL", "glip_a_tiny.pth")  # Use tiny for small until you add the model
        },
        "medium": {
            "config": os.path.join(GLIP_PATH, "configs/glip_Swin_L.yaml"),
            "weights": os.path.join("MODEL", "glip_a_medium.pth")
        },
        "large": {
            "config": os.path.join(GLIP_PATH, "configs/glip_Swin_L.yaml"),
            "weights": os.path.join("MODEL", "glip_a_large.pth")
        }
    }
    
    def __init__(self, model_size: str = "small"):
        """
        Initialize GLIP classifier
        
        Args:
            model_size: Size of the model (tiny, small, medium, large)
        """
        self._model_size = model_size.lower()
        
        # Make sure the model size is valid
        if self._model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid model size: {model_size}. "
                          f"Choose from {list(self.MODEL_CONFIGS.keys())}")
        
        # Define supported vehicle classes
        self._supported_classes = [
            "car", 
            "van", 
            "truck", 
            "bus", 
            "emergency vehicle", 
            "no-vehicle",
            "person"
        ]
        
        # Use model config based on size
        model_config = self.MODEL_CONFIGS[self._model_size]
        config_file = model_config["config"]
        weights_file = model_config["weights"]
        
        print(f"Loading GLIP classifier: {weights_file}")
        print(f"Using config: {config_file}")
        
        # Try to load the model
        try:
            # Update configuration
            cfg.merge_from_file(config_file)
            cfg.MODEL.WEIGHT = weights_file
            cfg.freeze()
            
            # Initialize GLIP Demo
            self.glip_demo = GLIPDemo(
                cfg,
                confidence_threshold=0.1,  # Low threshold for classification
                min_image_size=600,
                show_mask_heatmaps=False
            )
            
            # Create class prompt that includes all vehicle types
            self.vehicle_prompt = "car . van . truck . bus . emergency vehicle . police car . ambulance . person"
            self.use_fallback = False
            
        except Exception as e:
            print(f"Error loading GLIP classifier: {e}")
            print("GLIP classifier not available. Using fallback classification method.")
            self.use_fallback = True
    
    def classify(self, image: np.ndarray, detection: Detection) -> ClassificationResult:
        """
        Classify a detected object using GLIP
        
        Args:
            image: Full input image
            detection: Detection object containing bounding box
            
        Returns:
            ClassificationResult with class name and confidence
        """
        # If it's a person detection from the detector, return person directly
        if detection.is_person:
            return ClassificationResult(class_name="person", confidence=0.99)
        
        # Fallback method based on detection class ID
        if self.use_fallback:
            return self._fallback_classify(detection)
        
        # Extract vehicle from image using bounding box
        box = detection.box
        x1, y1, x2, y2 = box
        cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
        
        # Check if the cropped image has valid dimensions
        if cropped_img.size == 0 or cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            return ClassificationResult(class_name="no-vehicle", confidence=0.9)
        
        # Convert to RGB for GLIP
        image_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        
        # Process image using GLIP
        try:
            predictions = self.glip_demo.run_on_image(image_rgb, self.vehicle_prompt)
            
            # If no results, return no-vehicle
            if len(predictions) == 0:
                return ClassificationResult(class_name="no-vehicle", confidence=0.7)
                
            # Extract detection boxes, scores and labels
            scores = predictions.get_field("scores").tolist()
            labels = predictions.get_field("labels").tolist()
            
            # Map label names using the GLIP demo object
            label_names = [self.glip_demo.entities[label - 1] for label in labels]
            
        except Exception as e:
            print(f"Error during GLIP classification: {e}")
            return self._fallback_classify(detection)
            
        # Check for best match
        max_score = 0
        best_class = "no-vehicle"
        
        for score, label_name in zip(scores, label_names):
            # Map to our internal class names
            mapped_class = None
            
            if "car" in label_name and "emergency" not in label_name and "police" not in label_name:
                mapped_class = "car"
            elif "van" in label_name:
                mapped_class = "van"
            elif "truck" in label_name:
                mapped_class = "truck"
            elif "bus" in label_name:
                mapped_class = "bus"
            elif any(term in label_name for term in ["emergency", "police", "ambulance"]):
                mapped_class = "emergency vehicle"
            elif "person" in label_name:
                mapped_class = "person"
            
            if mapped_class and score > max_score:
                max_score = score
                best_class = mapped_class
        
        # If nothing detected well, fall back to no-vehicle
        if max_score < 0.2:
            best_class = "no-vehicle"
            max_score = 0.7
            
        return ClassificationResult(class_name=best_class, confidence=max_score)
    
    def _fallback_classify(self, detection: Detection) -> ClassificationResult:
        """Simple fallback classification based on detection class ID"""
        class_id = detection.class_id
        
        if class_id == 0:  # Person
            return ClassificationResult(class_name="person", confidence=0.9)
        elif class_id == 2:  # Car
            return ClassificationResult(class_name="car", confidence=0.8)
        elif class_id == 3:  # Motorcycle
            return ClassificationResult(class_name="no-vehicle", confidence=0.7)  # Classify motorcycles as no-vehicle
        elif class_id == 5:  # Bus
            return ClassificationResult(class_name="bus", confidence=0.8)
        elif class_id == 7:  # Truck
            return ClassificationResult(class_name="truck", confidence=0.8)
        else:
            return ClassificationResult(class_name="no-vehicle", confidence=0.6)
    
    @property
    def name(self) -> str:
        """Return the name/type of the classifier"""
        return "GLIP"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size
    
    @property
    def supported_classes(self) -> List[str]:
        """Return list of supported vehicle classes"""
        return self._supported_classes