"""
rtdetr_detector.py - RT-DETR (Real-Time Detection Transformer) implementation
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection

from core.interfaces import ObjectDetector, Detection


class RTDETRDetector(ObjectDetector):
    """Real-Time Detection Transformer (RT-DETR) object detector
    
    RT-DETR is a state-of-the-art transformer-based detector that 
    achieves excellent speed-accuracy trade-off by using efficient 
    hybrid encoder architecture with IoU-aware query selection.
    """
    
    # Correct model IDs from the PekingU organization
    MODEL_SIZES = {
        "small": "PekingU/rtdetr_r18vd",  # r18vd backbone
        "medium": "PekingU/rtdetr_r50vd",  # r50vd backbone
        "large": "PekingU/rtdetr_r101vd"   # r101vd backbone
    }
    
    # COCO dataset class IDs for vehicles and persons
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    PERSON_CLASS_ID = 0  # person
    
    def __init__(self, 
                 model_size: str = "medium", 
                 confidence_threshold: float = 0.25,
                 custom_model_path: Optional[str] = None):
        """
        Initialize RT-DETR detector
        
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
        
        # Use GPU if available, MPS on macOS if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Use MPS for macOS with Metal GPU
        else:
            self.device = torch.device("cpu")
            
        print(f"Using device: {self.device} for RT-DETR detector")
        
        try:
            # Load RT-DETR specific model and processor
            print(f"Loading model from: {self.model_path}")
            self.image_processor = RTDetrImageProcessor.from_pretrained(self.model_path)
            self.model = RTDetrForObjectDetection.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load COCO class mapping
            self.id2label = self.model.config.id2label
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load RT-DETR model: {e}. Please check if the model ID is correct and internet connection is available.")
    
    def _preprocess_image(self, image: np.ndarray):
        """
        Preprocess image for RT-DETR model
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Processed inputs and original image size
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL image
        pil_img = Image.fromarray(image_rgb)
        
        # Get original size for post-processing
        original_size = torch.tensor([(pil_img.height, pil_img.width)])
        
        # Process for RT-DETR
        inputs = self.image_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs, original_size
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in the image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        # Preprocess image
        inputs, original_size = self._preprocess_image(image)
        original_size = original_size.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process using the processor's method
        results = self.image_processor.post_process_object_detection(
            outputs, 
            target_sizes=original_size,
            threshold=self.confidence_threshold
        )
        
        # Convert to our Detection format
        detections = []
        
        # Process the first (and only) image in the batch
        result = results[0]
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            score_val = score.item()
            label_val = label.item()
            box_val = box.cpu().numpy()
            
            # Get class name and map to COCO ID
            class_name = self.id2label[label_val]
            coco_class_id = self._map_to_coco_id(class_name)
            
            # Skip if not person or vehicle
            if coco_class_id != self.PERSON_CLASS_ID and coco_class_id not in self.VEHICLE_CLASS_IDS:
                continue
                
            # Create detection
            detection = Detection(
                box=box_val.astype(np.float32),  # Already in [x1, y1, x2, y2] format
                class_id=coco_class_id,
                confidence=float(score_val),
                is_person=coco_class_id == self.PERSON_CLASS_ID,
                is_vehicle=coco_class_id in self.VEHICLE_CLASS_IDS
            )
            detections.append(detection)
        
        return detections
    
    def _map_to_coco_id(self, class_name):
        """Map RT-DETR class name to COCO class ID"""
        # Lower case and strip for consistent matching
        class_name = class_name.lower().strip()
        
        # COCO class ID mapping
        coco_mapping = {
            "person": 0,
            "bicycle": 1,
            "car": 2,
            "motorcycle": 3,
            "airplane": 4,
            "bus": 5,
            "train": 6,
            "truck": 7,
            "boat": 8,
        }
        
        # Handle synonym mapping
        synonyms = {
            "automobile": "car",
            "motorbike": "motorcycle",
            "aeroplane": "airplane",
            "motor": "motorcycle",
            "semi": "truck",
            "lorry": "truck"
        }
        
        if class_name in synonyms:
            class_name = synonyms[class_name]
            
        return coco_mapping.get(class_name, -1)  # Return -1 for non-mapped classes
    
    @property
    def name(self) -> str:
        """Return the name/type of the detector"""
        return "RT-DETR"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size