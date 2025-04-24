"""
vit_classifier.py - Vision Transformer (ViT) based classifier implementation
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, vit_b_32, ViT_B_16_Weights, ViT_B_32_Weights

from core.interfaces import Classifier, ClassificationResult, Detection


class ViTClassifier(Classifier):
    """Vision Transformer (ViT) based classifier from torchvision"""
    
    MODEL_SIZES = {
        "small": "vit_b_32",
        "medium": "vit_b_16",
    }
    
    def __init__(self, model_size: str = "medium"):
        """
        Initialize ViT classifier
        
        Args:
            model_size: Size of the model (small, medium)
        """
        self._model_size = model_size.lower()
        
        # Set up model type
        if self._model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size: {model_size}. "
                          f"Choose from {list(self.MODEL_SIZES.keys())}")
        
        self.model_type = self.MODEL_SIZES[self._model_size]
        
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
        
        # Initialize the model
        print(f"Loading ViT classifier: {self.model_type}")
        self._init_model()
    
    def _init_model(self):
        """Initialize the ViT model and preprocessing transforms"""
        # Use GPU if available, MPS (Metal Performance Shaders) on Mac if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # Load the model with pretrained weights
        if self.model_type == "vit_b_16":
            weights = ViT_B_16_Weights.DEFAULT
            self.model = vit_b_16(weights=weights)
        else:  # vit_b_32
            weights = ViT_B_32_Weights.DEFAULT
            self.model = vit_b_32(weights=weights)
        
        # Set model to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)
        
        # Get the preprocessing transforms
        self.preprocess = weights.transforms()
        
        # Load class mapping for ImageNet
        self.imagenet_classes = weights.meta["categories"]
        
        # Create a mapping from ImageNet classes to our vehicle classes
        self.class_mapping = self._create_class_mapping()
        
        # This is the last fully connected layer
        self.fc_layer = self.model.heads.head
        
        # Create custom classifier for vehicle types using the feature extractor
        self._initialize_custom_classifier()
    
    def _create_class_mapping(self):
        """
        Create a mapping from ImageNet classes to our vehicle classes
        """
        # Define keywords for each vehicle class
        class_keywords = {
            "car": ["car", "convertible", "sports car", "sedan", "hatchback"],
            "van": ["van", "minivan", "minibus"],
            "truck": ["truck", "pickup", "tractor", "semi"],
            "bus": ["bus", "trolleybus", "school bus"],
            "emergency vehicle": ["ambulance", "police", "fire engine", "fire truck"],
            "person": ["person", "pedestrian", "human"],
            "no-vehicle": []  # Default if nothing matches
        }
        
        # Create the mapping
        class_map = {}
        for i, class_name in enumerate(self.imagenet_classes):
            mapped_class = "no-vehicle"  # Default
            
            # Check each vehicle class for matching keywords
            for vehicle_class, keywords in class_keywords.items():
                if any(keyword.lower() in class_name.lower() for keyword in keywords):
                    mapped_class = vehicle_class
                    break
            
            class_map[i] = mapped_class
        
        return class_map
    
    def _initialize_custom_classifier(self):
        """Initialize a custom classifier head for vehicle classification"""
        # Define a simple classifier on top of ViT features
        self.vehicle_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, len(self._supported_classes))
        ).to(self.device)
        
        # Since we're not training, set to eval mode
        self.vehicle_classifier.eval()
    
    def _extract_features(self, image_tensor):
        """Extract features from the ViT model before the classification head"""
        # Forward pass through the model excluding the classification head
        with torch.no_grad():
            # Get features before the classification head
            x = self.model.conv_proj(image_tensor)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            
            # Add class token and position embeddings
            cls_token = self.model.class_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.model.encoder.pos_embedding
            
            # Pass through transformer encoder
            x = self.model.encoder(x)
            
            # Get CLS token features
            features = x[:, 0]
            
            return features
    
    def classify(self, image: np.ndarray, detection: Detection) -> ClassificationResult:
        """
        Classify a detected object using ViT
        
        Args:
            image: Full input image
            detection: Detection object containing bounding box
            
        Returns:
            ClassificationResult with class name and confidence
        """
        # If it's a person detection, return person class directly
        if detection.is_person:
            return ClassificationResult(class_name="person", confidence=0.99)
        
        # Extract vehicle from image using bounding box
        box = detection.box
        x1, y1, x2, y2 = box
        cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
        
        # Check if the cropped image has valid dimensions
        if cropped_img.size == 0 or cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            return ClassificationResult(class_name="no-vehicle", confidence=0.9)
        
        # Convert to PIL image for preprocessing
        pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        
        # Apply preprocessing transforms
        try:
            input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            
            # Get ImageNet predictions
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                
                # Get top class and probability
                top_prob, top_class = torch.max(probabilities, dim=1)
                
                # Map to our vehicle class
                imagenet_class_idx = top_class.item()
                mapped_class = self.class_mapping[imagenet_class_idx]
                confidence = top_prob.item()
                
                # Use detection class ID for additional context
                class_id = detection.class_id
                
                # Refine classification based on detection class ID
                if class_id == 2:  # Car in COCO
                    if mapped_class == "no-vehicle":
                        mapped_class = "car"
                        confidence = max(0.6, confidence)
                elif class_id == 5:  # Bus in COCO
                    if mapped_class == "no-vehicle":
                        mapped_class = "bus"
                        confidence = max(0.6, confidence)
                elif class_id == 7:  # Truck in COCO
                    if mapped_class == "no-vehicle":
                        mapped_class = "truck"
                        confidence = max(0.6, confidence)
            
            return ClassificationResult(class_name=mapped_class, confidence=confidence)
            
        except Exception as e:
            print(f"Error in ViT classification: {e}")
            
            # Fallback: Use detection class ID
            if detection.class_id == 2:
                return ClassificationResult(class_name="car", confidence=0.7)
            elif detection.class_id == 5:
                return ClassificationResult(class_name="bus", confidence=0.7)
            elif detection.class_id == 7:
                return ClassificationResult(class_name="truck", confidence=0.7)
            else:
                return ClassificationResult(class_name="no-vehicle", confidence=0.6)
    
    @property
    def name(self) -> str:
        """Return the name/type of the classifier"""
        return "ViT"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size
    
    @property
    def supported_classes(self) -> List[str]:
        """Return list of supported vehicle classes"""
        return self._supported_classes