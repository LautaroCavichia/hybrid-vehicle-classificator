"""
clip_classifier.py - CLIP-based fine-grained vehicle classifier
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
import cv2
import clip
from PIL import Image

from core.interfaces import Classifier, ClassificationResult, Detection


class CLIPClassifier(Classifier):
    """CLIP-based fine-grained vehicle classifier"""
    
    MODEL_SIZES = {
        "small": "ViT-B/16",
        "medium": "ViT-B/32",
        "large": "ViT-L/14"
    }
    
    def __init__(self, model_size: str = "medium"):
        """
        Initialize CLIP classifier
        
        Args:
            model_size: Size of the model (small, medium, large)
        """
        self._model_size = model_size.lower()
        
        # Set up model path
        if self._model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size: {model_size}. "
                          f"Choose from {list(self.MODEL_SIZES.keys())}")
        
        self.clip_model = self.MODEL_SIZES[self._model_size]
        
        # Define vehicle classes with expanded set
        self._supported_classes = [
            "car", 
            "van", 
            "truck", 
            "bus", 
            "emergency vehicle", 
            "non-vehicle",
        ]
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        self.model, self.preprocess = clip.load(self.clip_model, device=self.device)
        
        # Pre-encode text prompts for efficiency
        self._precompute_text_features()
    
    def _precompute_text_features(self):
        """Pre-encode enhanced text prompts for each vehicle class with richer context"""
        # Define multiple prompts per class to help CLIP differentiate better
        prompt_templates = {
            "car": [
                "a regular passenger car",
                "a small personal vehicle",
                "a sedan or compact car",
                "a station wagon or hatchback",
            ],
            "van": [
                "a minivan or small delivery van",
                "a long vehicle with sliding doors",
                "a small delivery van with side doors",
                "a compact commercial van",
                "a small utility vehicle used for deliveries"
            ],
            "truck": [
                "a pickup truck with an open bed",
                "a cargo truck with enclosed storage area",
                "a large delivery truck with a box-shaped cargo area",
                "a commercial truck for transporting goods",
                "a utility truck used for work and transportation"
            ],
            "bus": [
                "a passenger bus with many windows",
                "a large vehicle designed to carry many passengers",
                "a public transport bus with multiple doors",
                "a school bus with rows of seats"
            ],
            "emergency vehicle": [
                "an ambulance with emergency lights",
                "a police car with sirens",
                "a fire truck with emergency equipment",
                "a first responder vehicle with flashing lights",
                "an emergency response vehicle"
            ],
            "non-vehicle": [
                "a scene without any vehicles",
                "an empty street or road",
                "a background without any cars or vehicles",
                "a non-vehicle object",
                "a person walking or standing",
                "a pedestrian on the sidewalk",
                "a human figure",
                "a person crossing the street"
            ]
        }

        all_prompts = []
        class_to_indices = []
        for cls in self._supported_classes:
            prompts = prompt_templates[cls]
            start_idx = len(all_prompts)
            all_prompts.extend(prompts)
            end_idx = len(all_prompts)
            class_to_indices.append((start_idx, end_idx))

        # Tokenize all prompts
        text_inputs = torch.cat([clip.tokenize(p) for p in all_prompts]).to(self.device)

        with torch.no_grad():
            text_features_all = self.model.encode_text(text_inputs)
            text_features_all = text_features_all / text_features_all.norm(dim=-1, keepdim=True)

            # Average embeddings for each class
            text_features = []
            for start, end in class_to_indices:
                class_features = text_features_all[start:end]
                mean_feature = class_features.mean(dim=0)
                mean_feature = mean_feature / mean_feature.norm()  # Normalize again
                text_features.append(mean_feature)

            self.text_features = torch.stack(text_features)
    
    def classify(self, image: np.ndarray, detection: Detection) -> ClassificationResult:
        """
        Classify a detected object using CLIP
        
        Args:
            image: Full input image
            detection: Detection object containing bounding box
            
        Returns:
            ClassificationResult with class name and confidence
        """
        # If it's a person detection, return person class directly
        if detection.is_person:
            return ClassificationResult(class_name="non-vehicle", confidence=0.99)
        
        # Extract vehicle from image using bounding box
        box = detection.box
        x1, y1, x2, y2 = box
        cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
        
        # Check if the cropped image has valid dimensions
        if cropped_img.size == 0 or cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            return ClassificationResult(class_name="non-vehicle", confidence=0.9)
        
        # Convert to PIL image for CLIP
        pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        
        # Process image for CLIP
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        # Get features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        # Calculate similarity scores
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        scores = similarity[0].cpu().detach().numpy()
        
        # Get best class and confidence
        best_idx = np.argmax(scores)
        confidence = scores[best_idx]
        best_class = self._supported_classes[best_idx]
        
        return ClassificationResult(class_name=best_class, confidence=float(confidence))
    
    @property
    def name(self) -> str:
        """Return the name/type of the classifier"""
        return "CLIP"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size
    
    @property
    def supported_classes(self) -> List[str]:
        """Return list of supported vehicle classes"""
        return self._supported_classes