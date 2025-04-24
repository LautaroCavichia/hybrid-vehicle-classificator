"""
dino_classifier.py - DINO-based fine-grained vehicle classifier
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from core.interfaces import Classifier, ClassificationResult, Detection


class DINOClassifier(Classifier):
    """DINOv2-based fine-grained vehicle classifier using zero-shot approach"""
    
    MODEL_SIZES = {
        "small": "facebook/dinov2-small",
        "medium": "facebook/dinov2-base",
        "large": "facebook/dinov2-large"
    }
    
    def __init__(self, model_size: str = "medium"):
        """
        Initialize DINO classifier
        
        Args:
            model_size: Size of the model (small, medium, large)
        """
        self._model_size = model_size.lower()
        
        # Set up model path
        if self._model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size: {model_size}. "
                          f"Choose from {list(self.MODEL_SIZES.keys())}")
        
        self.model_path = self.MODEL_SIZES[self._model_size]
        
        # Define vehicle classes
        self._supported_classes = [
            "car", 
            "van", 
            "truck", 
            "bus", 
            "emergency vehicle", 
            "no-vehicle",
            "person"
        ]
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        
        # Create text embeddings for zero-shot classification
        self.text_embeddings = self._create_text_embeddings()
    
    def _create_text_embeddings(self):
        """
        Create text embeddings for zero-shot classification using class names
        and descriptive prompts.
        
        Since DINO doesn't natively support text inputs, we use pre-computed
        embeddings that represent the visual characteristics of each class.
        """
        # These are pre-computed embeddings (simplified implementation)
        # In a full implementation, you would use a multimodal model like CLIP
        # to create these embeddings from textual descriptions
        
        # Create random embeddings of the correct dimensionality for demonstration
        # In production, replace with actual representative embeddings
        embedding_dim = 768  # Standard dim for dinov2-base
        if "small" in self.model_path:
            embedding_dim = 384
        elif "large" in self.model_path:
            embedding_dim = 1024
            
        # In a real implementation, we would use actual embeddings
        # This is a placeholder for demonstration
        np.random.seed(42)  # For reproducibility
        embeddings = {}
        
        # Generate representative embeddings for each class
        for class_name in self._supported_classes:
            embedding = np.random.randn(embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings[class_name] = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            
        return embeddings
    
    def classify(self, image: np.ndarray, detection: Detection) -> ClassificationResult:
        """
        Classify a detected object using DINO
        
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
        
        # Convert to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        
        # Process image for DINO
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        # Get features
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token features for classification
            image_features = outputs.last_hidden_state[:, 0, :]
            image_features = F.normalize(image_features, dim=1)
            
        # Compare with text embeddings
        similarities = {}
        for class_name, text_embedding in self.text_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(
                image_features, text_embedding.unsqueeze(0)
            ).item()
            similarities[class_name] = similarity
        
        # Get best class and confidence
        best_class = max(similarities, key=similarities.get)
        confidence = similarities[best_class]
        
        # Scale confidence to 0-1 range
        confidence = (confidence + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        return ClassificationResult(class_name=best_class, confidence=confidence)
    
    @property
    def name(self) -> str:
        """Return the name/type of the classifier"""
        return "DINO"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size
    
    @property
    def supported_classes(self) -> List[str]:
        """Return list of supported vehicle classes"""
        return self._supported_classes