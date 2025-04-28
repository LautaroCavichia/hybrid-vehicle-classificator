"""
owlvit_classifier.py - OwlViT-based classifier for fine-grained vehicle classification
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
from PIL import Image
import cv2
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from functools import lru_cache

from core.interfaces import Classifier, ClassificationResult, Detection


class OwlViTClassifier(Classifier):
    """
    OwlViT-based classifier that uses natural language queries for zero-shot classification.
    This model is especially efficient on macOS as it works well with MPS.
    """
    
    MODEL_SIZES = {
        "small": "owlvit-base-patch32",
        "medium": "owlvit-base-patch16",
        "large": "owlvit-large-patch14",
    }
    
    def __init__(self, model_size: str = "medium", cache_size: int = 64):
        """
        Initialize OwlViT classifier
        
        Args:
            model_size: Size of the model (small, medium, large)
            cache_size: Size of the LRU cache for feature caching
        """
        self._model_size = model_size.lower()
        self._cache_size = cache_size
        
        # Set up model type
        if self._model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size: {model_size}. "
                          f"Choose from {list(self.MODEL_SIZES.keys())}")
        
        self.model_path = f"google/{self.MODEL_SIZES[self._model_size]}"
        
        # Define supported vehicle classes with detailed descriptions
        self._supported_classes = [
            "car", 
            "van", 
            "truck", 
            "bus", 
            "emergency vehicle", 
            "non-vehicle",
        ]
        
        # Initialize the model
        print(f"Loading OwlViT classifier: {self.model_path}")
        self._init_model()
        
        # Set up text query templates
        self._init_text_queries()
        
        # Set up image feature cache with LRU caching
        self._setup_cache()
    
    def _init_model(self):
        """Initialize the OwlViT model and processor"""
        # Optimized device selection for macOS
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        try:
            # Load processor
            self.processor = OwlViTProcessor.from_pretrained(self.model_path)
            
            # Load model to CPU first to avoid meta tensor issues
            self.model = OwlViTForObjectDetection.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float32,
                device_map="auto" if self.device == torch.device("cuda") else None
            )
            
            # Move model to device if not using device_map="auto"
            if self.device != torch.device("cuda"):
                self.model = self.model.to(self.device)
        except Exception as e:
            print(f"Error during model initialization: {e}")
            # Fallback loading method
            print("Using fallback method for loading model")
            self.processor = OwlViTProcessor.from_pretrained(self.model_path)
            self.model = OwlViTForObjectDetection.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
    
    def _init_text_queries(self):
        """Initialize text queries for each class with detailed descriptions"""
        self.text_queries = {
            "car": [
                "a passenger car", 
                "a sedan car",
                "a sports car", 
                "a hatchback car",
                "a family car",
                "a small car",
                "a coupe car",
                "an SUV",
                "a personal vehicle with four wheels",
            ],
            "van": [
                "a minivan",
                "a passenger van",
                "a delivery van",
                "a panel van",
                "a cargo van",
                "a van with sliding doors",
                "a family minivan",
            ],
            "truck": [
                "a pickup truck",
                "a delivery truck",
                "a cargo truck",
                "a flatbed truck",
                "a semi truck",
                "a commercial truck",
                "a utility truck",
            ],
            "bus": [
                "a passenger bus",
                "a city bus",
                "a school bus",
                "a tour bus",
                "a public transit bus",
                "a double-decker bus",
            ],
            "emergency vehicle": [
                "an ambulance",
                "a police car",
                "a fire truck",
                "an emergency response vehicle",
                "a police van",
                "a paramedic vehicle",
            ],
            "non-vehicle": [
                "a pedestrian",
                "a person standing",
                "a person walking",
                "a bicycle",
                "a motorcycle",
                "a street sign",
                "a building",
                "background with no vehicle",
            ],
        }
        
        # Create flattened list for batch processing
        self.all_text_queries = []
        for class_name, queries in self.text_queries.items():
            for query in queries:
                self.all_text_queries.append((class_name, query))
    
    def _setup_cache(self):
        """Set up LRU cache for text and image features"""
        @lru_cache(maxsize=self._cache_size)
        def cached_process_image(image_hash):
            """Cache image processing results based on image hash"""
            # This function body will be populated during inference
            pass
        
        self._cached_process_image = cached_process_image
        
        # Pre-encode all text queries for efficiency
        self._precompute_text_features()
    
    def _precompute_text_features(self):
        """Pre-compute text features for all queries to improve inference speed"""
        # Extract just the text queries without the class names
        text_queries = [query for _, query in self.all_text_queries]
        
        try:
            with torch.no_grad():
                # Process text in smaller batches to avoid OOM issues
                batch_size = 8
                all_text_features = []
                
                for i in range(0, len(text_queries), batch_size):
                    batch_queries = text_queries[i:i+batch_size]
                    
                    # Process batch - FIXED: Use text_processor correctly
                    text_inputs = self.processor(
                        text=batch_queries,
                        return_tensors="pt",
                        padding=True,
                    )
                    
                    # Move to device
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    
                    # Get text features - FIXED: Get text embeddings from output
                    outputs = self.model.text_model(**{k: v for k, v in text_inputs.items() 
                                                   if k in ['input_ids', 'attention_mask']})
                    text_features = outputs.pooler_output
                    all_text_features.append(text_features)
                
                # Concatenate all batches
                self.text_features = torch.cat(all_text_features, dim=0)
                
                # Create mapping from class name to indices in text features
                self.class_to_indices = {}
                for i, (class_name, _) in enumerate(self.all_text_queries):
                    if class_name not in self.class_to_indices:
                        self.class_to_indices[class_name] = []
                    self.class_to_indices[class_name].append(i)
                
                print(f"Successfully precomputed text features for {len(text_queries)} queries")
        except Exception as e:
            print(f"Error precomputing text features: {e}")
            # Set a flag to compute text features on-the-fly if precomputation fails
            self.text_features = None
            print("Will compute text features during inference")
    
    def _get_image_hash(self, cropped_img):
        """Create a simple hash of the image for caching purposes"""
        # Resize to tiny resolution for fast hashing
        small_img = cv2.resize(cropped_img, (32, 32))
        # Simple hash based on average pixel values
        return hash(small_img.mean(axis=(0, 1)).tobytes())
    
    def _compute_text_features_on_fly(self, class_name):
        """Compute text features for a specific class on the fly"""
        queries = self.text_queries[class_name]
        
        with torch.no_grad():
            text_inputs = self.processor(
                text=queries,
                return_tensors="pt",
                padding=True,
            )
            
            # Move to device
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            # Get text features - FIXED: Get text embeddings from text_model
            outputs = self.model.text_model(**{k: v for k, v in text_inputs.items() 
                                           if k in ['input_ids', 'attention_mask']})
            text_features = outputs.pooler_output
            
            return text_features
    
    def classify(self, image: np.ndarray, detection: Detection) -> ClassificationResult:
        """
        Classify a detected object using OwlViT
        
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
        
        try:
            # Convert to PIL image for the model
            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            
            # Process image
            with torch.no_grad():
                # Process the image
                image_inputs = self.processor(images=pil_img, return_tensors="pt")
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                
                # Get image features - FIXED: Extract image features from vision model
                outputs = self.model.vision_model(**{k: v for k, v in image_inputs.items() 
                                                 if k in ['pixel_values', 'pixel_mask']})
                image_features = outputs.pooler_output
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                if self.text_features is not None:
                    # Use precomputed text features
                    # Normalize text features
                    text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    
                    # Calculate average score per class
                    class_scores = {}
                    for class_name, indices in self.class_to_indices.items():
                        class_scores[class_name] = similarities[0, indices].mean().item()
                else:
                    # Compute text features on the fly for each class
                    class_scores = {}
                    for class_name in self._supported_classes:
                        text_features = self._compute_text_features_on_fly(class_name)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        class_scores[class_name] = similarity.mean().item()
                
                # Get best class and confidence
                best_class = max(class_scores, key=class_scores.get)
                confidence = class_scores[best_class]
                
                # Use COCO class ID as additional context for close predictions
                # Get second best class and its confidence
                sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_classes) > 1:
                    second_best_class, second_confidence = sorted_classes[1]
                    
                    # If confidence difference is small, use detection class as tiebreaker
                    if confidence - second_confidence < 0.1:
                        class_id = detection.class_id
                        if class_id == 2 and "car" in class_scores:  # Car in COCO
                            best_class = "car"
                            confidence = max(0.65, class_scores["car"])
                        elif class_id == 5 and "bus" in class_scores:  # Bus in COCO
                            best_class = "bus"
                            confidence = max(0.65, class_scores["bus"])
                        elif class_id == 7 and "truck" in class_scores:  # Truck in COCO
                            best_class = "truck"
                            confidence = max(0.65, class_scores["truck"])
        
        except Exception as e:
            print(f"Error in OwlViT classification: {e}")
            # Fallback based on detector class
            if detection.class_id == 2:
                return ClassificationResult(class_name="car", confidence=0.7)
            elif detection.class_id == 5:
                return ClassificationResult(class_name="bus", confidence=0.7)
            elif detection.class_id == 7:
                return ClassificationResult(class_name="truck", confidence=0.7)
            else:
                return ClassificationResult(class_name="non-vehicle", confidence=0.6)
        
        return ClassificationResult(class_name=best_class, confidence=float(confidence))
    
    @property
    def name(self) -> str:
        """Return the name/type of the classifier"""
        return "OwlViT"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size
    
    @property
    def supported_classes(self) -> List[str]:
        """Return list of supported vehicle classes"""
        return self._supported_classes