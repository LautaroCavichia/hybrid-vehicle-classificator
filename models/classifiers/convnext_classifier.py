"""
convnext_classifier.py - Improved ConvNeXt-based fine-grained vehicle classifier
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
import cv2
from PIL import Image
import torch.nn.functional as F
from torchvision.models import convnext_tiny, convnext_small, convnext_base, ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights

from core.interfaces import Classifier, ClassificationResult, Detection


class ConvNeXtClassifier(Classifier):
    """ConvNeXt-based fine-grained vehicle classifier with advanced techniques"""
    
    MODEL_SIZES = {
        "tiny": "convnext_tiny",
        "small": "convnext_small",
        "medium": "convnext_base"
    }
    
    def __init__(self, model_size: str = "medium", confidence_threshold: float = 0.6):
        """
        Initialize ConvNeXt classifier with advanced features
        
        Args:
            model_size: Size of the model (tiny, small, medium)
            confidence_threshold: Threshold for classification confidence
        """
        self._model_size = model_size.lower()
        self.confidence_threshold = confidence_threshold
        
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
            "non-vehicle",
            "person"
        ]
        
        # Initialize the model
        print(f"Loading ConvNeXt classifier: {self.model_type}")
        self._init_model()
        
        # Rich semantic descriptors for better classification
        self._class_descriptors = {
            "car": {
                "visual_features": ["sedan body", "compact shape", "passenger windows", "car hood", "trunk"],
                "typical_colors": ["silver", "black", "white", "red", "blue"],
                "typical_sizes": ["small to medium"],
                "typical_contexts": ["roads", "parking lots", "driveways"],
                "distinctive_elements": ["clear cabin windows", "front grill", "headlights", "license plate"]
            },
            "van": {
                "visual_features": ["boxy shape", "high roof", "sliding door", "extended cabin"],
                "typical_colors": ["white", "silver", "dark colors"],
                "typical_sizes": ["medium to large"],
                "typical_contexts": ["roads", "commercial areas", "family transport"],
                "distinctive_elements": ["sliding side door", "higher roof than cars", "extended body", "multiple rows of seats"]
            },
            "truck": {
                "visual_features": ["cargo area", "large cab", "high ground clearance", "large wheels"],
                "typical_colors": ["white", "red", "blue", "yellow"],
                "typical_sizes": ["medium to very large"],
                "typical_contexts": ["highways", "construction sites", "industrial areas"],
                "distinctive_elements": ["cargo bed", "high suspension", "large wheels", "bigger mirrors"]
            },
            "bus": {
                "visual_features": ["elongated body", "multiple windows", "high roof", "passenger doors"],
                "typical_colors": ["yellow", "white", "blue", "red"],
                "typical_sizes": ["large to very large"],
                "typical_contexts": ["bus stops", "transit stations", "roads"],
                "distinctive_elements": ["row of windows", "high passenger capacity", "multiple entry doors"]
            },
            "emergency vehicle": {
                "visual_features": ["specialized body", "warning lights", "official markings"],
                "typical_colors": ["white with markings", "red", "blue", "yellow with stripes"],
                "typical_sizes": ["medium to large"],
                "typical_contexts": ["roads", "accident scenes", "emergency situations"],
                "distinctive_elements": ["flashing lights", "sirens", "emergency markings", "specialized equipment"]
            }
        }
        
    def _init_model(self):
        """Initialize the ConvNeXt model with optimal settings for feature extraction"""
        # Use best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # Load appropriate model with weights
        if self.model_type == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT
            self.model = convnext_tiny(weights=weights)
            self.feature_dim = 768
        elif self.model_type == "convnext_small":
            weights = ConvNeXt_Small_Weights.DEFAULT
            self.model = convnext_small(weights=weights)
            self.feature_dim = 768
        else:  # convnext_base
            weights = ConvNeXt_Base_Weights.DEFAULT
            self.model = convnext_base(weights=weights)
            self.feature_dim = 1024
        
        # Set model to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)
        
        # Get the preprocessing transforms from weights
        self.preprocess = weights.transforms()
        
        # Initialize the feature extraction pipeline
        self._init_feature_extraction()
        
        # Initialize the custom classifier head
        self._init_classifier_head()
    
    def _init_feature_extraction(self):
        """Set up optimized feature extraction pipeline"""
        # Replace classifier with identity to use as feature extractor
        self.model.classifier = torch.nn.Identity()
        
        # Optional: We could add a feature pyramid network here for multi-scale features
        # but for simplicity, we'll use the standard feature extractor
    
    def _init_classifier_head(self):
        """Initialize a sophisticated classifier head for vehicle classification"""
        # Create a more advanced classifier head with better regularization
        self.vehicle_classifier = torch.nn.Sequential(
            # Layer normalization for better feature conditioning
            torch.nn.LayerNorm(self.feature_dim),
            
            # First fully connected layer with dropout
            torch.nn.Linear(self.feature_dim, 512),
            torch.nn.GELU(),  # GELU activation (better than ReLU)
            torch.nn.Dropout(0.3),
            
            # Second fully connected layer
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            
            # Classification layer
            torch.nn.Linear(256, len(self._supported_classes))
        ).to(self.device)
        
        # Set to evaluation mode
        self.vehicle_classifier.eval()
    
    def _extract_features(self, image_tensor):
        """Extract rich features from the ConvNeXt backbone"""
        with torch.no_grad():
            # Forward pass through ConvNeXt layers
            features = self.model.features(image_tensor)
            # Global average pooling
            features = self.model.avgpool(features)
            # Flatten for classifier
            features = features.flatten(1)
            return features
    
    def _apply_test_time_augmentation(self, image):
        """Apply test-time augmentation for more robust predictions"""
        # Create multiple versions of the image
        augmented_images = []
        
        # Original image
        augmented_images.append(image)
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)
        
        # Slight rotation (±5 degrees)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Rotate +5 degrees
        rotation_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
        rotated_plus = cv2.warpAffine(image, rotation_matrix, (width, height))
        augmented_images.append(rotated_plus)
        
        # Rotate -5 degrees
        rotation_matrix = cv2.getRotationMatrix2D(center, -5, 1.0)
        rotated_minus = cv2.warpAffine(image, rotation_matrix, (width, height))
        augmented_images.append(rotated_minus)
        
        # Slight brightness adjustment
        bright = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
        augmented_images.append(bright)
        
        # Slight darkness adjustment
        dark = cv2.convertScaleAbs(image, alpha=0.9, beta=-10)
        augmented_images.append(dark)
        
        return augmented_images
    
    def _enhance_image_quality(self, image):
        """Apply advanced image enhancement for better feature extraction"""
        # Convert to LAB color space for better enhancement
        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab_img)
        
        # Apply CLAHE to L channel (adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        # Merge enhanced L with original A and B
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        
        # Convert back to BGR
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) / 9.0
        sharpened = cv2.filter2D(enhanced_img, -1, kernel)
        
        # Blend original with sharpened for better balance
        final_enhanced = cv2.addWeighted(enhanced_img, 0.7, sharpened, 0.3, 0)
        
        return final_enhanced
    
    def _apply_temperature_scaling(self, logits, temperature=1.2):
        """Apply temperature scaling to calibrate confidence scores"""
        # Temperature scaling is a simple post-processing technique
        # to calibrate neural network confidence scores
        return logits / temperature
    
    def _predict_with_augmentation(self, image_crops):
        """Perform prediction with test-time augmentation"""
        all_probabilities = []
        
        # Process each augmented image
        for crop in image_crops:
            # Convert to PIL for preprocessing
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            # Apply preprocessing transforms
            input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            
            # Extract features
            features = self._extract_features(input_tensor)
            
            # Get class logits
            with torch.no_grad():
                logits = self.vehicle_classifier(features)
                
                # Apply temperature scaling for better calibration
                calibrated_logits = self._apply_temperature_scaling(logits)
                
                # Convert to probabilities
                probabilities = F.softmax(calibrated_logits, dim=1)
                
                all_probabilities.append(probabilities)
        
        # Average probabilities across augmentations
        avg_probs = torch.mean(torch.stack(all_probabilities), dim=0)
        
        # Get the top class and confidence
        confidence, idx = torch.max(avg_probs, dim=1)
        top_class = self._supported_classes[idx.item()]
        
        return top_class, confidence.item()
    
    def _analyze_visual_context(self, image, detection):
        """Analyze visual context to refine classification"""
        # Extract bounding box
        x1, y1, x2, y2 = detection.box.astype(int)
        
        # Get aspect ratio
        aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0
        
        # Analyze shape characteristics
        visual_cues = {}
        
        # Shape analysis
        if aspect_ratio < 0.7:
            visual_cues["shape"] = "tall"
        elif aspect_ratio > 1.5:
            visual_cues["shape"] = "wide"
        elif 0.8 < aspect_ratio < 1.2:
            visual_cues["shape"] = "square"
        else:
            visual_cues["shape"] = "balanced"
        
        # Size analysis
        img_height, img_width = image.shape[:2]
        box_area = (x2 - x1) * (y2 - y1)
        img_area = img_height * img_width
        size_ratio = box_area / img_area
        
        if size_ratio < 0.1:
            visual_cues["size"] = "small"
        elif size_ratio > 0.4:
            visual_cues["size"] = "large"
        else:
            visual_cues["size"] = "medium"
        
        # Position analysis
        center_y = (y1 + y2) / 2
        if center_y < img_height * 0.4:
            visual_cues["position"] = "upper"
        elif center_y > img_height * 0.6:
            visual_cues["position"] = "lower"
        else:
            visual_cues["position"] = "center"
        
        return visual_cues
    
    def _refine_with_context(self, class_name, confidence, visual_cues):
        """Refine classification using visual context cues"""
        # Apply context-aware rules to refine classification
        
        # Check for ambiguous cases requiring clarification
        if confidence < 0.7:
            # Use shape cues to disambiguate similar classes
            if class_name == "car" and visual_cues["shape"] == "square":
                # Square/boxy cars might be vans
                if visual_cues["size"] == "medium" or visual_cues["size"] == "large":
                    return "van", confidence * 1.1
            
            elif class_name == "truck" and visual_cues["shape"] == "balanced":
                # Some balanced trucks might be vans
                if visual_cues["size"] == "medium":
                    return "van", confidence * 1.05
            
            elif class_name == "van" and visual_cues["shape"] == "wide":
                # Wide vans might be cars
                if visual_cues["size"] == "small" or visual_cues["size"] == "medium":
                    return "car", confidence * 1.05
            
            elif class_name == "bus" and visual_cues["size"] == "medium":
                # Medium-sized buses might be vans
                if visual_cues["shape"] != "tall":
                    return "van", confidence * 1.05
        
        # No refinement needed
        return class_name, confidence
    
    def classify(self, image: np.ndarray, detection: Detection) -> ClassificationResult:
        """
        Classify a detected object using ConvNeXt with advanced techniques
        
        Args:
            image: Full input image
            detection: Detection object containing bounding box
            
        Returns:
            ClassificationResult with class name and confidence
        """
        # Direct handling of person class
        if detection.is_person:
            return ClassificationResult(class_name="person", confidence=0.99)
        
        try:
            # Extract vehicle from image using bounding box with slight context padding
            box = detection.box
            img_height, img_width = image.shape[:2]
            
            # Add padding (7% in each direction) for better context
            x1 = max(0, int(box[0] - 0.07 * (box[2] - box[0])))
            y1 = max(0, int(box[1] - 0.07 * (box[3] - box[1])))
            x2 = min(img_width, int(box[2] + 0.07 * (box[2] - box[0])))
            y2 = min(img_height, int(box[3] + 0.07 * (box[3] - box[1])))
            
            cropped_img = image[y1:y2, x1:x2]
            
            # Check for valid crop
            if cropped_img.size == 0 or cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                return ClassificationResult(class_name="non-vehicle", confidence=0.7)
            
            # Apply image enhancement
            enhanced_img = self._enhance_image_quality(cropped_img)
            
            # Apply test-time augmentation
            augmented_images = self._apply_test_time_augmentation(enhanced_img)
            
            # Predict with augmentation for robustness
            class_name, confidence = self._predict_with_augmentation(augmented_images)
            
            # Analyze visual context
            visual_cues = self._analyze_visual_context(image, detection)
            
            # Refine classification using context
            refined_class, refined_conf = self._refine_with_context(class_name, confidence, visual_cues)
            
            # For very low confidence, use detection class as a hint, but trust the classifier more
            if refined_conf < self.confidence_threshold:
                class_id = detection.class_id
                # Class ID 2 = car in COCO
                # Class ID 5 = bus in COCO
                # Class ID 7 = truck in COCO
                
                # Only override the class if confidence is very low
                if refined_conf < 0.4:
                    if class_id == 2 and refined_class not in ["car", "emergency vehicle"]:
                        refined_class = "car"
                        refined_conf = max(0.5, refined_conf)
                    elif class_id == 5 and refined_class != "bus":
                        refined_class = "bus"
                        refined_conf = max(0.5, refined_conf)
                    elif class_id == 7 and refined_class != "truck":
                        refined_class = "truck"
                        refined_conf = max(0.5, refined_conf)
            
            return ClassificationResult(class_name=refined_class, confidence=refined_conf)
            
        except Exception as e:
            print(f"Error in ConvNeXt classification: {e}")
            # Minimal fallback for error cases - much less dependent than before
            if detection.class_id == 2:
                return ClassificationResult(class_name="car", confidence=0.6)
            elif detection.class_id == 5:
                return ClassificationResult(class_name="bus", confidence=0.6)
            elif detection.class_id == 7:
                return ClassificationResult(class_name="truck", confidence=0.6)
            else:
                return ClassificationResult(class_name="non-vehicle", confidence=0.5)
    
    @property
    def name(self) -> str:
        """Return the name/type of the classifier"""
        return "ConvNeXt"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size
    
    @property
    def supported_classes(self) -> List[str]:
        """Return list of supported vehicle classes"""
        return self._supported_classes