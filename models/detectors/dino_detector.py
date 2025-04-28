"""
dino_detector.py - DINOv2-based object detector implementation
"""

import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from core.interfaces import ObjectDetector, Detection


class DINODetector(ObjectDetector):
    """DINOv2-based object detector with attention-based object localization"""
    
    MODEL_SIZES = {
        "small": "facebook/dinov2-small",
        "medium": "facebook/dinov2-base",
        "large": "facebook/dinov2-large"
    }
    
    # COCO dataset class IDs for vehicles and persons
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    PERSON_CLASS_ID = 0  # person
    
    def __init__(self, 
                 model_size: str = "medium", 
                 confidence_threshold: float = 0.25,
                 custom_model_path: Optional[str] = None):
        """
        Initialize DINO detector
        
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
            
        print(f"Using device: {self.device}")
        
        # Load DINO model and processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing parameters
        self.target_size = 518  # DINO's default input size
        self.patch_size = 14 if "small" in self.model_path else 16  # Varies based on model
        
        # Set threshold for attention-based detection
        self.attention_threshold = 0.6
        
        # Initialize detection head
        self._initialize_detection_head()
    
    def _initialize_detection_head(self):
        """
        Initialize a detection head on top of DINO features
        """
        # Determine embedding dimension based on model size
        if "small" in self.model_path:
            embedding_dim = 384
        elif "base" in self.model_path:
            embedding_dim = 768
        else:  # large
            embedding_dim = 1024
        
        # Create detection head with class and box regression
        self.detection_head = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 9)  # 8 classes (COCO subset) + background
        ).to(self.device)
        
        # We don't train the detection head here, as it would be in a real implementation
        self.detection_head.eval()
        
        # Define class mapping (subset of COCO classes we care about)
        self.class_mapping = {
            0: 0,   # Person
            1: 2,   # Car
            2: 3,   # Motorcycle
            3: 5,   # Bus
            4: 7,   # Truck
            5: 1,   # Bicycle
            6: 4,   # Airplane
            7: 6,   # Train
        }
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for DINO model
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Processed tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL image
        pil_img = Image.fromarray(image_rgb)
        
        # Process for DINO
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        return pixel_values
    
    def _get_attention_map(self, attentions):
        """
        Process attention maps to extract region proposals
        
        Args:
            attentions: Attention maps from DINO
            
        Returns:
            Processed attention map
        """
        # Use the attention of the last layer, last head
        last_layer_attn = attentions[-1]  # Last layer
        head_idx = 0  # First head (can experiment with different heads)
        
        # Get CLS token attention map
        cls_attn = last_layer_attn[0, head_idx, 0, 1:]  # Skip the CLS token itself
        
        # Reshape to spatial dimensions
        num_patches = cls_attn.shape[0]
        size = int(np.sqrt(num_patches))
        attn_map = cls_attn.reshape(size, size).cpu().numpy()
        
        # Normalize to [0, 1]
        if attn_map.max() > attn_map.min():
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        
        return attn_map
    
    def _get_candidate_boxes(self, attn_map, image_shape):
        """
        Get candidate bounding boxes from attention map
        
        Args:
            attn_map: Attention map
            image_shape: Original image shape
            
        Returns:
            List of candidate boxes
        """
        # Resize attention map to original image size
        h, w = image_shape[:2]
        attn_map_resized = cv2.resize(attn_map, (w, h))
        
        # Apply threshold to get binary map
        binary_map = (attn_map_resized > self.attention_threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours and convert to bounding boxes
        candidate_boxes = []
        for contour in contours:
            # Skip small contours
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum area threshold
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            candidate_boxes.append([x, y, x + w, y + h])
        
        return candidate_boxes
    
    def _refine_detections(self, features, candidate_boxes, image_shape):
        """
        Refine detections using the detection head
        
        Args:
            features: DINO features
            candidate_boxes: Candidate bounding boxes
            image_shape: Original image shape
            
        Returns:
            List of refined detections
        """
        h, w = image_shape[:2]
        refined_detections = []
        
        # CLS token features
        cls_features = features[:, 0, :]  # [batch_size, embedding_dim]
        
        # Run detection head
        with torch.no_grad():
            outputs = self.detection_head(cls_features)
            
            # Interpret outputs as class probabilities and box refinement
            class_probs = F.softmax(outputs[0, :8], dim=0)  # First 8 values are class probabilities
            background_prob = torch.sigmoid(outputs[0, 8])  # Last value is background probability
            
            # Consider detection valid if background probability is low
            if background_prob < 0.5:
                # Get predicted class
                cls_id = torch.argmax(class_probs).item()
                confidence = class_probs[cls_id].item()
                
                # Skip if confidence is below threshold
                if confidence < self.confidence_threshold:
                    return []
                
                # Map to COCO class ID
                coco_cls_id = self.class_mapping[cls_id]
                
                # Only include person and vehicle classes
                if coco_cls_id == self.PERSON_CLASS_ID or coco_cls_id in self.VEHICLE_CLASS_IDS:
                    # If we have candidate boxes, use them
                    if candidate_boxes:
                        for box in candidate_boxes:
                            # Create detection
                            detection = Detection(
                                box=np.array(box, dtype=np.float32),
                                class_id=coco_cls_id,
                                confidence=confidence,
                                is_person=coco_cls_id == self.PERSON_CLASS_ID,
                                is_vehicle=coco_cls_id in self.VEHICLE_CLASS_IDS
                            )
                            refined_detections.append(detection)
                    else:
                        # If no candidate boxes, use whole image
                        box = np.array([0, 0, w, h], dtype=np.float32)
                        detection = Detection(
                            box=box,
                            class_id=coco_cls_id,
                            confidence=confidence * 0.8,  # Reduce confidence for whole-image detections
                            is_person=coco_cls_id == self.PERSON_CLASS_ID,
                            is_vehicle=coco_cls_id in self.VEHICLE_CLASS_IDS
                        )
                        refined_detections.append(detection)
        
        return refined_detections
    
    def _split_image_to_tiles(self, image, tile_size=518, overlap=0.3):
        """
        Split large image into overlapping tiles for better detection
        
        Args:
            image: Input image
            tile_size: Size of each tile
            overlap: Overlap between tiles (0-1)
            
        Returns:
            Tiles and their coordinates
        """
        h, w = image.shape[:2]
        tiles = []
        coords = []
        
        stride = int(tile_size * (1 - overlap))
        
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = image[y:y + tile_size, x:x + tile_size]
                tiles.append(tile)
                coords.append((x, y))
        
        # Handle edge cases - add right and bottom edges if needed
        if h > tile_size and (h - tile_size) % stride != 0:
            for x in range(0, w - tile_size + 1, stride):
                y = h - tile_size
                tile = image[y:y + tile_size, x:x + tile_size]
                tiles.append(tile)
                coords.append((x, y))
                
        if w > tile_size and (w - tile_size) % stride != 0:
            for y in range(0, h - tile_size + 1, stride):
                x = w - tile_size
                tile = image[y:y + tile_size, x:x + tile_size]
                tiles.append(tile)
                coords.append((x, y))
                
        # Add bottom-right corner if needed
        if h > tile_size and w > tile_size and (h - tile_size) % stride != 0 and (w - tile_size) % stride != 0:
            x, y = w - tile_size, h - tile_size
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
            coords.append((x, y))
        
        return tiles, coords
    
    def _merge_detections(self, tile_detections, coords, original_shape, iou_threshold=0.5):
        """
        Merge detections from tiles using non-maximum suppression
        
        Args:
            tile_detections: List of detections per tile
            coords: Coordinates of each tile
            original_shape: Original image shape
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Merged detections
        """
        # Adjust box coordinates to original image
        all_detections = []
        for detections, (tile_x, tile_y) in zip(tile_detections, coords):
            for detection in detections:
                box = detection.box.copy()
                box[0] += tile_x
                box[1] += tile_y
                box[2] += tile_x
                box[3] += tile_y
                
                # Create new detection with adjusted box
                adjusted_detection = Detection(
                    box=box,
                    class_id=detection.class_id,
                    confidence=detection.confidence,
                    is_person=detection.is_person,
                    is_vehicle=detection.is_vehicle
                )
                all_detections.append(adjusted_detection)
        
        # Group detections by class
        class_detections = {}
        for detection in all_detections:
            class_id = detection.class_id
            if class_id not in class_detections:
                class_detections[class_id] = []
            class_detections[class_id].append(detection)
        
        # Apply NMS for each class
        final_detections = []
        for class_id, detections in class_detections.items():
            # Sort by confidence (descending)
            detections.sort(key=lambda x: x.confidence, reverse=True)
            
            keep = []
            while detections:
                current = detections.pop(0)
                keep.append(current)
                
                i = 0
                while i < len(detections):
                    # Calculate IoU
                    box1 = current.box
                    box2 = detections[i].box
                    iou = self._calculate_iou(box1, box2)
                    
                    if iou > iou_threshold:
                        # Remove overlapping detection
                        detections.pop(i)
                    else:
                        i += 1
            
            final_detections.extend(keep)
        
        return final_detections
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        # Convert to [x1, y1, x2, y2] format if needed
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in the image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        h, w = image.shape[:2]
        
        # For large images, use tiling approach
        if h > 800 or w > 800:
            # Split image into tiles
            tiles, coords = self._split_image_to_tiles(image, tile_size=518, overlap=0.3)
            
            # Process each tile
            tile_detections = []
            for tile in tiles:
                # Preprocess tile
                tile_tensor = self._preprocess_image(tile)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(tile_tensor, output_attentions=True)
                    features = outputs.last_hidden_state
                    attentions = outputs.attentions
                
                # Get attention map for object localization
                attn_map = self._get_attention_map(attentions)
                
                # Get candidate boxes from attention map
                candidate_boxes = self._get_candidate_boxes(attn_map, tile.shape)
                
                # Refine detections
                detections = self._refine_detections(features, candidate_boxes, tile.shape)
                tile_detections.append(detections)
            
            # Merge detections from all tiles
            return self._merge_detections(tile_detections, coords, image.shape)
        else:
            # For smaller images, process directly
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(image_tensor, output_attentions=True)
                features = outputs.last_hidden_state
                attentions = outputs.attentions
            
            # Get attention map for object localization
            attn_map = self._get_attention_map(attentions)
            
            # Get candidate boxes from attention map
            candidate_boxes = self._get_candidate_boxes(attn_map, image.shape)
            
            # Refine detections
            return self._refine_detections(features, candidate_boxes, image.shape)
    
    @property
    def name(self) -> str:
        """Return the name/type of the detector"""
        return "DINO"
    
    @property
    def model_size(self) -> str:
        """Return the model size"""
        return self._model_size