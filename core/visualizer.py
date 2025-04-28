"""
visualizer.py - Handles visualization of detection and classification results
"""

from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import cv2

from core.interfaces import Visualizer, Detection, ClassificationResult


class DefaultVisualizer(Visualizer):
    """Handles visualization of detection and classification results"""
    
    def __init__(self, show_confidence: bool = True, show_class_colors: bool = True):
        """
        Initialize visualizer
        
        Args:
            show_confidence: Whether to show confidence scores
            show_class_colors: Whether to use different colors for different classes
        """
        self.show_confidence = show_confidence
        self.show_class_colors = show_class_colors
        
        # Define color mapping for different classes
        self.class_colors = {
            "car": (0, 0, 255),         # Red
            "van": (0, 165, 255),       # Orange
            "truck": (0, 128, 255),     # Deep orange
            "bus": (0, 255, 0),         # Green
            "emergency vehicle": (255, 0, 255),  # Purple
            "person": (255, 255, 0),    # Cyan
            "non-vehicle": (128, 128, 128)  # Gray
        }
        
        # Default color for unknown classes
        self.default_color = (255, 255, 255)  # White
        
        # Main object highlight color
        self.main_object_color = (0, 240, 0)  # Bright green
    
    def draw_annotations(self,
                        image: np.ndarray,
                        detections: List[Detection],
                        classifications: List[ClassificationResult],
                        main_idx: Optional[int] = None) -> np.ndarray:
        """
        Draw annotations on the image
        
        Args:
            image: Input image
            detections: List of Detection objects
            classifications: List of ClassificationResult objects
            main_idx: Index of the main object (if any)
            
        Returns:
            Annotated image
        """
        result_image = image.copy()
        
        # Draw each detection
        for i, (detection, classification) in enumerate(zip(detections, classifications)):
            box = detection.box
            x1, y1, x2, y2 = box
            class_name = classification.class_name
            confidence = classification.confidence
            
            # Determine color based on object type
            if i == main_idx:
                color = self.main_object_color
            elif self.show_class_colors:
                color = self.class_colors.get(class_name, self.default_color)
            else:
                color = (0, 0, 255)  # Default to red
            
            # Draw bounding box
            cv2.rectangle(result_image, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         color, 2)
            
            # Prepare label text
            if i == main_idx:
                label = f"MAIN: {class_name}" 
                if self.show_confidence:
                    label += f" {confidence:.2f}"
            else:
                label = f"{class_name}"
                if self.show_confidence:
                    label += f" {confidence:.2f}"
            
            # Add label above the bounding box
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_image, 
                         (int(x1), int(y1) - text_size[1] - 10), 
                         (int(x1) + text_size[0], int(y1)), 
                         color, -1)
            
            cv2.putText(result_image, label, 
                       (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add summary information on top of the image
        if main_idx is not None:
            main_class = classifications[main_idx].class_name
            main_conf = classifications[main_idx].confidence
            cv2.putText(
                result_image,
                f"Main object: {main_class}" + (f" ({main_conf:.2f})" if self.show_confidence else ""),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        elif detections:
            # Objects detected but none is the main one
            cv2.putText(
                result_image,
                "No active objects detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        else:
            # No objects detected 
            cv2.putText(
                result_image,
                "No objects detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            
        return result_image


class DebugVisualizer(DefaultVisualizer):
    """Extended visualizer with additional debug information"""
    
    def draw_annotations(self,
                        image: np.ndarray,
                        detections: List[Detection],
                        classifications: List[ClassificationResult],
                        main_idx: Optional[int] = None) -> np.ndarray:
        """
        Draw annotations with additional debug information
        
        Args:
            image: Input image
            detections: List of Detection objects
            classifications: List of ClassificationResult objects
            main_idx: Index of the main object (if any)
            
        Returns:
            Annotated image with debug information
        """
        # Get base annotations
        result_image = super().draw_annotations(image, detections, classifications, main_idx)
        
        # Add detector and classifier statistics
        if detections:
            # Calculate average confidence
            detection_conf = sum(d.confidence for d in detections) / len(detections)
            classification_conf = sum(c.confidence for c in classifications) / len(classifications)
            
            # Draw debug info at the bottom
            height = result_image.shape[0]
            cv2.putText(
                result_image,
                f"Objects: {len(detections)}, Det Conf: {detection_conf:.2f}, Class Conf: {classification_conf:.2f}",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            # Draw center motion region
            h, w = image.shape[:2]
            center_x1 = int(w * 0.2)
            center_y1 = int(h * 0.2)
            center_x2 = int(w * 0.8)
            center_y2 = int(h * 0.8)
            
            # Draw motion region rectangle
            cv2.rectangle(
                result_image,
                (center_x1, center_y1),
                (center_x2, center_y2),
                (0, 165, 255),  # Orange
                1
            )
            
            # Label the region
            cv2.putText(
                result_image,
                "Motion Region",
                (center_x1 + 5, center_y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 165, 255),
                1
            )
        
        return result_image