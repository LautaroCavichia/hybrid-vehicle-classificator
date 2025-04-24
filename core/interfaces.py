"""
interfaces.py - Abstract base classes defining the interfaces for all system components
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    """Standard detection result object"""
    box: np.ndarray  # [x1, y1, x2, y2] format
    confidence: float
    class_id: int
    is_person: bool = False
    is_vehicle: bool = False
    

@dataclass
class ClassificationResult:
    """Standard classification result object"""
    class_name: str  # Fine-grained class name
    confidence: float


class ObjectDetector(ABC):
    """Abstract base class for all detector implementations"""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize detector with optional parameters"""
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in the image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name/type of the detector"""
        pass
    
    @property
    @abstractmethod
    def model_size(self) -> str:
        """Return the model size (small, medium, large, etc.)"""
        pass


class Classifier(ABC):
    """Abstract base class for all classifier implementations"""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize classifier with optional parameters"""
        pass
    
    @abstractmethod
    def classify(self, image: np.ndarray, detection: Detection) -> ClassificationResult:
        """
        Classify the detected object
        
        Args:
            image: Full input image as numpy array (BGR format)
            detection: Detection object containing bounding box
            
        Returns:
            ClassificationResult object with class name and confidence
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name/type of the classifier"""
        pass
    
    @property
    @abstractmethod
    def model_size(self) -> str:
        """Return the model size (small, medium, large, etc.)"""
        pass
    
    @property
    @abstractmethod
    def supported_classes(self) -> List[str]:
        """Return list of supported vehicle classes"""
        pass


class ClassificationRefiner(ABC):
    """Abstract base class for classification refinement modules"""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize refiner with optional parameters"""
        pass
    
    @abstractmethod
    def refine(self, 
               image: np.ndarray,
               detection: Detection, 
               classification: ClassificationResult,
               image_shape: Tuple[int, int, int]) -> ClassificationResult:
        """
        Refine the classification based on additional heuristics
        
        Args:
            image: Full input image
            detection: Detection object
            classification: Initial classification result
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Refined ClassificationResult
        """
        pass


class ObjectPrioritizer(ABC):
    """Abstract base class for object prioritization"""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize prioritizer with optional parameters"""
        pass
    
    @abstractmethod
    def find_main_object(self, 
                         detections: List[Detection],
                         classifications: List[ClassificationResult],
                         image_shape: Tuple[int, int, int]) -> Tuple[Optional[int], float]:
        """
        Find the main/active object in the scene
        
        Args:
            detections: List of Detection objects
            classifications: List of ClassificationResult objects
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Tuple of (index of main object, score) or (None, 0.0) if no main object
        """
        pass


class Visualizer(ABC):
    """Abstract base class for visualization"""
    
    @abstractmethod
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
        pass