"""
vehicle_classifier.py - Main pipeline that coordinates detection, classification, and visualization
"""

import os
import time
import random
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any, Optional
from PIL import Image

from core.interfaces import (
    ObjectDetector, 
    Classifier, 
    ClassificationRefiner,
    ObjectPrioritizer, 
    Visualizer,
    Detection,
    ClassificationResult
)


class VehicleClassifier:
    """Main class that coordinates object detection, classification, and visualization"""
    
    def __init__(self, 
                detector: ObjectDetector,
                classifier: Classifier,
                refiner: Optional[ClassificationRefiner] = None,
                prioritizer: Optional[ObjectPrioritizer] = None,
                visualizer: Optional[Visualizer] = None):
        """
        Initialize the vehicle classifier pipeline
        
        Args:
            detector: Object detector implementation
            classifier: Fine-grained classifier implementation
            refiner: Optional classification refiner
            prioritizer: Optional object prioritizer
            visualizer: Optional visualizer
        """
        self.detector = detector
        self.classifier = classifier
        self.refiner = refiner
        self.prioritizer = prioritizer
        self.visualizer = visualizer
        
        # Log the components being used
        print(f"Initializing VehicleClassifier with:")
        print(f"  - Detector: {detector.name} ({detector.model_size})")
        print(f"  - Classifier: {classifier.name} ({classifier.model_size})")
        print(f"  - Supported classes: {classifier.supported_classes}")
    
    def process_image(self, image, show_result: bool = True):
        """
        Process a single image and return results
        
        Args:
            image: Input image as numpy array or PIL image
            show_result: Whether to create visualization
            
        Returns:
            Tuple of (result_image, main_object_class, main_object_conf, class_names, confidences, detections)
        """
        # Convert to numpy array if PIL image
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Make a copy for visualization
        result_image = image.copy()
        
        # Detect objects
        detections = self.detector.detect(image)
        
        class_names = []
        confidences = []
        classifications = []
        
        # Process each detection for classification
        if detections:
            for detection in detections:
                # Classify the detection
                classification = self.classifier.classify(image, detection)
                
                # Refine the classification if a refiner is available
                # if self.refiner:
                #     classification = self.refiner.refine(
                #         image, detection, classification, image.shape
                #     )
                
                # Store the results
                class_names.append(classification.class_name)
                confidences.append(classification.confidence)
                classifications.append(classification)
            
            # Find the most prominent active object if a prioritizer is available
            if self.prioritizer:
                main_object_idx, _ = self.prioritizer.find_main_object(
                    detections, classifications, image.shape
                )
            else:
                # Default to the first detection if no prioritizer
                main_object_idx = 0 if detections else None
            
            # Set main object class and confidence
            if main_object_idx is not None:
                main_object_class = class_names[main_object_idx]
                main_object_conf = confidences[main_object_idx]
            else:
                main_object_class = "no-vehicle"
                main_object_conf = 1.0
            
            # Create annotated image if requested
            if show_result and self.visualizer:
                result_image = self.visualizer.draw_annotations(
                    image, 
                    detections, 
                    classifications, 
                    main_object_idx
                )
        else:
            # No objects detected
            main_object_class = "no-vehicle"
            main_object_conf = 1.0
            main_object_idx = None
            
            if show_result and self.visualizer:
                # Add "No objects detected" text
                cv2.putText(
                    result_image,
                    "No objects detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
        
        return result_image, main_object_class, main_object_conf, class_names, confidences, detections
    
    def batch_process(self, 
                     input_dir: str = "./img", 
                     output_dir: str = "./output", 
                     num_images: int = 50, 
                     save_visualizations: bool = True,
                     include_detector_name: bool = True,
                     include_classifier_name: bool = True):
        """
        Process multiple images from input directory and save results
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output files
            num_images: Number of images to process (0 for all)
            save_visualizations: Whether to save visualization images
            include_detector_name: Whether to include detector name in output filenames
            include_classifier_name: Whether to include classifier name in output filenames
            
        Returns:
            Tuple of (results, csv_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create identifiers for the models used
        detector_id = f"_{self.detector.name}_{self.detector.model_size}" if include_detector_name else ""
        classifier_id = f"_{self.classifier.name}_{self.classifier.model_size}" if include_classifier_name else ""
        
        # Create a CSV file for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"vehicle_classification{detector_id}{classifier_id}_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        with open(csv_path, 'w') as f:
            f.write("image_name,main_object_class,main_object_confidence,all_detected_classes\n")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        
        for file in os.listdir(input_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(input_dir, file))
        
        # Select random subset if needed
        if 0 < num_images < len(image_files):
            image_files = random.sample(image_files, num_images)
        
        # Process each image
        results = []
        total_start_time = time.time()
        
        for i, img_path in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {img_path}")
            
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"  Warning: Could not load image {img_path}")
                    continue
                
                # Process image
                start_time = time.time()
                result_image, main_class, main_conf, all_classes, all_confs, detections = self.process_image(
                    image, save_visualizations
                )
                process_time = time.time() - start_time
                
                # Save results
                base_name = os.path.basename(img_path)
                filename, ext = os.path.splitext(base_name)
                
                # Save visualization if requested
                if save_visualizations:
                    output_path = os.path.join(
                        output_dir, 
                        f"{filename}{detector_id}{classifier_id}_annotated{ext}"
                    )
                    cv2.imwrite(output_path, result_image)
                
                # Append to CSV
                all_detected = ";".join([f"{cls}:{conf:.2f}" for cls, conf in zip(all_classes, all_confs)])
                with open(csv_path, 'a') as f:
                    f.write(f"{base_name},{main_class},{main_conf:.4f},{all_detected}\n")
                
                # Store result for summary
                results.append({
                    'filename': base_name,
                    'main_class': main_class,
                    'main_conf': main_conf,
                    'process_time': process_time
                })
                
                print(f"  Main object: {main_class} (Confidence: {main_conf:.2f}) - Time: {process_time:.2f}s")
                
            except Exception as e:
                print(f"  Error processing {img_path}: {str(e)}")
        
        # Print summary
        total_time = time.time() - total_start_time
        print("\nBatch Processing Summary:")
        print(f"Processed {len(results)} images in {total_time:.2f} seconds")
        print(f"Average processing time: {total_time/max(len(results),1):.2f} seconds per image")
        
        # Count object classes
        class_counts = {}
        for result in results:
            cls = result['main_class']
            if cls in class_counts:
                class_counts[cls] += 1
            else:
                class_counts[cls] = 1
        
        print("\nObject Class Distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} ({count/len(results)*100:.1f}%)")
        
        return results, csv_path