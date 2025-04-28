"""
yolo_world_standalone.py - Complete standalone implementation of YOLO-World for
both detection and classification of vehicles in a single pass
"""

import os
import argparse
import time
import csv
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
import random
from typing import List, Dict, Tuple, Optional, Union


class YOLOWorldVehicleSystem:
    """
    Standalone YOLO-World system for vehicle detection and classification
    Uses YOLO-World's open-vocabulary detection for direct fine-grained vehicle classification
    """
    
    def __init__(self, 
                 model_size: str = "medium",
                 confidence_threshold: float = 0.25,
                 custom_model_path: Optional[str] = None):
        """
        Initialize the YOLO-World Vehicle System
        
        Args:
            model_size: Size of the model (small, medium, large)
            confidence_threshold: Detection confidence threshold
            custom_model_path: Path to custom model weights
        """
        # Model configuration
        self.confidence_threshold = confidence_threshold
        self.model_paths = {
            "small": "yolov8s-worldv2.pt",   # YOLO-World models
            "medium": "yolov8m-worldv2.pt",
            "large": "yolov8l-worldv2.pt"
        }
        
        # Load model
        if custom_model_path and os.path.exists(custom_model_path):
            self.model_path = custom_model_path
        else:
            if model_size not in self.model_paths:
                raise ValueError(f"Invalid model size: {model_size}. Choose from {', '.join(self.model_paths.keys())}.")
            self.model_path = self.model_paths[model_size]
            
            # Check if model exists in models/weights directory
            weights_dir = os.path.join("models", "weights")
            weights_path = os.path.join(weights_dir, self.model_path)
            if os.path.exists(weights_path):
                self.model_path = weights_path
        
        print(f"Loading YOLO-World model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Define detailed class prompts for fine-grained vehicle classification
        self.class_prompts = [
            "person walking or standing",
            "car: sedan, hatchback, coupe, convertible or SUV",
            "van: delivery van, minivan or panel van",
            "truck: pickup truck, box truck, flatbed truck or delivery truck",
            "bus: city bus, school bus, coach bus or double-decker bus",
            "emergency vehicle: ambulance, police car, fire truck or emergency response vehicle",
            "motorcycle or scooter",
        ]
        
        # Set classes using YOLO-World's set_classes method
        try:
            self.model.set_classes(self.class_prompts)
            print("Successfully set custom classes for YOLO-World model")
        except Exception as e:
            print(f"Warning: Could not set custom classes: {e}")
            print("Will attempt to use classes during prediction instead")
    
    def process_image(self, image: np.ndarray, 
                    show_confidence: bool = True, 
                    detection_only: bool = False) -> Tuple[np.ndarray, Dict, List]:
        """
        Process an image to detect and classify vehicles
        
        Args:
            image: Input image (BGR format from OpenCV)
            show_confidence: Whether to show confidence values in visualization
            detection_only: If True, only perform detection without classification
            
        Returns:
            Tuple of (annotated image, main object dict, all detections list)
        """
        # Run YOLO-World inference
        try:
            if detection_only:
                # Standard detection without custom classes
                results = self.model.predict(
                    image,
                    conf=self.confidence_threshold,
                    verbose=False
                )
            else:
                # Use the set classes (should already be configured in __init__)
                results = self.model.predict(
                    image,
                    conf=self.confidence_threshold,
                    verbose=False
                )
        except Exception as e:
            print(f"Prediction error: {e}")
            print("Falling back to basic detection...")
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                verbose=False
            )
        
        # Process results
        detections = []
        
        if results and len(results) > 0:
            # Extract all detections
            for r in results:
                if not hasattr(r, 'boxes') or len(r.boxes) == 0:
                    continue
                    
                for i in range(len(r.boxes)):
                    # Get box coordinates
                    box = r.boxes[i].xyxy.cpu().numpy()[0]
                    
                    # Get class info
                    class_id = int(r.boxes[i].cls.cpu().numpy()[0])
                    
                    # Get the class name
                    if hasattr(r, 'names') and class_id in r.names:
                        original_class = r.names[class_id]
                    else:
                        original_class = f"class_{class_id}"
                    
                    # Map to our vehicle types based on original class name
                    if detection_only:
                        # Basic mapping for standard detection
                        if "person" in original_class.lower():
                            class_name = "non-vehicle"
                        elif any(x in original_class.lower() for x in ["car", "sedan", "coupe", "suv"]):
                            class_name = "car"
                        elif "van" in original_class.lower():
                            class_name = "van"
                        elif "truck" in original_class.lower():
                            class_name = "truck"
                        elif "bus" in original_class.lower():
                            class_name = "bus"
                        elif any(x in original_class.lower() for x in ["ambulance", "police", "emergency", "fire"]):
                            class_name = "emergency vehicle"
                        elif any(x in original_class.lower() for x in ["motorcycle", "scooter", "bike"]):
                            class_name = "motorcycle"
                        else:
                            class_name = "unknown"
                    else:
                        # Use our class prompts mapping
                        if class_id < len(self.class_prompts):
                            prompt = self.class_prompts[class_id]
                            if "person" in prompt:
                                class_name = "non-vehicle"
                            elif "car:" in prompt:
                                class_name = "car"
                            elif "van:" in prompt:
                                class_name = "van"
                            elif "truck:" in prompt:
                                class_name = "truck"
                            elif "bus:" in prompt:
                                class_name = "bus"
                            elif "emergency" in prompt:
                                class_name = "emergency vehicle"
                            elif "motorcycle" in prompt:
                                class_name = "motorcycle"
                            else:
                                class_name = "unknown"
                        else:
                            # If class_id is out of range of our prompts
                            # Fall back to original class name mapping
                            if "person" in original_class.lower():
                                class_name = "non-vehicle"
                            elif "car" in original_class.lower():
                                class_name = "car"
                            elif "van" in original_class.lower():
                                class_name = "van"
                            elif "truck" in original_class.lower():
                                class_name = "truck"
                            elif "bus" in original_class.lower():
                                class_name = "bus"
                            elif any(x in original_class.lower() for x in ["ambulance", "police", "emergency", "fire"]):
                                class_name = "emergency vehicle"
                            elif any(x in original_class.lower() for x in ["motorcycle", "scooter", "bike"]):
                                class_name = "motorcycle"
                            else:
                                class_name = "unknown"
                    
                    # Get confidence
                    confidence = float(r.boxes[i].conf.cpu().numpy()[0])
                    
                    detections.append({
                        'box': box,
                        'class_id': class_id, 
                        'class_name': class_name,
                        'original_class': original_class,
                        'confidence': confidence,
                        'is_person': class_name == "non-vehicle",
                        'is_vehicle': class_name != "non-vehicle" and class_name != "unknown"
                    })
        
        # Create annotated image
        annotated_image = self._visualize_detections(
            image.copy(), 
            detections, 
            show_confidence
        )
        
        # Find main object (largest vehicle near center)
        main_object = self._find_main_object(image.shape, detections)
        
        return annotated_image, main_object, detections
    
    def _find_main_object(self, 
                        image_shape: Tuple[int, int, int], 
                        detections: List[Dict]) -> Optional[Dict]:
        """
        Find the main object in the image based on size and position
        
        Args:
            image_shape: Shape of the image (height, width, channels)
            detections: List of detection dictionaries
            
        Returns:
            Dictionary for the main object or None if no objects detected
        """
        if not detections:
            return None
            
        # Define scoring parameters
        center_weight = 2.0  # Weight for object proximity to center
        size_weight = 1.1    # Weight for object size
        emergency_bonus = 1.5  # Bonus for emergency vehicles (high priority)
        
        # Calculate center of image
        img_height, img_width = image_shape[:2]
        img_center_x = img_width / 2
        img_center_y = img_height / 2
        
        # Score each detection
        scores = []
        for detection in detections:
            box = detection['box']
            class_name = detection['class_name']
            
            # Calculate center of box
            box_center_x = (box[0] + box[2]) / 2
            box_center_y = (box[1] + box[3]) / 2
            
            # Distance from center (normalized)
            distance = np.sqrt(
                ((box_center_x - img_center_x) / img_width)**2 + 
                ((box_center_y - img_center_y) / img_height)**2
            )
            centrality = 1 - min(distance, 1.0)
            
            # Size score (normalized)
            area = (box[2] - box[0]) * (box[3] - box[1]) / (img_height * img_width)
            
            # Apply bonuses
            emergency_factor = emergency_bonus if class_name == "emergency vehicle" else 1.0
            
            # Calculate final score
            score = (centrality * center_weight + area * size_weight) * emergency_factor
            scores.append(score)
        
        # Find best scoring object
        best_idx = np.argmax(scores) if scores else 0
        
        return detections[best_idx] if detections else None
    
    def _visualize_detections(self, 
                            image: np.ndarray, 
                            detections: List[Dict],
                            show_confidence: bool = True) -> np.ndarray:
        """
        Create visualization with bounding boxes and labels
        
        Args:
            image: Original image
            detections: List of detection dictionaries 
            show_confidence: Whether to show confidence values
            
        Returns:
            Annotated image
        """
        # Define colors for different classes
        colors = {
            "non-vehicle": (255, 0, 255),     # Magenta
            "car": (0, 0, 255),          # Red
            "van": (0, 140, 255),        # Orange
            "truck": (0, 69, 255),       # Orange-red
            "bus": (0, 255, 0),          # Green
            "emergency vehicle": (255, 0, 127),  # Purple
            "motorcycle": (255, 255, 0)  # Cyan
        }
        
        # Draw each detection
        for detection in detections:
            # Get detection info
            box = detection['box'].astype(int)
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Determine color
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(image, 
                        (box[0], box[1]), 
                        (box[2], box[3]), 
                        color, 2)
            
            # Create label
            if show_confidence:
                label = f"{class_name} {confidence:.2f}"
            else:
                label = class_name
                
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, 
                        (box[0], box[1] - text_size[1] - 10), 
                        (box[0] + text_size[0], box[1]), 
                        color, -1)
            
            # Draw label text
            cv2.putText(image, label, 
                      (box[0], box[1] - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # If no detections, add text indicating this
        if not detections:
            cv2.putText(image, 
                      "No vehicles detected", 
                      (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (0, 0, 255), 2)
        
        return image
    
    def batch_process(self, 
                     input_dir: str,
                     output_dir: str,
                     num_images: int = 0,
                     save_visualizations: bool = True) -> Tuple[List[Dict], str]:
        """
        Process multiple images from a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output
            num_images: Number of images to process (0 for all)
            save_visualizations: Whether to save annotated images
            
        Returns:
            Tuple of (results list, CSV path)
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        
        for file in os.listdir(input_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(input_dir, file))
        
        # Select random subset if needed
        if 0 < num_images < len(image_files):
            image_files = random.sample(image_files, num_images)
        
        # Create CSV for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"yolo_world_results_{timestamp}.csv")
        
        with open(csv_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "image_name", 
                "main_class", 
                "main_confidence", 
                "processing_time",
                "all_detections"
            ])
        
        # Process each image
        results = []
        total_start = time.time()
        
        for i, img_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_path}")
            
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"  Error: Could not load image {img_path}")
                    continue
                
                # Process image
                start_time = time.time()
                annotated, main_object, detections = self.process_image(image)
                process_time = time.time() - start_time
                
                # Prepare results
                filename = os.path.basename(img_path)
                
                # Get main object info
                if main_object:
                    main_class = main_object['class_name']
                    main_conf = main_object['confidence']
                else:
                    main_class = "none"
                    main_conf = 0.0
                
                # Save visualization if requested
                if save_visualizations:
                    output_path = os.path.join(
                        output_dir, 
                        f"{os.path.splitext(filename)[0]}_annotated.jpg"
                    )
                    cv2.imwrite(output_path, annotated)
                
                # Create detection summary
                all_detections = ";".join([
                    f"{d['class_name']}:{d['confidence']:.2f}" 
                    for d in detections
                ])
                
                # Add to results list
                results.append({
                    'filename': filename,
                    'main_class': main_class,
                    'main_confidence': main_conf,
                    'processing_time': process_time,
                    'all_detections': all_detections
                })
                
                # Append to CSV
                with open(csv_path, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        filename,
                        main_class,
                        f"{main_conf:.4f}",
                        f"{process_time:.4f}",
                        all_detections
                    ])
                
                print(f"  Main object: {main_class} ({main_conf:.2f}) - Time: {process_time:.3f}s")
                
            except Exception as e:
                print(f"  Error processing {img_path}: {str(e)}")
        
        # Print summary
        total_time = time.time() - total_start
        print(f"\nProcessed {len(results)} images in {total_time:.2f}s")
        print(f"Average time per image: {total_time/max(len(results),1):.3f}s")
        
        # Summarize class distribution
        class_counts = {}
        for r in results:
            cls = r['main_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print("\nClass distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} ({count/len(results)*100:.1f}%)")
        
        return results, csv_path


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLO-World Vehicle Detection and Classification")
    
    parser.add_argument("--input", required=True,
                      help="Input image path or directory of images")
    parser.add_argument("--output", default="./output",
                      help="Output directory for results")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], 
                      default="medium", help="YOLO-World model size")
    parser.add_argument("--confidence", type=float, default=0.25,
                      help="Detection confidence threshold")
    parser.add_argument("--batch", action="store_true",
                      help="Process a directory of images (batch mode)")
    parser.add_argument("--num-images", type=int, default=0,
                      help="Number of images to process in batch mode (0 for all)")
    parser.add_argument("--detection-only", action="store_true",
                      help="Use regular detection without open-vocabulary classification")
    parser.add_argument("--custom-model", 
                      help="Path to custom model weights")
    
    args = parser.parse_args()
    
    # Initialize YOLO-World system
    system = YOLOWorldVehicleSystem(
        model_size=args.model_size,
        confidence_threshold=args.confidence,
        custom_model_path=args.custom_model if hasattr(args, 'custom_model') else None
    )
    
    if args.batch or os.path.isdir(args.input):
        # Batch processing mode
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            return
            
        # Process directory
        system.batch_process(
            input_dir=args.input,
            output_dir=args.output,
            num_images=args.num_images,
            save_visualizations=True
        )
    else:
        # Single image mode
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} is not a file")
            return
            
        # Load and process image
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not load image {args.input}")
            return
            
        # Process image
        start_time = time.time()
        use_detection_only = args.detection_only if hasattr(args, 'detection_only') else False
        annotated, main_object, detections = system.process_image(image, detection_only=use_detection_only)
        process_time = time.time() - start_time
        
        # Print results
        print(f"Processing time: {process_time:.3f}s")
        
        if main_object:
            print(f"Main object: {main_object['class_name']} ({main_object['confidence']:.2f})")
        else:
            print("No main object detected")
            
        print("All detections:")
        for i, d in enumerate(detections):
            print(f"  {i+1}. {d['class_name']} ({d['confidence']:.2f})")
        
        # Save result
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(
            args.output,
            f"{os.path.splitext(os.path.basename(args.input))[0]}_result.jpg"
        )
        cv2.imwrite(output_path, annotated)
        print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()