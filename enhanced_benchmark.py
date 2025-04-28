#!/usr/bin/env python3
"""
enhanced_benchmark.py - Improved benchmarking system for vehicle classification

This script integrates object detection and classification while tracking
bounding boxes and classification results for evaluation against ground truth.
"""

import os
import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from tabulate import tabulate
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import cv2

# Core interfaces and components
from core.interfaces import (
    ObjectDetector, 
    Classifier, 
    ClassificationRefiner,
    ObjectPrioritizer, 
    Visualizer,
    Detection,
    ClassificationResult
)

class EnhancedBenchmark:
    """Enhanced benchmarking system for vehicle classification"""
    
    def __init__(self, 
                 output_dir: str = "./benchmark_results",
                 annotations_path: Optional[str] = None,
                 refiner: Optional[ClassificationRefiner] = None,
                 prioritizer: Optional[ObjectPrioritizer] = None,
                 visualizer: Optional[Visualizer] = None):
        """
        Initialize the benchmark system
        
        Args:
            output_dir: Directory to save benchmark results
            annotations_path: Path to COCO annotations file for evaluation
            refiner: Classification refiner to use for all tests
            prioritizer: Object prioritizer to use for all tests
            visualizer: Visualizer to use for all tests
        """
        self.output_dir = output_dir
        self.refiner = refiner
        self.prioritizer = prioritizer
        self.visualizer = visualizer
        self.annotations_path = annotations_path
        self.ground_truth = self._load_ground_truth() if annotations_path else None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_ground_truth(self) -> Dict:
        """Load COCO format annotations"""
        if not self.annotations_path:
            return {}
            
        try:
            with open(self.annotations_path, "r") as f:
                data = json.load(f)
            
            # Create mapping from image filename to annotations
            gt_data = {}
            
            # Build image id to filename mapping
            image_id_map = {}
            for image in data["images"]:
                image_id_map[image["id"]] = os.path.basename(image["file_name"])
            
            # Map category IDs to names
            category_map = {}
            for category in data["categories"]:
                category_map[category["id"]] = category["name"]
            
            # Process annotations
            for annotation in data["annotations"]:
                image_file = image_id_map[annotation["image_id"]]
                
                if image_file not in gt_data:
                    gt_data[image_file] = []
                
                # COCO format is [x, y, width, height]
                # Convert to [x1, y1, x2, y2] format
                x, y, w, h = annotation["bbox"]
                bbox = [x, y, x + w, y + h]
                
                # Add the annotation
                gt_data[image_file].append({
                    "bbox": bbox,
                    "category": category_map[annotation["category_id"]]
                })
            
            print(f"Loaded ground truth annotations for {len(gt_data)} images")
            return gt_data
            
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return {}
    
    def _select_main_vehicle(self, annotations: List[Dict]) -> Dict:
        """
        Select the main vehicle from multiple annotations
        
        Strategy: 
        1. If only one annotation, use it
        2. Otherwise, select the largest one (likely to be the main vehicle)
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            Selected annotation
        """
        if not annotations:
            return None
            
        if len(annotations) == 1:
            return annotations[0]
            
        # Find the annotation with the largest area
        max_area = 0
        main_annotation = annotations[0]
        
        for ann in annotations:
            # Calculate area - bbox format is [x1, y1, x2, y2]
            area = (ann["bbox"][2] - ann["bbox"][0]) * (ann["bbox"][3] - ann["bbox"][1])
            
            if area > max_area:
                max_area = area
                main_annotation = ann
                
        return main_annotation
    
    def run_benchmark(self, 
                     detectors: List[ObjectDetector] = None,
                     classifiers: List[Classifier] = None,
                     end_to_end_systems: List[Any] = None,
                     input_dir: str = None,
                     num_images: Optional[int] = None,
                     save_visualizations: bool = True,
                     run_name: Optional[str] = None):
        """
        Run benchmark on all combinations of detectors and classifiers,
        and also on end-to-end systems
        
        Args:
            detectors: List of detector implementations to test
            classifiers: List of classifier implementations to test
            end_to_end_systems: List of end-to-end systems that handle both detection and classification
            input_dir: Directory containing input images
            num_images: Number of images to process (None for all)
            save_visualizations: Whether to save visualization images
            run_name: Optional name for this benchmark run
            
        Returns:
            Dictionary of benchmark results
        """
        if run_name is None:
            run_name = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
        
        benchmark_output_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(benchmark_output_dir, exist_ok=True)
        
        # Find images
        image_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        # Limit number of images if specified
        if num_images is not None and num_images < len(image_files):
            image_files = image_files[:num_images]
        
        print(f"Found {len(image_files)} images for processing")
        
        # Results dict for all combinations
        all_results = {}
        summary_metrics = []
        
        # Run detector-classifier combinations
        if detectors and classifiers:
            for detector in detectors:
                for classifier in classifiers:
                    # Skip if this is an end-to-end wrapper used in both lists
                    if hasattr(detector, 'process_image') and detector is classifier:
                        continue
                        
                    combo_name = f"{detector.name}_{detector.model_size}_{classifier.name}_{classifier.model_size}"
                    print(f"\n===== Testing {combo_name} =====")
                    
                    # Create output directory for this combination
                    combo_output_dir = os.path.join(benchmark_output_dir, combo_name)
                    os.makedirs(combo_output_dir, exist_ok=True)
                    
                    # Process images with this combination
                    result = self._process_with_detector_classifier(
                        detector, classifier, image_files, combo_output_dir, save_visualizations
                    )
                    
                    all_results[combo_name] = result
                    summary_metrics.append(result)
        
        # Run end-to-end systems
        if end_to_end_systems:
            for system in end_to_end_systems:
                system_name = f"{system.name}_{system.model_size}_end_to_end"
                print(f"\n===== Testing {system_name} =====")
                
                # Create output directory for this system
                system_output_dir = os.path.join(benchmark_output_dir, system_name)
                os.makedirs(system_output_dir, exist_ok=True)
                
                # Process images with this system
                result = self._process_with_end_to_end_system(
                    system, image_files, system_output_dir, save_visualizations
                )
                
                all_results[system_name] = result
                summary_metrics.append(result)
        
        # Save overall results as JSON
        results_file = os.path.join(benchmark_output_dir, "benchmark_results.json")
        
        # Create a copy without the detailed results to save as JSON
        json_results = {}
        for k, v in all_results.items():
            json_results[k] = {key: val for key, val in v.items() if key != 'results'}
            
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Generate summary reports
        self._generate_reports(summary_metrics, benchmark_output_dir)
        
        # Evaluate against ground truth if available
        if self.ground_truth:
            self._evaluate_against_ground_truth(all_results, benchmark_output_dir)
        
        return all_results
    
    def _process_with_detector_classifier(self, 
                                        detector, 
                                        classifier, 
                                        image_files, 
                                        output_dir, 
                                        save_visualizations):
        """Process images with a detector-classifier combination"""
        # Prepare results list and CSV
        combo_results = []
        csv_path = os.path.join(output_dir, "results.csv")
        
        # CSV header fields
        csv_fields = ['image_name', 'main_box', 'main_object_class', 'main_object_confidence', 
                      'detection_time', 'classification_time', 'total_time']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()
            
            # Process each image
            total_start_time = time.time()
            
            for img_path in tqdm(image_files, desc=f"Processing with {detector.name}+{classifier.name}"):
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                    
                    # Detect objects
                    detection_start = time.time()
                    detections = detector.detect(img)
                    detection_time = time.time() - detection_start
                    
                    # Classify each detected object
                    classifications = []
                    classification_start = time.time()
                    
                    for det in detections:
                        # Only classify vehicle detections
                        if det.is_vehicle:
                            cls_result = classifier.classify(img, det)
                            
                            # Apply refiner if available
                            if self.refiner:
                                cls_result = self.refiner.refine(img, det, cls_result, img.shape)
                                
                            classifications.append(cls_result)
                        else:
                            # Add placeholder for non-vehicle detections
                            classifications.append(ClassificationResult(class_name="non_vehicle", confidence=0.0))
                    
                    classification_time = time.time() - classification_start
                    
                    # Find main vehicle (if any)
                    main_idx = None
                    main_score = 0.0
                    
                    if self.prioritizer and detections:
                        main_idx, main_score = self.prioritizer.find_main_object(
                            detections, classifications, img.shape
                        )
                    elif detections:
                        # Simple fallback: choose the detection with highest confidence
                        main_idx = max(range(len(detections)), key=lambda i: detections[i].confidence)
                        main_score = detections[main_idx].confidence
                    
                    # Create visualization if requested
                    if save_visualizations and self.visualizer:
                        vis_img = self.visualizer.draw_annotations(
                            img.copy(), detections, classifications, main_idx
                        )
                        
                        # Save visualization
                        vis_path = os.path.join(output_dir, os.path.basename(img_path))
                        cv2.imwrite(vis_path, vis_img)
                    
                    # Extract details for main vehicle (if found)
                    main_box = None
                    main_class = "non_vehicle"
                    main_conf = 0.0
                    
                    if main_idx is not None and 0 <= main_idx < len(detections):
                        # Convert to string format for CSV: "x1,y1,x2,y2"
                        main_box = [float(x) for x in detections[main_idx].box]
                        main_box_str = f"{main_box[0]},{main_box[1]},{main_box[2]},{main_box[3]}"
                        
                        if main_idx < len(classifications):
                            main_class = classifications[main_idx].class_name
                            main_conf = classifications[main_idx].confidence
                    else:
                        main_box_str = ""
                    
                    # Calculate total processing time
                    total_time = detection_time + classification_time
                    
                    # Store result
                    result = {
                        'image_name': os.path.basename(img_path),
                        'main_box': main_box_str,
                        'main_object_class': main_class,
                        'main_object_confidence': main_conf,
                        'detection_time': detection_time,
                        'classification_time': classification_time,
                        'total_time': total_time
                    }
                    
                    combo_results.append(result)
                    writer.writerow(result)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Calculate overall metrics
        total_time = time.time() - total_start_time
        num_processed = len(combo_results)
        avg_time = total_time / max(num_processed, 1)
        
        # Compile class distribution
        class_distribution = {}
        for result in combo_results:
            cls = result['main_object_class']
            if cls in class_distribution:
                class_distribution[cls] += 1
            else:
                class_distribution[cls] = 1
        
        # Calculate average confidence
        avg_confidence = sum(float(r['main_object_confidence']) for r in combo_results) / max(num_processed, 1)
        
        # Calculate average detection and classification times
        avg_detection_time = sum(float(r['detection_time']) for r in combo_results) / max(num_processed, 1)
        avg_classification_time = sum(float(r['classification_time']) for r in combo_results) / max(num_processed, 1)
        
        # Store overall results
        overall_result = {
            'detector': detector.name,
            'detector_size': detector.model_size,
            'classifier': classifier.name,
            'classifier_size': classifier.model_size,
            'end_to_end': False,
            'num_images': num_processed,
            'total_time': total_time,
            'avg_time_per_image': avg_time,
            'avg_detection_time': avg_detection_time,
            'avg_classification_time': avg_classification_time,
            'avg_confidence': avg_confidence,
            'class_distribution': class_distribution,
            'output_dir': output_dir,
            'csv_path': csv_path,
            'results': combo_results
        }
        
        print(f"Processed {num_processed} images in {total_time:.2f}s ({avg_time:.4f}s per image)")
        print(f"Average confidence: {avg_confidence:.4f}")
        print(f"Average detection time: {avg_detection_time:.4f}s")
        print(f"Average classification time: {avg_classification_time:.4f}s")
        print("Class distribution:")
        for cls, count in class_distribution.items():
            print(f"  {cls}: {count} ({count/num_processed*100:.1f}%)")
        
        return overall_result
    
    def _process_with_end_to_end_system(self, 
                                      system, 
                                      image_files, 
                                      output_dir, 
                                      save_visualizations):
        """Process images with an end-to-end system"""
        # Prepare results list and CSV
        system_results = []
        csv_path = os.path.join(output_dir, "results.csv")
        
        # CSV header fields
        csv_fields = ['image_name', 'main_box', 'main_object_class', 'main_object_confidence', 
                      'processing_time']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()
            
            # Process each image
            total_start_time = time.time()
            
            for img_path in tqdm(image_files, desc=f"Processing with {system.name} end-to-end"):
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                    
                    # Process image with end-to-end system
                    processing_start = time.time()
                    detections, classifications = system.process_image(img)
                    processing_time = time.time() - processing_start
                    
                    # Find main vehicle (if any)
                    main_idx = None
                    main_score = 0.0
                    
                    if self.prioritizer and detections:
                        main_idx, main_score = self.prioritizer.find_main_object(
                            detections, classifications, img.shape
                        )
                    elif detections:
                        # Simple fallback: choose the detection with highest confidence
                        main_idx = max(range(len(detections)), key=lambda i: detections[i].confidence)
                        main_score = detections[main_idx].confidence
                    
                    # Create visualization if requested
                    if save_visualizations and self.visualizer:
                        vis_img = self.visualizer.draw_annotations(
                            img.copy(), detections, classifications, main_idx
                        )
                        
                        # Save visualization
                        vis_path = os.path.join(output_dir, os.path.basename(img_path))
                        cv2.imwrite(vis_path, vis_img)
                    
                    # Extract details for main vehicle (if found)
                    main_box = None
                    main_class = "non_vehicle"
                    main_conf = 0.0
                    
                    if main_idx is not None and 0 <= main_idx < len(detections):
                        # Convert to string format for CSV: "x1,y1,x2,y2"
                        main_box = [float(x) for x in detections[main_idx].box]
                        main_box_str = f"{main_box[0]},{main_box[1]},{main_box[2]},{main_box[3]}"
                        
                        if main_idx < len(classifications):
                            main_class = classifications[main_idx].class_name
                            main_conf = classifications[main_idx].confidence
                    else:
                        main_box_str = ""
                    
                    # Store result
                    result = {
                        'image_name': os.path.basename(img_path),
                        'main_box': main_box_str,
                        'main_object_class': main_class,
                        'main_object_confidence': main_conf,
                        'processing_time': processing_time,
                        'detection_time': processing_time,  # For compatibility with detector-classifier results
                        'classification_time': 0.0,  # For compatibility with detector-classifier results
                        'total_time': processing_time  # For compatibility with detector-classifier results
                    }
                    
                    system_results.append(result)
                    writer.writerow(result)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Calculate overall metrics
        total_time = time.time() - total_start_time
        num_processed = len(system_results)
        avg_time = total_time / max(num_processed, 1)
        
        # Compile class distribution
        class_distribution = {}
        for result in system_results:
            cls = result['main_object_class']
            if cls in class_distribution:
                class_distribution[cls] += 1
            else:
                class_distribution[cls] = 1
        
        # Calculate average confidence
        avg_confidence = sum(float(r['main_object_confidence']) for r in system_results) / max(num_processed, 1)
        
        # Calculate average processing time
        avg_processing_time = sum(float(r['processing_time']) for r in system_results) / max(num_processed, 1)
        
        # Store overall results
        overall_result = {
            'detector': system.name,
            'detector_size': system.model_size,
            'classifier': system.name,  # Same as detector for end-to-end
            'classifier_size': system.model_size,  # Same as detector for end-to-end
            'end_to_end': True,
            'num_images': num_processed,
            'total_time': total_time,
            'avg_time_per_image': avg_time,
            'avg_detection_time': avg_processing_time,  # For compatibility with detector-classifier results
            'avg_classification_time': 0.0,  # For compatibility with detector-classifier results
            'avg_confidence': avg_confidence,
            'class_distribution': class_distribution,
            'output_dir': output_dir,
            'csv_path': csv_path,
            'results': system_results
        }
        
        print(f"Processed {num_processed} images in {total_time:.2f}s ({avg_time:.4f}s per image)")
        print(f"Average confidence: {avg_confidence:.4f}")
        print(f"Average processing time: {avg_processing_time:.4f}s")
        print("Class distribution:")
        for cls, count in class_distribution.items():
            print(f"  {cls}: {count} ({count/num_processed*100:.1f}%)")
        
        return overall_result
    
    def _generate_reports(self, metrics: List[Dict], output_dir: str):
        """Generate benchmark summary reports and visualizations"""
        # Create summary table
        table_data = []
        headers = ["Type", "Detector", "Size", "Classifier", "Size", "Avg Time", "Det Time", "Cls Time", "Avg Conf", "# Images"]
        
        for item in metrics:
            # Determine system type
            system_type = "End-to-End" if item.get('end_to_end', False) else "Two-Stage"
            
            row = [
                system_type,
                item['detector'],
                item['detector_size'],
                item['classifier'],
                item['classifier_size'],
                f"{item['avg_time_per_image']:.4f}s",
                f"{item['avg_detection_time']:.4f}s",
                f"{item['avg_classification_time']:.4f}s",
                f"{item['avg_confidence']:.4f}",
                item['num_images']
            ]
            table_data.append(row)
        
        # Sort by processing time
        table_data.sort(key=lambda x: float(x[5][:-1]))
        
        # Generate text report
        report_path = os.path.join(output_dir, "benchmark_report.txt")
        with open(report_path, 'w') as f:
            f.write("# Vehicle Classification System Benchmark Results\n\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
            f.write("\n\n")
        
        # Generate performance comparison plot
        self._plot_performance_comparison(metrics, output_dir)
    
    def _plot_performance_comparison(self, metrics: List[Dict], output_dir: str):
        """Plot performance comparison of different model combinations"""
        # Extract data for plotting
        combos = []
        times = []
        det_times = []
        cls_times = []
        confs = []
        types = []  # End-to-End or Two-Stage
        
        for item in metrics:
            # Determine system type
            system_type = "End-to-End" if item.get('end_to_end', False) else "Two-Stage"
            types.append(system_type)
            
            # Create label for x-axis
            if system_type == "End-to-End":
                combo = f"{item['detector']} {item['detector_size']}\n(End-to-End)"
            else:
                combo = f"{item['detector']} {item['detector_size']}\n+ {item['classifier']} {item['classifier_size']}"
            
            combos.append(combo)
            times.append(item['avg_time_per_image'])
            det_times.append(item['avg_detection_time'])
            cls_times.append(item['avg_classification_time'])
            confs.append(item['avg_confidence'])
        
        # Sort by processing time
        sorted_indices = np.argsort(times)
        combos = [combos[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]
        det_times = [det_times[i] for i in sorted_indices]
        cls_times = [cls_times[i] for i in sorted_indices]
        confs = [confs[i] for i in sorted_indices]
        types = [types[i] for i in sorted_indices]
        
        # Create figure for time comparison
        plt.figure(figsize=(14, 8))
        
        # Create bar chart with stacked detection and classification times
        bars = []
        
        # Create handles for legend
        end_to_end_handle = None
        detection_handle = None
        classification_handle = None
        
        for i, (det_time, cls_time, system_type) in enumerate(zip(det_times, cls_times, types)):
            if system_type == "End-to-End":
                # For end-to-end, use a single bar with a different color
                bar = plt.bar([i], [det_time], color='mediumseagreen')
                if end_to_end_handle is None:
                    end_to_end_handle = bar
            else:
                # For two-stage, use stacked bars
                det_bar = plt.bar([i], [det_time], color='skyblue')
                cls_bar = plt.bar([i], [cls_time], bottom=[det_time], color='coral')
                if detection_handle is None:
                    detection_handle = det_bar
                if classification_handle is None:
                    classification_handle = cls_bar
        
        # Add legend with handles
        legend_handles = []
        legend_labels = []
        
        if end_to_end_handle is not None:
            legend_handles.append(end_to_end_handle)
            legend_labels.append('End-to-End Processing')
        if detection_handle is not None:
            legend_handles.append(detection_handle)
            legend_labels.append('Detection Time')
        if classification_handle is not None:
            legend_handles.append(classification_handle)
            legend_labels.append('Classification Time')
            
        plt.legend(legend_handles, legend_labels)
        
        plt.ylabel('Average Processing Time (s)')
        plt.title('Performance Comparison by Model Combination')
        plt.xticks(range(len(combos)), combos, rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=300)
        plt.close()
        
        # Create confidence comparison figure
        plt.figure(figsize=(14, 6))
        
        # Use different colors for end-to-end and two-stage systems
        bar_colors = ['mediumseagreen' if t == 'End-to-End' else 'lightblue' for t in types]
        plt.bar(range(len(combos)), confs, color=bar_colors)
        
        plt.ylabel('Average Confidence')
        plt.title('Confidence Comparison by Model Combination')
        plt.xticks(range(len(combos)), combos, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'confidence_comparison.png'), dpi=300)
        plt.close()

    def _evaluate_against_ground_truth(self, all_results: Dict, output_dir: str):
        """Evaluate benchmark results against ground truth annotations"""
        if not self.ground_truth:
            print("No ground truth annotations available for evaluation")
            return
        
        # Create evaluation directory
        eval_dir = os.path.join(output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Results for all combinations
        eval_metrics = []
        
        for combo_name, result in all_results.items():
            print(f"\nEvaluating {combo_name} against ground truth...")
            
            # Initialize evaluation metrics
            tp = 0  # True positives (correct class and sufficient IoU)
            fp = 0  # False positives (either wrong class or insufficient IoU)
            fn = 0  # False negatives (missed ground truth objects)
            
            correct_cls = 0  # Correct classifications
            total_cls = 0    # Total classifications attempted
            
            ious = []  # IoU values for all correct detections
            
            # Process each result
            for prediction in result['results']:
                img_name = prediction['image_name']
                
                # Check if we have ground truth for this image
                if img_name not in self.ground_truth:
                    continue
                
                gt_annotations = self.ground_truth[img_name]
                
                # Skip images without annotations
                if not gt_annotations:
                    continue
                
                # Get main ground truth annotation
                main_gt = self._select_main_vehicle(gt_annotations)
                
                # Check if we have a prediction
                if prediction['main_box'] and prediction['main_object_class'] != 'none':
                    # We have a prediction, extract bounding box
                    pred_parts = prediction['main_box'].split(',')
                    if len(pred_parts) == 4:
                        pred_box = [float(x) for x in pred_parts]
                        
                        # Calculate IoU
                        iou = self._calculate_iou(pred_box, main_gt["bbox"])
                        
                        # Check if IoU is sufficient (typically > 0.5)
                        if iou > 0.5:
                            ious.append(iou)
                            
                            # Check if class is correct
                            if prediction['main_object_class'].lower() == main_gt["category"].lower():
                                print(f"Correct detection: {prediction['main_object_class']}  and main_gt: {main_gt['category']}")
                                tp += 1
                                correct_cls += 1
                            else:
                                fp += 1  # Detection OK but wrong class
                                print(f"wrong detection: {prediction['main_object_class']}  and main_gt: {main_gt['category']}")
                            
                            total_cls += 1
                        else:
                            fp += 1  # Low IoU means wrong detection
                    else:
                        fp += 1  # Invalid box format
                else:
                    fn += 1  # No prediction but we have ground truth
            
            # Calculate metrics
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-5)
            
            mean_iou = np.mean(ious) if ious else 0.0
            cls_accuracy = correct_cls / max(total_cls, 1)
            
            # Store metrics
            metrics = {
                'detector': result['detector'],
                'detector_size': result['detector_size'],
                'classifier': result['classifier'],
                'classifier_size': result['classifier_size'],
                'end_to_end': result.get('end_to_end', False),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mean_iou': mean_iou,
                'cls_accuracy': cls_accuracy,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
            
            eval_metrics.append(metrics)
            
            # Update the original results with evaluation metrics
            all_results[combo_name].update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mean_iou': mean_iou,
                'cls_accuracy': cls_accuracy
            })
            
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"Mean IoU: {mean_iou:.4f}, Classification Accuracy: {cls_accuracy:.4f}")
        
        # Generate evaluation report
        self._generate_evaluation_report(eval_metrics, eval_dir)

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes
        
        Args:
            box1: First box in format [x1, y1, x2, y2]
            box2: Second box in format [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection area
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / max(union_area, 1e-5)
        
        return iou

    def _generate_evaluation_report(self, metrics: List[Dict], output_dir: str):
        """Generate evaluation report with visualizations"""
        # Create summary table
        table_data = []
        headers = ["Type", "Detector", "Size", "Classifier", "Size", "Precision", "Recall", 
                "F1", "IoU", "Cls Acc", "TP", "FP", "FN"]
        
        for item in metrics:
            # Determine system type
            system_type = "End-to-End" if item.get('end_to_end', False) else "Two-Stage"
            
            row = [
                system_type,
                item['detector'],
                item['detector_size'],
                item['classifier'],
                item['classifier_size'],
                f"{item['precision']:.4f}",
                f"{item['recall']:.4f}",
                f"{item['f1_score']:.4f}",
                f"{item['mean_iou']:.4f}",
                f"{item['cls_accuracy']:.4f}",
                item['true_positives'],
                item['false_positives'],
                item['false_negatives']
            ]
            table_data.append(row)
        
        # Sort by F1 score
        table_data.sort(key=lambda x: float(x[7]), reverse=True)
        
        # Generate text report
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write("# Vehicle Classification System Evaluation Results\n\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
            f.write("\n\n")
        
        # Plot evaluation metrics
        self._plot_evaluation_metrics(metrics, output_dir)
    
    def _plot_evaluation_metrics(self, metrics: List[Dict], output_dir: str):
            """Plot evaluation metrics for different model combinations"""
            # Extract data for plotting
            combos = []
            precisions = []
            recalls = []
            f1_scores = []
            ious = []
            cls_accs = []
            types = []  # End-to-End or Two-Stage
            
            for item in metrics:
                # Determine system type
                system_type = "End-to-End" if item.get('end_to_end', False) else "Two-Stage"
                types.append(system_type)
                
                # Create label for x-axis
                if system_type == "End-to-End":
                    combo = f"{item['detector']} {item['detector_size']}\n(End-to-End)"
                else:
                    combo = f"{item['detector']} {item['detector_size']}\n+ {item['classifier']} {item['classifier_size']}"
                
                combos.append(combo)
                precisions.append(item['precision'])
                recalls.append(item['recall'])
                f1_scores.append(item['f1_score'])
                ious.append(item['mean_iou'])
                cls_accs.append(item['cls_accuracy'])
            
            # Sort by F1 score
            sorted_indices = np.argsort(f1_scores)[::-1]  # Descending
            combos = [combos[i] for i in sorted_indices]
            precisions = [precisions[i] for i in sorted_indices]
            recalls = [recalls[i] for i in sorted_indices]
            f1_scores = [f1_scores[i] for i in sorted_indices]
            ious = [ious[i] for i in sorted_indices]
            cls_accs = [cls_accs[i] for i in sorted_indices]
            types = [types[i] for i in sorted_indices]
            
            # Create figure for precision, recall, F1
            plt.figure(figsize=(14, 8))
            
            x = np.arange(len(combos))
            width = 0.25
            
            # Use different marker styles for end-to-end and two-stage systems
            for i, (prec, rec, f1, t) in enumerate(zip(precisions, recalls, f1_scores, types)):
                if t == "End-to-End":
                    marker = 'o'
                    alpha = 0.8
                else:
                    marker = 's'
                    alpha = 0.6
                
                if i == 0:  # First element gets a label for the legend
                    plt.bar(x[i] - width, prec, width, label='Precision', color='skyblue', alpha=alpha)
                    plt.bar(x[i], rec, width, label='Recall', color='coral', alpha=alpha)
                    plt.bar(x[i] + width, f1, width, label='F1 Score', color='lightgreen', alpha=alpha)
                else:
                    plt.bar(x[i] - width, prec, width, color='skyblue', alpha=alpha)
                    plt.bar(x[i], rec, width, color='coral', alpha=alpha)
                    plt.bar(x[i] + width, f1, width, color='lightgreen', alpha=alpha)
            
            plt.ylabel('Score')
            plt.title('Precision, Recall, and F1 by Model Combination')
            plt.xticks(x, combos, rotation=45, ha='right')
            plt.legend()
            plt.ylim(0, 1.05)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, 'precision_recall_f1.png'), dpi=300)
            plt.close()
            
            # Create figure for IoU and Classification Accuracy
            plt.figure(figsize=(14, 6))
            
            width = 0.35
            for i, (iou, acc, t) in enumerate(zip(ious, cls_accs, types)):
                if t == "End-to-End":
                    alpha = 0.8
                    hatch = '\\'
                else:
                    alpha = 0.6
                    hatch = None
                
                if i == 0:  # First element gets a label for the legend
                    plt.bar(x[i] - width/2, iou, width, label='Mean IoU', color='purple', alpha=alpha, hatch=hatch)
                    plt.bar(x[i] + width/2, acc, width, label='Classification Accuracy', color='orange', alpha=alpha, hatch=hatch)
                else:
                    plt.bar(x[i] - width/2, iou, width, color='purple', alpha=alpha, hatch=hatch)
                    plt.bar(x[i] + width/2, acc, width, color='orange', alpha=alpha, hatch=hatch)
            
            plt.ylabel('Score')
            plt.title('IoU and Classification Accuracy by Model Combination')
            plt.xticks(x, combos, rotation=45, ha='right')
            plt.legend()
            plt.ylim(0, 1.05)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, 'iou_cls_accuracy.png'), dpi=300)
            plt.close()


    def main():
        """Example usage of the EnhancedBenchmark class"""
        import argparse
        import psutil
        import sys
        
        parser = argparse.ArgumentParser(description="Run enhanced vehicle classification benchmark")
        parser.add_argument("--images", type=str, required=True, help="Path to test images")
        parser.add_argument("--annotations", type=str, default=None, help="Path to COCO annotations file")
        parser.add_argument("--output", type=str, default="./benchmark_results", help="Output directory")
        parser.add_argument("--num-images", type=int, default=None, help="Number of images to process (None=all)")
        parser.add_argument("--save-vis", action="store_true", help="Save visualization images")
        parser.add_argument("--detectors", type=str, default="all", 
                        help="Comma-separated list of detectors to evaluate (all, yolo, retinanet, ssd)")
        parser.add_argument("--classifiers", type=str, default="all", 
                        help="Comma-separated list of classifiers to evaluate (all, clip, dino, vit)")
        parser.add_argument("--end-to-end", type=str, default="none", 
                        help="End-to-end systems to evaluate (none, yolo_world, all)")
        parser.add_argument("--model-sizes", type=str, default="all",
                        help="Comma-separated list of model sizes to use (all, small, medium, large)")
        parser.add_argument("--cpu-only", action="store_true", help="Use CPU-only optimized models")
        parser.add_argument("--track-memory", action="store_true", help="Track memory usage during benchmarking")
        
        args = parser.parse_args()
        
        # You need to import your own modules here
        try:
            # Import detector implementations
            from models.detectors.yolo_detector import YOLOv12Detector
            from models.detectors.supervision_detector import SupervisionDetector
            from models.detectors.ssd_detector import SSDDetector
            # from models.detectors.retinanet_detector import RetinaNetDetector
            from models.detectors.rtdetr_detector import RTDETRDetector
            
            # Import classifier implementations
            from models.classifiers.clip_classifier import CLIPClassifier
            from models.classifiers.vit_classifier import ViTClassifier
            from models.classifiers.owlvit_classifier import OwlViTClassifier
            
            # Import end-to-end systems
            from models.yolo_world_wrapper import YOLOWorldWrapper
            
            # Import other components
            from models.refiners.heuristic_refiner import HeuristicRefiner
            from core.prioritizer import DefaultPrioritizer
            from core.visualizer import DefaultVisualizer
        except ImportError as e:
            print(f"Error importing modules: {e}")
            print("Make sure all required modules are installed and in the Python path.")
            sys.exit(1)
        
        # Initialize components
        refiner = HeuristicRefiner()
        prioritizer = DefaultPrioritizer()
        visualizer = DefaultVisualizer()
        
        # Initialize benchmark
        benchmark = EnhancedBenchmark(
            output_dir=args.output,
            annotations_path=args.annotations,
            refiner=refiner,
            prioritizer=prioritizer,
            visualizer=visualizer
        )
        
        # Prepare detectors
        detectors = []
        model_sizes = args.model_sizes.split(",") if args.model_sizes != "all" else ["small", "medium", "large"]
        
        if args.cpu_only:
            # For CPU-only mode, only use small models
            model_sizes = ["small"]
        
        if args.detectors == "all" or "yolo" in args.detectors.split(","):
            for size in model_sizes:
                if size == "large" and args.cpu_only:
                    continue  # Skip large models in CPU-only mode
                detectors.append(YOLOv12Detector(model_size=size))
        
        # if args.detectors == "all" or "retinanet" in args.detectors.split(","):
        #     for size in model_sizes:
        #         if size == "large" and args.cpu_only:
        #             continue
        #         detectors.append(RetinaNetDetector(model_size=size))
        
        if args.detectors == "all" or "ssd" in args.detectors.split(","):
            for size in model_sizes:
                if size == "large" and args.cpu_only:
                    continue
                detectors.append(SSDDetector(model_size=size))
                
        if args.detectors == "all" or "rt" in args.detectors.split(","):
            for size in model_sizes:
                if size == "large" and args.cpu_only:
                    continue  # Skip large models in CPU-only mode
                detectors.append(RTDETRDetector(model_size=size))
                    
        if args.detectors == "all" or "supervision" in args.detectors.split(","):
            for size in model_sizes:
                if size == "large" and args.cpu_only:
                    continue
                detectors.append(SupervisionDetector(model_size=size))
        
        # Prepare classifiers
        classifiers = []
        if args.classifiers == "all" or "clip" in args.classifiers.split(","):
            for size in model_sizes:
                if size == "large" and args.cpu_only:
                    continue
                classifiers.append(CLIPClassifier(model_size=size))

        if args.classifiers == "all" or "vit" in args.classifiers.split(","):
            for size in model_sizes:
                if size == "large" and args.cpu_only:
                    continue
                classifiers.append(ViTClassifier(model_size=size))
        
        if args.classifiers == "all" or "owlvit" in args.classifiers.split(","):
            for size in model_sizes:
                if size == "large" and args.cpu_only:
                    continue
                classifiers.append(OwlViTClassifier(model_size=size))
        
        # Prepare end-to-end systems
        end_to_end_systems = []
        if args.end_to_end == "all" or "yolo_world" in args.end_to_end.split(","):
            for size in model_sizes:
                if size == "large" and args.cpu_only:
                    continue
                end_to_end_systems.append(YOLOWorldWrapper(model_size=size))
        
        # Check if we have valid models to evaluate
        if not detectors and not end_to_end_systems:
            print("Error: No detectors or end-to-end systems selected for evaluation")
            return
        
        if not classifiers and not end_to_end_systems:
            print("Error: No classifiers or end-to-end systems selected for evaluation")
            return
        
        # Print benchmark configuration
        print(f"\n{'='*50}")
        print(f"Running benchmark with the following configuration:")
        print(f"{'='*50}")
        print(f"Image directory: {args.images}")
        print(f"Annotation file: {args.annotations if args.annotations else 'None (performance only)'}")
        print(f"Number of images: {args.num_images if args.num_images else 'All available'}")
        print(f"Save visualizations: {'Yes' if args.save_vis else 'No'}")
        print(f"CPU-only optimized: {'Yes' if args.cpu_only else 'No'}")
        print(f"Track memory usage: {'Yes' if args.track_memory else 'No'}")
        
        if detectors:
            print(f"\nDetectors:")
            for detector in detectors:
                print(f"  - {detector.name} ({detector.model_size})")
        
        if classifiers:
            print(f"\nClassifiers:")
            for classifier in classifiers:
                print(f"  - {classifier.name} ({classifier.model_size})")
        
        if end_to_end_systems:
            print(f"\nEnd-to-End Systems:")
            for system in end_to_end_systems:
                print(f"  - {system.name} ({system.model_size})")
        
        print(f"{'='*50}\n")
        
        # Set up memory tracker if requested
        if args.track_memory:
            memory_usage = {'baseline': psutil.Process().memory_info().rss / 1024 / 1024}  # MB
            print(f"Baseline memory usage: {memory_usage['baseline']:.2f} MB")
        
        # Run benchmark
        try:
            results = benchmark.run_benchmark(
                detectors=detectors,
                classifiers=classifiers,
                end_to_end_systems=end_to_end_systems,
                input_dir=args.images,
                num_images=args.num_images,
                save_visualizations=args.save_vis,
                run_name=f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Print top results sorted by performance if we have ground truth
            if args.annotations:
                print("\nTop model combinations by F1 score:")
                
                top_combos = []
                for combo_name, result in results.items():
                    # Check if this result has evaluation metrics
                    if 'precision' in result and 'recall' in result and 'f1_score' in result:
                        system_type = "End-to-End" if result.get('end_to_end', False) else "Two-Stage"
                        if system_type == "End-to-End":
                            name = f"{result['detector']} {result['detector_size']} (End-to-End)"
                        else:
                            name = f"{result['detector']} {result['detector_size']} + {result['classifier']} {result['classifier_size']}"
                            
                        top_combos.append({
                            'name': name,
                            'type': system_type,
                            'f1': result['f1_score'],
                            'precision': result['precision'],
                            'recall': result['recall'],
                            'mean_iou': result['mean_iou'],
                            'cls_accuracy': result['cls_accuracy'],
                            'time': result['avg_time_per_image']
                        })
                
                # Sort by F1 score
                top_combos.sort(key=lambda x: x['f1'], reverse=True)
                
                # Print top 3 or all if fewer
                for i, combo in enumerate(top_combos[:min(3, len(top_combos))]):
                    print(f"{i+1}. {combo['name']} ({combo['type']})")
                    print(f"   F1 Score: {combo['f1']:.4f}")
                    print(f"   Precision: {combo['precision']:.4f}, Recall: {combo['recall']:.4f}")
                    print(f"   Mean IoU: {combo['mean_iou']:.4f}, Classification Accuracy: {combo['cls_accuracy']:.4f}")
                    print(f"   Average processing time: {combo['time']:.4f}s")
                    print()
            
            # Otherwise, print top results sorted by speed
            else:
                print("\nTop model combinations by processing speed:")
                
                speed_combos = []
                for combo_name, result in results.items():
                    system_type = "End-to-End" if result.get('end_to_end', False) else "Two-Stage"
                    if system_type == "End-to-End":
                        name = f"{result['detector']} {result['detector_size']} (End-to-End)"
                    else:
                        name = f"{result['detector']} {result['detector_size']} + {result['classifier']} {result['classifier_size']}"
                        
                    speed_combos.append({
                        'name': name,
                        'type': system_type,
                        'time': result['avg_time_per_image'],
                        'confidence': result['avg_confidence']
                    })
                
                # Sort by processing time
                speed_combos.sort(key=lambda x: x['time'])
                
                # Print top 3 or all if fewer
                for i, combo in enumerate(speed_combos[:min(3, len(speed_combos))]):
                    print(f"{i+1}. {combo['name']} ({combo['type']})")
                    print(f"   Average processing time: {combo['time']:.4f}s")
                    print(f"   Average confidence: {combo['confidence']:.4f}")
                    print()
            
            # Report final memory usage if requested
            if args.track_memory:
                current_mem = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"Final memory usage: {current_mem:.2f} MB")
                print(f"Memory increase during benchmarking: {current_mem - memory_usage['baseline']:.2f} MB")
            
            print(f"\nBenchmark complete. Results saved to {args.output}")
            
        except Exception as e:
            import traceback
            print(f"Error during benchmarking: {e}")
            traceback.print_exc()


    if __name__ == "__main__":
        main()