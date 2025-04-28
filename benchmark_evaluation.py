"""
benchmark_evaluator.py - Extends benchmark.py to evaluate model performance against annotated ground truth
"""

import os
import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import cv2
from tqdm import tqdm

from core.interfaces import (
    ObjectDetector, 
    Classifier, 
    ClassificationRefiner,
    ObjectPrioritizer, 
    Visualizer,
    Detection,
    ClassificationResult
)
from pipeline.vehicle_classifier import VehicleClassifier
from benchmark import ModelBenchmark


class AnnotationEvaluator(ModelBenchmark):
    """Extends ModelBenchmark to validate against ground truth annotations"""
    
    def __init__(self, 
                ground_truth_path: str,
                annotation_format: str = "coco",
                **kwargs):
        """
        Initialize the evaluator
        
        Args:
            ground_truth_path: Path to the ground truth annotations
            annotation_format: Format of annotations (coco, yolo)
            **kwargs: Additional arguments passed to ModelBenchmark
        """
        super().__init__(**kwargs)
        self.ground_truth_path = ground_truth_path
        self.annotation_format = annotation_format
        self.ground_truth = self._load_ground_truth()
        
    def _load_ground_truth(self) -> Dict:
        """Load ground truth annotations based on format"""
        if self.annotation_format.lower() == "coco":
            return self._load_coco_annotations()
        elif self.annotation_format.lower() == "yolo":
            return self._load_yolo_annotations()
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
    
    def _load_coco_annotations(self) -> Dict:
        """Load COCO format annotations"""
        with open(os.path.join(self.ground_truth_path, "annotations.json"), "r") as f:
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
            
            # MODIFIED: No longer checking for is_main attribute
            # Since you're only annotating the main vehicle, we assume all 
            # annotations are main vehicles
            
            # Add the annotation
            gt_data[image_file].append({
                "bbox": annotation["bbox"],  # [x, y, width, height] in COCO
                "category": category_map[annotation["category_id"]]
            })
        
        return gt_data
    
    def _load_yolo_annotations(self) -> Dict:
        """Load YOLO format annotations"""
        gt_data = {}
        
        # Read class names
        classes_file = os.path.join(self.ground_truth_path, "obj.names")
        if not os.path.exists(classes_file):
            classes_file = os.path.join(self.ground_truth_path, "classes.txt")
        
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Process annotation files
        for root, _, files in os.walk(self.ground_truth_path):
            for file in files:
                if file.endswith(".txt") and not file == "obj.names" and not file == "classes.txt":
                    # Get corresponding image file
                    image_file = None
                    for ext in [".jpg", ".jpeg", ".png"]:
                        img_path = file.replace(".txt", ext)
                        if os.path.exists(os.path.join(root, img_path)):
                            image_file = img_path
                            break
                    
                    if not image_file:
                        # Default to jpg if not found
                        image_file = file.replace(".txt", ".jpg")
                    
                    with open(os.path.join(root, file), "r") as f:
                        annotations = []
                        for line in f.readlines():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # MODIFIED: No longer checking for is_main attribute
                                # Assume all annotations are main vehicles
                                
                                # Convert YOLO format to [x, y, width, height]
                                # Note: YOLO uses normalized coordinates
                                annotations.append({
                                    "bbox": [x_center, y_center, width, height],
                                    "category": classes[class_id],
                                    "format": "yolo"  # Flag for conversion later
                                })
                        
                        if annotations:
                            gt_data[image_file] = annotations
        
        return gt_data
    
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
            # Calculate area
            area = ann["bbox"][2] * ann["bbox"][3]
            
            if area > max_area:
                max_area = area
                main_annotation = ann
                
        return main_annotation
    
    def run_evaluation(self,
                      detectors: List[ObjectDetector],
                      classifiers: List[Classifier],
                      input_dir: str,
                      **kwargs):
        """
        Run benchmark and evaluate against ground truth
        
        Args:
            detectors: List of detector implementations to test
            classifiers: List of classifier implementations to test
            input_dir: Directory containing input images
            **kwargs: Additional arguments passed to run_benchmark
            
        Returns:
            Dictionary of evaluation results
        """
        # Run standard benchmark
        benchmark_results = self.run_benchmark(
            detectors=detectors,
            classifiers=classifiers,
            input_dir=input_dir,
            **kwargs
        )
        
        # Extend with evaluation against ground truth
        evaluation_results = {}
        
        for combo_name, result in benchmark_results.items():
            print(f"\n===== Evaluating {combo_name} against ground truth =====")
            
            # Load prediction results from CSV
            predictions = self._load_predictions(result['csv_path'])
            
            # Compare with ground truth
            metrics = self._evaluate_predictions(predictions)
            
            # Store evaluation metrics
            evaluation_results[combo_name] = {
                **result,  # Include original benchmark results
                "evaluation": metrics
            }
            
            # Print metrics summary
            print(f"Mean IoU: {metrics['mean_iou']:.4f}")
            print(f"Detection Accuracy: {metrics['detection_accuracy']:.4f}")
            print(f"Classification Accuracy: {metrics['classification_accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        # Generate evaluation report and visualizations
        self._generate_evaluation_report(evaluation_results)
        
        return evaluation_results
    
    def _load_predictions(self, csv_path: str) -> Dict:
        """Load prediction results from CSV output"""
        predictions = {}
        
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_file = os.path.basename(row['image_name'])
                
                # Parse bounding box
                if 'main_box' in row and row['main_box']:
                    try:
                        # Format might be "[x1, y1, x2, y2]" or "x1,y1,x2,y2"
                        bbox_str = row['main_box'].replace('[', '').replace(']', '')
                        bbox = [float(x) for x in bbox_str.split(',')]
                        
                        # Convert [x1, y1, x2, y2] to [x, y, width, height]
                        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    except Exception as e:
                        print(f"Error parsing bbox for {image_file}: {e}")
                        bbox = None
                else:
                    bbox = None
                
                predictions[image_file] = {
                    'bbox': bbox,
                    'class': row.get('main_object_class', None),
                    'confidence': float(row.get('main_object_confidence', 0))
                }
        
        return predictions
    
    def _evaluate_predictions(self, predictions: Dict) -> Dict:
        """Evaluate predictions against ground truth"""
        # Metrics to calculate
        total_images = 0
        correct_classifications = 0
        
        # For confusion matrix
        all_classes = set()
        for data in self.ground_truth.values():
            for ann in data:
                all_classes.add(ann['category'])
                
        for pred in predictions.values():
            if pred['class']:
                all_classes.add(pred['class'])
        
        all_classes = sorted(list(all_classes))
        y_true = []
        y_pred = []
        
        # Evaluate each image
        for image_file, gt_annotations in self.ground_truth.items():
            total_images += 1
            
            if image_file not in predictions:
                # No prediction for this image
                main_gt = self._select_main_vehicle(gt_annotations)
                if main_gt:
                    y_true.append(main_gt['category'])
                    y_pred.append("unknown")
                continue
            
            prediction = predictions[image_file]
            
            if not gt_annotations:
                # No ground truth annotations
                continue
                
            # Use _select_main_vehicle to pick the most likely main vehicle
            main_gt = self._select_main_vehicle(gt_annotations)
            if not main_gt:
                continue
            
            # Since we don't have bounding box information in the CSV,
            # we'll focus on classification accuracy only
            if main_gt['category'].lower() == prediction['class'].lower():
                correct_classifications += 1
                    
            # For confusion matrix
            y_true.append(main_gt['category'])
            y_pred.append(prediction['class'] if prediction['class'] else "unknown")
        
        # Calculate metrics
        classification_accuracy = correct_classifications / max(total_images, 1)
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_classes + ["unknown"])
        
        # Since we don't have bounding boxes, we'll set detection metrics to N/A
        # but we'll keep the structure similar for compatibility
        return {
            'mean_iou': 0.0,  # We can't calculate IoU without bounding boxes
            'detection_accuracy': 0.0,  # We can't measure detection accuracy
            'classification_accuracy': classification_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classes': all_classes + ["unknown"]
        }
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate IoU between two boxes
        box format: [x, y, width, height]
        """
        # Convert to [x1, y1, x2, y2]
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Intersection coordinates
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union = box1_area + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / max(union, 1e-5)
        return iou
    
    def _generate_evaluation_report(self, results: Dict):
        """Generate detailed evaluation report with visualizations"""
        # Create output directory for evaluation results
        eval_dir = os.path.join(self.output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Prepare data for metrics comparison
        combos = []
        classification_acc = []
        f1_scores = []
        
        for combo_name, result in results.items():
            metrics = result['evaluation']
            combos.append(f"{result['detector']} {result['detector_size']}\n+ {result['classifier']} {result['classifier_size']}")
            classification_acc.append(metrics['classification_accuracy'])
            f1_scores.append(metrics['f1_score'])
        
        # Sort by F1 score
        sorted_indices = np.argsort(f1_scores)[::-1]  # Descending
        combos = [combos[i] for i in sorted_indices]
        classification_acc = [classification_acc[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]
        
        # Plot metrics comparison (modified to show only classification metrics)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Classification accuracy
        axes[0].bar(combos, classification_acc, color='lightgreen')
        axes[0].set_title('Classification Accuracy')
        axes[0].set_ylim(0, 1.0)
        axes[0].tick_params(axis='x', rotation=45)
        
        # F1 Score
        axes[1].bar(combos, f1_scores, color='purple')
        axes[1].set_title('F1 Score')
        axes[1].set_ylim(0, 1.0)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, 'metrics_comparison.png'), dpi=300)
        plt.close()
        
        # Generate confusion matrices for each model
        for combo_name, result in results.items():
            metrics = result['evaluation']
            cm = metrics['confusion_matrix']
            classes = metrics['classes']
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - {result['detector']} + {result['classifier']}")
            plt.colorbar()
            
            # Add class labels
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45, ha='right')
            plt.yticks(tick_marks, classes)
            
            # Add values to cells
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j],
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True Class')
            plt.xlabel('Predicted Class')
            plt.savefig(os.path.join(eval_dir, f'confusion_matrix_{combo_name}.png'), dpi=300)
            plt.close()
        
        # Generate summary table
        table_data = []
        headers = ["Detector", "Size", "Classifier", "Size", "Class. Acc", "F1"]
        
        for combo_name, result in results.items():
            metrics = result['evaluation']
            row = [
                result['detector'],
                result['detector_size'],
                result['classifier'],
                result['classifier_size'],
                f"{metrics['classification_accuracy']:.4f}",
                f"{metrics['f1_score']:.4f}"
            ]
            table_data.append(row)
        
        # Sort by F1 score
        table_data.sort(key=lambda x: float(x[5]), reverse=True)
        
        # Generate text report
        report_path = os.path.join(eval_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write("# Vehicle Classification System Evaluation Results\n\n")
            f.write("Note: This evaluation focuses on classification accuracy only, as bounding box information was not available in the prediction CSV.\n\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
            f.write("\n\n")
            
            # Add per-class performance metrics
            f.write("## Per-Class Performance\n\n")
            for combo_name, result in results.items():
                f.write(f"### {result['detector']} {result['detector_size']} + {result['classifier']} {result['classifier_size']}\n\n")
                
                # Extract class distribution
                classes = result['evaluation']['classes']
                cm = result['evaluation']['confusion_matrix']
                
                # Calculate per-class metrics
                class_metrics = []
                for i, cls in enumerate(classes[:-1]):  # Skip "unknown" class
                    true_pos = cm[i, i]
                    false_pos = sum(cm[:, i]) - true_pos
                    false_neg = sum(cm[i, :]) - true_pos
                    
                    precision = true_pos / max(true_pos + false_pos, 1)
                    recall = true_pos / max(true_pos + false_neg, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-5)
                    
                    class_metrics.append([
                        cls,
                        f"{precision:.4f}",
                        f"{recall:.4f}",
                        f"{f1:.4f}"
                    ])
                
                f.write(tabulate(class_metrics, headers=["Class", "Precision", "Recall", "F1"], tablefmt="grid"))
                f.write("\n\n")