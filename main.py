"""
main.py - Main entry point for the vehicle classification system (updated)
"""

import os
import argparse
import cv2
import torch
from typing import Dict, List, Tuple
import warnings

# Import components
from models.detectors.yolo_detector import YOLODetector
from models.detectors.supervision_detector import SupervisionDetector
from models.detectors.ssd_detector import SSDDetector
from models.detectors.retinanet_detector import RetinaNetDetector
from models.detectors.dino_detector import DINODetector
from models.detectors.rtdetr_detector import RTDETRDetector

from models.classifiers.clip_classifier import CLIPClassifier
from models.classifiers.owlvit_classifier import DINOClassifier
from models.classifiers.vit_classifier import ViTClassifier
from models.classifiers.convnext_classifier import ConvNeXtClassifier

from models.refiners.heuristic_refiner import HeuristicRefiner

from core.prioritizer import DefaultPrioritizer
from core.visualizer import DefaultVisualizer, DebugVisualizer

from pipeline.vehicle_classifier import VehicleClassifier
from benchmark import ModelBenchmark


def create_pipeline(detector_name, detector_size, classifier_name, classifier_size, debug_vis=False):
    """
    Create a vehicle classification pipeline with the specified components
    
    Args:
        detector_name: Name of the detector to use
        detector_size: Size of the detector model
        classifier_name: Name of the classifier to use
        classifier_size: Size of the classifier model
        debug_vis: Whether to use the debug visualizer
        
    Returns:
        VehicleClassifier instance
    """
    # Initialize detector
    if detector_name.lower() == "yolo":
        detector = YOLODetector(model_size=detector_size)
    elif detector_name.lower() == "supervision":
        detector = SupervisionDetector(model_size=detector_size)
    elif detector_name.lower() == "ssd":
        detector = SSDDetector(model_size=detector_size)
    elif detector_name.lower() == "retinanet":
        detector = RetinaNetDetector(model_size=detector_size)
    elif detector_name.lower() == "dino":
        detector = DINODetector(model_size=detector_size)
    elif detector_name.lower() == "rt":
        detector = RTDETRDetector(model_size=detector_size)
    else:
        print(f"Unknown detector: {detector_name}, falling back to YOLOv8")
        detector = YOLODetector(model_size="medium")
    
    # Initialize classifier
    if classifier_name.lower() == "clip":
        classifier = CLIPClassifier(model_size=classifier_size)
    elif classifier_name.lower() == "convnext":
        classifier = ConvNeXtClassifier(model_size=classifier_size)
    elif classifier_name.lower() == "vit":
        classifier = ViTClassifier(model_size=classifier_size)
    else:
        print(f"Unknown classifier: {classifier_name}, falling back to CLIP")
        classifier = CLIPClassifier(model_size="medium")
    
    # Initialize other components
    refiner = HeuristicRefiner()
    prioritizer = DefaultPrioritizer()
    
    # Choose visualizer based on debug flag
    if debug_vis:
        visualizer = DebugVisualizer()
    else:
        visualizer = DefaultVisualizer()
    
    # Create and return the pipeline
    return VehicleClassifier(
        detector=detector,
        classifier=classifier,
        refiner=refiner,
        prioritizer=prioritizer,
        visualizer=visualizer
    )


def run_single_image(pipeline, image_path, output_path=None):
    """Process a single image and display/save the result"""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Process the image
    result_image, main_class, main_conf, class_names, confidences, detections = pipeline.process_image(image)
    
    # Print results
    print(f"Main detected object: {main_class} (Confidence: {main_conf:.2f})")
    print(f"All detected objects:")
    for cls, conf in zip(class_names, confidences):
        print(f"  - {cls}: {conf:.2f}")
    
    # Save or display the result
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to {output_path}")
    else:
        # Display the result
        cv2.imshow("Vehicle Classification Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_batch_process(pipeline, input_dir, output_dir, num_images=0):
    """Run batch processing on a directory of images"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run batch processing
    results, csv_path = pipeline.batch_process(
        input_dir=input_dir,
        output_dir=output_dir,
        num_images=num_images,
        save_visualizations=True
    )
    
    print(f"\nBatch processing complete. Results saved to {csv_path}")
    return results, csv_path


def run_benchmark(input_dir, output_dir, num_images=20):
    """Run benchmark on available model combinations"""
    # Create components for benchmarking
    refiner = HeuristicRefiner()
    prioritizer = DefaultPrioritizer()
    visualizer = DefaultVisualizer()
    
    # Initialize benchmark
    benchmark = ModelBenchmark(
        output_dir=output_dir,
        refiner=refiner,
        prioritizer=prioritizer,
        visualizer=visualizer
    )
    
    # Create detectors to benchmark (limit to more stable options)
    detectors = [
        YOLODetector(model_size="small"),
        YOLODetector(model_size="medium"),
        SupervisionDetector(model_size="medium"),
    ]
    
    
    # Create classifiers to benchmark
    classifiers = [
        CLIPClassifier(model_size="small"),
        CLIPClassifier(model_size="medium"),
        ViTClassifier(model_size="small"),
        ViTClassifier(model_size="medium"),
        DINOClassifier(model_size="small"),
        DINOClassifier(model_size="medium"),
    ]
    

    # Run the benchmark
    results = benchmark.run_benchmark(
        detectors=detectors,
        classifiers=classifiers,
        input_dir=input_dir,
        num_images=num_images,
        save_visualizations=True
    )
    
    print(f"\nBenchmark complete. Results saved to {output_dir}")
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vehicle Classification System")
    
    # Mode selection
    parser.add_argument("--mode", choices=["single", "batch", "benchmark"], default="single",
                      help="Operation mode: single image, batch processing, or benchmark")
    
    # Common arguments
    parser.add_argument("--input", required=True, 
                      help="Input image path (single mode) or directory (batch/benchmark mode)")
    parser.add_argument("--output", default="./output",
                      help="Output directory or file path")
    
    # Model selection (for single and batch modes)
    parser.add_argument("--detector", choices=["yolo", "supervision", "ssd", "retinanet", "dino", "rt"], default="yolo",
                      help="Detector to use")
    parser.add_argument("--detector-size", choices=["small", "medium", "large"], default="medium",
                      help="Size of the detector model")
    parser.add_argument("--classifier", choices=["clip", "convnext", "vit"], default="clip",
                      help="Classifier to use")
    parser.add_argument("--classifier-size", choices=["small", "medium", "large"], default="medium",
                      help="Size of the classifier model")
    
    # Batch processing options
    parser.add_argument("--num-images", type=int, default=0,
                      help="Number of images to process (0 for all)")
    
    # Debug visualization
    parser.add_argument("--debug-vis", action="store_true",
                      help="Use debug visualizer with additional information")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check for MPS (Metal Performance Shaders) on macOS
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available on this macOS device")
    
    if args.mode == "single":
        # Single image mode
        pipeline = create_pipeline(
            args.detector, args.detector_size, 
            args.classifier, args.classifier_size,
            args.debug_vis
        )
        
        # If output is a directory, create filename for the result
        if os.path.isdir(args.output) or not args.output.lower().endswith(('.jpg', '.jpeg', '.png')):
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.basename(args.input)
            filename, ext = os.path.splitext(base_name)
            output_path = os.path.join(args.output, f"{filename}_result{ext}")
        else:
            output_path = args.output
        
        run_single_image(pipeline, args.input, output_path)
        
    elif args.mode == "batch":
        # Batch processing mode
        pipeline = create_pipeline(
            args.detector, args.detector_size, 
            args.classifier, args.classifier_size,
            args.debug_vis
        )
        
        run_batch_process(pipeline, args.input, args.output, args.num_images)
        
    elif args.mode == "benchmark":
        # Benchmark mode
        run_benchmark(args.input, args.output, args.num_images)


if __name__ == "__main__":
    # Filter out expected warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    
    main()