#!/usr/bin/env python3
"""
run_benchmark.py - Script to run the enhanced benchmark system for vehicle classification

This script uses the EnhancedBenchmark class to evaluate combinations of object
detectors and classifiers, as well as end-to-end systems like YOLO-World,
comparing them against ground truth annotations if available.
"""

import os
import sys
import argparse
import time
import json
from typing import Dict, List, Tuple, Optional

# Import the enhanced benchmark
from enhanced_benchmark import EnhancedBenchmark
from memory_tracker import MemoryTracker

# Import interfaces
from core.interfaces import (
    ObjectDetector, 
    Classifier, 
    ClassificationRefiner,
    ObjectPrioritizer, 
    Visualizer
)

def setup_argparse():
    """Set up argument parser for the benchmark script"""
    parser = argparse.ArgumentParser(description="Run vehicle classification benchmark")
    
    # Required arguments
    parser.add_argument("--images", type=str, required=True, 
                      help="Path to directory containing test images")
    
    # Optional arguments with defaults
    parser.add_argument("--annotations", type=str, default=None,
                      help="Path to COCO annotations file for evaluation")
    parser.add_argument("--output", type=str, default="./benchmark_results",
                      help="Output directory for benchmark results")
    parser.add_argument("--num-images", type=int, default=None,
                      help="Number of images to process (None for all)")
    parser.add_argument("--batch-size", type=int, default=1,
                      help="Batch size for processing (1 for single image)")
    parser.add_argument("--save-vis", action="store_true",
                      help="Save visualization images with annotations")
    
    # Model selection options
    parser.add_argument("--detectors", type=str, default="all",
                      help="Comma-separated list of detectors to evaluate (all, yolo, retinanet, ssd)")
    parser.add_argument("--classifiers", type=str, default="all",
                      help="Comma-separated list of classifiers to evaluate (all, clip, dino, vit)")
    parser.add_argument("--end-to-end", type=str, default="none",
                      help="End-to-end systems to evaluate (none, yolo_world, all)")
    parser.add_argument("--model-sizes", type=str, default="all",
                      help="Comma-separated list of model sizes to use (all, small, medium, large)")
    
    # System options
    parser.add_argument("--cpu-only", action="store_true",
                      help="Use CPU-only models (skips large models)")
    parser.add_argument("--gpu-id", type=int, default=0,
                      help="GPU ID to use if available")
    parser.add_argument("--track-memory", action="store_true",
                      help="Track and report memory usage during benchmarking")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    return parser.parse_args()

def get_model_classes(args):
    """Import and return detector, classifier, and end-to-end system classes based on arguments"""
    # First attempt to import all model implementations
    try:
        # Import detector implementations
        from models.detectors.yolo_detector import YOLOv12Detector
        from models.detectors.ssd_detector import SSDDetector
        from models.detectors.supervision_detector import SupervisionDetector
        from models.detectors.retinanet_detector import RetinaNetDetector
        from models.detectors.rtdetr_detector import RTDETRDetector
        
        # Import classifier implementations
        from models.classifiers.clip_classifier import CLIPClassifier
        from models.classifiers.vit_classifier import ViTClassifier
        from models.classifiers.owlvit_classifier import OwlViTClassifier
        
        # Import end-to-end systems
        from models.yolo_world_wrapper import YOLOWorldWrapper
        
        # Import auxiliary components
        from models.refiners.heuristic_refiner import HeuristicRefiner
        from core.prioritizer import DefaultPrioritizer
        from core.visualizer import DefaultVisualizer
    except ImportError as e:
        print(f"Error importing model classes: {e}")
        print("Make sure all required modules are installed and in the Python path.")
        sys.exit(1)
    
    # Return the model classes
    detector_classes = {
        "yolo": YOLOv12Detector,
        "ssd": SSDDetector,
        # "retinanet": RetinaNetDetector,
        "supervision" : SupervisionDetector,
        "rtdetr": RTDETRDetector,
    }
    
    classifier_classes = {
        "clip": CLIPClassifier,
        "vit": ViTClassifier,
        "owlvit": OwlViTClassifier,
    }
    
    end_to_end_classes = {
        "yolo_world": YOLOWorldWrapper,
    }
    
    auxiliary_classes = {
        "refiner": HeuristicRefiner,
        "prioritizer": DefaultPrioritizer,
        "visualizer": DefaultVisualizer
    }
    
    return detector_classes, classifier_classes, end_to_end_classes, auxiliary_classes

def instantiate_models(args, detector_classes, classifier_classes, end_to_end_classes, auxiliary_classes):
    """Instantiate detector, classifier, and end-to-end models based on arguments"""
    # Parse model sizes to use
    model_sizes = args.model_sizes.split(",") if args.model_sizes != "all" else ["small", "medium", "large"]
    
    if args.cpu_only:
        # For CPU-only mode, only use small models
        model_sizes = ["small"]
    
    # Initialize detectors
    detectors = []
    detector_types = args.detectors.split(",") if args.detectors != "all" else detector_classes.keys()
    
    for det_type in detector_types:
        if det_type in detector_classes:
            for size in model_sizes:
                if size == "large" and args.cpu_only:
                    continue  # Skip large models in CPU-only mode
                try:
                    detector = detector_classes[det_type](model_size=size)
                    detectors.append(detector)
                    print(f"Initialized {det_type} detector with size {size}")
                except Exception as e:
                    print(f"Error initializing {det_type} detector (size: {size}): {e}")
    
    # Initialize classifiers
    classifiers = []
    classifier_types = args.classifiers.split(",") if args.classifiers != "all" else classifier_classes.keys()
    
    for cls_type in classifier_types:
        if cls_type in classifier_classes:
            for size in model_sizes:
                if size == "large" and args.cpu_only:
                    continue
                try:
                    classifier = classifier_classes[cls_type](model_size=size)
                    classifiers.append(classifier)
                    print(f"Initialized {cls_type} classifier with size {size}")
                except Exception as e:
                    print(f"Error initializing {cls_type} classifier (size: {size}): {e}")
    
    # Initialize end-to-end systems
    end_to_end_systems = []
    end_to_end_types = args.end_to_end.split(",") if args.end_to_end != "all" and args.end_to_end != "none" else end_to_end_classes.keys() if args.end_to_end != "none" else []
    
    for system_type in end_to_end_types:
        if system_type in end_to_end_classes:
            for size in model_sizes:
                if size == "large" and args.cpu_only:
                    continue
                try:
                    system = end_to_end_classes[system_type](model_size=size)
                    end_to_end_systems.append(system)
                    print(f"Initialized {system_type} end-to-end system with size {size}")
                except Exception as e:
                    print(f"Error initializing {system_type} end-to-end system (size: {size}): {e}")
    
    # Initialize auxiliary components
    refiner = auxiliary_classes["refiner"]()
    prioritizer = auxiliary_classes["prioritizer"]()
    visualizer = auxiliary_classes["visualizer"]()
    
    return detectors, classifiers, end_to_end_systems, refiner, prioritizer, visualizer

def setup_environment(args):
    """Set up environment variables and system configuration"""
    import numpy as np
    import random
    import torch
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
        torch.backends.mps.enabled = True
    
    # GPU setup if available
    if torch.cuda.is_available() and not args.cpu_only:
        print(f"Setting GPU device to: {args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        use_gpu = True
    else:
        if not args.cpu_only:
            print("GPU not available, falling back to CPU")
            # use mpls if available
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("Using MPS (Metal Performance Shaders) for GPU acceleration")
                torch.backends.mps.enabled = True
            else:
                print("No GPU available, using CPU") 
        use_gpu = False
    
    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
    
    return use_gpu

def main():
    """Main function to run the benchmark"""
    # Parse command line arguments
    args = setup_argparse()
    
    # Setup system environment
    use_gpu = setup_environment(args)
    
    # Get model classes
    detector_classes, classifier_classes, end_to_end_classes, auxiliary_classes = get_model_classes(args)
    
    # Instantiate models
    detectors, classifiers, end_to_end_systems, refiner, prioritizer, visualizer = instantiate_models(
        args, detector_classes, classifier_classes, end_to_end_classes, auxiliary_classes
    )
    
    # Check if we have valid models to evaluate
    if not detectors and not end_to_end_systems:
        print("Error: No detectors or end-to-end systems selected for evaluation")
        return
    
    if not classifiers and not end_to_end_systems:
        print("Error: No classifiers or end-to-end systems selected for evaluation")
        return
    
    # Initialize memory tracker if requested
    memory_tracker = None
    if args.track_memory:
        memory_tracker = MemoryTracker(
            output_dir=os.path.join(args.output, "memory_stats")
        )
        print("Memory tracking enabled")
        memory_tracker.start_tracking("start")
        memory_tracker.stop_tracking()
    
    # Print benchmark configuration
    print(f"\n{'='*50}")
    print(f"Running benchmark with the following configuration:")
    print(f"{'='*50}")
    print(f"Image directory: {args.images}")
    print(f"Annotation file: {args.annotations if args.annotations else 'None (performance only)'}")
    print(f"Number of images: {args.num_images if args.num_images else 'All available'}")
    print(f"Save visualizations: {'Yes' if args.save_vis else 'No'}")
    print(f"CPU-only optimized: {'Yes' if args.cpu_only else 'No'}")
    print(f"Using GPU: {'Yes' if use_gpu else 'No'}")
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
    
    # Initialize benchmark
    benchmark = EnhancedBenchmark(
        output_dir=args.output,
        annotations_path=args.annotations,
        refiner=refiner,
        prioritizer=prioritizer,
        visualizer=visualizer
    )
    
    # Create a unique run name
    run_name = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Run benchmark with memory tracking if enabled
    try:
        if memory_tracker:
            # Run each model combination separately to track memory
            all_results = {}
            
            # Track detector-classifier combinations
            for detector in detectors:
                for classifier in classifiers:
                    combo_name = f"{detector.name}_{detector.model_size}_{classifier.name}_{classifier.model_size}"
                    print(f"\n===== Testing {combo_name} with memory tracking =====")
                    
                    # Start memory tracking
                    memory_tracker.start_tracking(combo_name)
                    
                    # Run just this combination
                    results = benchmark.run_benchmark(
                        detectors=[detector],
                        classifiers=[classifier],
                        end_to_end_systems=None,
                        input_dir=args.images,
                        num_images=args.num_images,
                        save_visualizations=args.save_vis,
                        run_name=run_name
                    )
                    
                    # Stop memory tracking
                    memory_stats = memory_tracker.stop_tracking()
                    
                    # Store results
                    all_results.update(results)
                    
                    # Add memory stats to results
                    for key in results:
                        if 'memory_stats' not in all_results[key]:
                            all_results[key]['memory_stats'] = memory_stats
            
            # Track end-to-end systems
            for system in end_to_end_systems:
                system_name = f"{system.name}_{system.model_size}_end_to_end"
                print(f"\n===== Testing {system_name} with memory tracking =====")
                
                # Start memory tracking
                memory_tracker.start_tracking(system_name)
                
                # Run just this system
                results = benchmark.run_benchmark(
                    detectors=None,
                    classifiers=None,
                    end_to_end_systems=[system],
                    input_dir=args.images,
                    num_images=args.num_images,
                    save_visualizations=args.save_vis,
                    run_name=run_name
                )
                
                # Stop memory tracking
                memory_stats = memory_tracker.stop_tracking()
                
                # Store results
                all_results.update(results)
                
                # Add memory stats to results
                for key in results:
                    if 'memory_stats' not in all_results[key]:
                        all_results[key]['memory_stats'] = memory_stats
            
            # Generate memory usage report
            memory_tracker.plot_peak_comparison()
            memory_report = memory_tracker.generate_report()
            
            print("\nMemory usage report:")
            print(memory_report)
            
        else:
            # Run all combinations at once without memory tracking
            all_results = benchmark.run_benchmark(
                detectors=detectors,
                classifiers=classifiers,
                end_to_end_systems=end_to_end_systems,
                input_dir=args.images,
                num_images=args.num_images,
                save_visualizations=args.save_vis,
                run_name=run_name
            )
        
        # Print top results sorted by performance if we have ground truth
        if args.annotations:
            print("\nTop model combinations by F1 score:")
            
            top_combos = []
            for combo_name, result in all_results.items():
                # Filter for results that have evaluation metrics
                if 'precision' in result and 'recall' in result and 'f1_score' in result:
                    system_type = "End-to-End" if result.get('end_to_end', False) else "Two-Stage"
                    
                    # Format name based on system type
                    if system_type == "End-to-End":
                        name = f"{result['detector']} {result['detector_size']} (End-to-End)"
                    else:
                        name = f"{result['detector']} {result['detector_size']} + {result['classifier']} {result['classifier_size']}"
                    
                    memory_info = result.get('memory_stats', {})
                    peak_memory = memory_info.get('max', 'N/A')
                    
                    top_combos.append({
                        'name': name,
                        'type': system_type,
                        'f1': result['f1_score'],
                        'precision': result['precision'],
                        'recall': result['recall'],
                        'mean_iou': result['mean_iou'],
                        'cls_accuracy': result['cls_accuracy'],
                        'time': result['avg_time_per_image'],
                        'peak_memory': peak_memory
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
                if combo['peak_memory'] != 'N/A':
                    print(f"   Peak memory usage: {combo['peak_memory']:.2f} MB")
                print()
        
        # Otherwise, print top results sorted by speed
        else:
            print("\nTop model combinations by processing speed:")
            
            speed_combos = []
            for combo_name, result in all_results.items():
                system_type = "End-to-End" if result.get('end_to_end', False) else "Two-Stage"
                
                # Format name based on system type
                if system_type == "End-to-End":
                    name = f"{result['detector']} {result['detector_size']} (End-to-End)"
                else:
                    name = f"{result['detector']} {result['detector_size']} + {result['classifier']} {result['classifier_size']}"
                
                memory_info = result.get('memory_stats', {})
                peak_memory = memory_info.get('max', 'N/A')
                
                speed_combos.append({
                    'name': name,
                    'type': system_type,
                    'time': result['avg_time_per_image'],
                    'confidence': result['avg_confidence'],
                    'peak_memory': peak_memory
                })
            
            # Sort by processing time
            speed_combos.sort(key=lambda x: x['time'])
            
            # Print top 3 or all if fewer
            for i, combo in enumerate(speed_combos[:min(3, len(speed_combos))]):
                print(f"{i+1}. {combo['name']} ({combo['type']})")
                print(f"   Average processing time: {combo['time']:.4f}s")
                print(f"   Average confidence: {combo['confidence']:.4f}")
                if combo['peak_memory'] != 'N/A':
                    print(f"   Peak memory usage: {combo['peak_memory']:.2f} MB")
                print()
        
        print(f"\nBenchmark complete. Results saved to {args.output}")
        
    except Exception as e:
        import traceback
        print(f"Error during benchmarking: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()