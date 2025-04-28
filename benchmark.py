"""
benchmark.py - Utilities for benchmarking different model combinations
"""

import os
import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from tabulate import tabulate

from core.interfaces import (
    ObjectDetector, 
    Classifier, 
    ClassificationRefiner,
    ObjectPrioritizer, 
    Visualizer
)
from pipeline.vehicle_classifier import VehicleClassifier


class ModelBenchmark:
    """Benchmark different detector and classifier combinations"""
    
    def __init__(self, 
                output_dir: str = "./benchmark_results",
                refiner: Optional[ClassificationRefiner] = None,
                prioritizer: Optional[ObjectPrioritizer] = None,
                visualizer: Optional[Visualizer] = None):
        """
        Initialize the benchmark system
        
        Args:
            output_dir: Directory to save benchmark results
            refiner: Classification refiner to use for all tests
            prioritizer: Object prioritizer to use for all tests
            visualizer: Visualizer to use for all tests
        """
        self.output_dir = output_dir
        self.refiner = refiner
        self.prioritizer = prioritizer
        self.visualizer = visualizer
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_benchmark(self, 
                     detectors: List[ObjectDetector],
                     classifiers: List[Classifier],
                     input_dir: str,
                     num_images: int = 20,
                     save_visualizations: bool = True,
                     run_name: Optional[str] = None):
        """
        Run benchmark on all combinations of detectors and classifiers
        
        Args:
            detectors: List of detector implementations to test
            classifiers: List of classifier implementations to test
            input_dir: Directory containing input images
            num_images: Number of images to process for each combination
            save_visualizations: Whether to save visualization images
            run_name: Optional name for this benchmark run
            
        Returns:
            Dictionary of benchmark results
        """
        if run_name is None:
            run_name = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
        
        benchmark_output_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(benchmark_output_dir, exist_ok=True)
        
        results = {}
        all_results = []
        
        # Run each combination
        for detector in detectors:
            for classifier in classifiers:
                combo_name = f"{detector.name}_{detector.model_size}_{classifier.name}_{classifier.model_size}"
                print(f"\n===== Testing {combo_name} =====")
                
                # Create classifier instance
                vehicle_classifier = VehicleClassifier(
                    detector=detector,
                    classifier=classifier,
                    refiner=self.refiner,
                    prioritizer=self.prioritizer,
                    visualizer=self.visualizer
                )
                
                # Create output directory for this combination
                combo_output_dir = os.path.join(benchmark_output_dir, combo_name)
                os.makedirs(combo_output_dir, exist_ok=True)
                
                # Run batch processing
                start_time = time.time()
                batch_results, csv_path = vehicle_classifier.batch_process(
                    input_dir=input_dir,
                    output_dir=combo_output_dir,
                    num_images=num_images,
                    save_visualizations=save_visualizations,
                    include_detector_name=False,
                    include_classifier_name=False
                )
                total_time = time.time() - start_time
                
                # Calculate statistics
                num_processed = len(batch_results)
                avg_time = total_time / max(num_processed, 1)
                
                # Compile class distribution
                class_distribution = {}
                for result in batch_results:
                    cls = result['main_class']
                    if cls in class_distribution:
                        class_distribution[cls] += 1
                    else:
                        class_distribution[cls] = 1
                
                # Calculate average confidence
                avg_confidence = sum(r['main_conf'] for r in batch_results) / max(num_processed, 1)
                
                # Store results
                combo_results = {
                    'detector': detector.name,
                    'detector_size': detector.model_size,
                    'classifier': classifier.name,
                    'classifier_size': classifier.model_size,
                    'num_images': num_processed,
                    'total_time': total_time,
                    'avg_time_per_image': avg_time,
                    'avg_confidence': avg_confidence,
                    'class_distribution': class_distribution,
                    'output_dir': combo_output_dir,
                    'csv_path': csv_path
                }
                
                results[combo_name] = combo_results
                all_results.append(combo_results)
                
                print(f"Processed {num_processed} images in {total_time:.2f}s ({avg_time:.4f}s per image)")
                print(f"Average confidence: {avg_confidence:.4f}")
                print("Class distribution:")
                for cls, count in class_distribution.items():
                    print(f"  {cls}: {count} ({count/num_processed*100:.1f}%)")
        
        # Save overall results
        results_file = os.path.join(benchmark_output_dir, "benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary report
        self._generate_report(all_results, benchmark_output_dir)
        
        return results
    
    def _generate_report(self, results: List[Dict], output_dir: str):
        """Generate benchmark summary report"""
        # Create summary table
        table_data = []
        headers = ["Detector", "Size", "Classifier", "Size", "Avg Time", "Avg Conf", "# Images"]
        
        for result in results:
            row = [
                result['detector'],
                result['detector_size'],
                result['classifier'],
                result['classifier_size'],
                f"{result['avg_time_per_image']:.4f}s",
                f"{result['avg_confidence']:.4f}",
                result['num_images']
            ]
            table_data.append(row)
        
        # Sort by processing time
        table_data.sort(key=lambda x: float(x[4][:-1]))
        
        # Generate text report
        report_path = os.path.join(output_dir, "benchmark_report.txt")
        with open(report_path, 'w') as f:
            f.write("# Vehicle Classification System Benchmark Results\n\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
            f.write("\n\n## Class Distribution by Model Combination\n\n")
            
            for result in results:
                combo = f"{result['detector']} {result['detector_size']} + {result['classifier']} {result['classifier_size']}"
                f.write(f"### {combo}\n\n")
                
                dist_data = []
                for cls, count in result['class_distribution'].items():
                    percentage = count / result['num_images'] * 100
                    dist_data.append([cls, count, f"{percentage:.1f}%"])
                
                f.write(tabulate(dist_data, headers=["Class", "Count", "Percentage"], tablefmt="grid"))
                f.write("\n\n")
        
        # Generate performance comparison plot
        self._plot_performance_comparison(results, output_dir)
    
    def _plot_performance_comparison(self, results: List[Dict], output_dir: str):
        """Plot performance comparison of different model combinations"""
        # Extract data for plotting
        combos = []
        times = []
        confs = []
        
        for result in results:
            combo = f"{result['detector']} {result['detector_size']}\n+ {result['classifier']} {result['classifier_size']}"
            combos.append(combo)
            times.append(result['avg_time_per_image'])
            confs.append(result['avg_confidence'])
        
        # Sort by processing time
        sorted_indices = np.argsort(times)
        combos = [combos[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]
        confs = [confs[i] for i in sorted_indices]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot processing time
        ax1.bar(combos, times, color='skyblue')
        ax1.set_ylabel('Avg. Processing Time (s)')
        ax1.set_title('Performance Comparison by Model Combination')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot confidence
        ax2.bar(combos, confs, color='lightgreen')
        ax2.set_ylabel('Avg. Confidence')
        ax2.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300)
        plt.close()
        
        # Create class distribution plot
        self._plot_class_distribution(results, output_dir)
    
    def _plot_class_distribution(self, results: List[Dict], output_dir: str):
        """Plot class distribution for each model combination"""
        # Get all possible classes
        all_classes = set()
        for result in results:
            all_classes.update(result['class_distribution'].keys())
        
        all_classes = sorted(list(all_classes))
        
        # Prepare data for plotting
        combos = []
        distributions = []
        
        for result in results:
            combo = f"{result['detector']} {result['detector_size']} + {result['classifier']} {result['classifier_size']}"
            combos.append(combo)
            
            # Calculate percentages
            dist = []
            for cls in all_classes:
                count = result['class_distribution'].get(cls, 0)
                percentage = count / result['num_images'] * 100
                dist.append(percentage)
            
            distributions.append(dist)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bottom = np.zeros(len(combos))
        for i, cls in enumerate(all_classes):
            values = [dist[i] for dist in distributions]
            ax.bar(combos, values, bottom=bottom, label=cls)
            bottom += values
        
        ax.set_title('Class Distribution by Model Combination')
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(0, 100)
        ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
        plt.close()
        
        
