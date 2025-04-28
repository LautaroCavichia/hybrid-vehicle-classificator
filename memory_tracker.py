"""
memory_tracker.py - Component for tracking memory usage during benchmarking
"""

import os
import time
import psutil
import threading
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable


class MemoryTracker:
    """Tracks memory usage over time during model execution"""
    
    def __init__(self, 
                 output_dir: str = "./memory_stats",
                 sampling_interval: float = 0.1):
        """
        Initialize the memory tracker
        
        Args:
            output_dir: Directory to save memory usage plots
            sampling_interval: Time interval between memory measurements (seconds)
        """
        self.output_dir = output_dir
        self.sampling_interval = sampling_interval
        self.is_tracking = False
        self.tracking_thread = None
        self.current_label = None
        self.memory_data = {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def start_tracking(self, label: str):
        """
        Start tracking memory usage
        
        Args:
            label: Label for the current tracking session
        """
        if self.is_tracking:
            self.stop_tracking()
        
        self.current_label = label
        self.memory_data[label] = {
            'timestamps': [],
            'memory_usage': []
        }
        
        self.is_tracking = True
        self.tracking_thread = threading.Thread(target=self._track_memory)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
    
    def stop_tracking(self) -> Dict:
        """
        Stop the current tracking session
        
        Returns:
            Dictionary with memory usage statistics
        """
        if not self.is_tracking:
            return {}
        
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2.0)
        
        # Calculate statistics
        if self.current_label and self.current_label in self.memory_data:
            data = self.memory_data[self.current_label]
            if data['memory_usage']:
                stats = {
                    'max': max(data['memory_usage']),
                    'min': min(data['memory_usage']),
                    'mean': np.mean(data['memory_usage']),
                    'peak_increase': max(data['memory_usage']) - data['memory_usage'][0] if data['memory_usage'] else 0
                }
                self.memory_data[self.current_label]['stats'] = stats
                return stats
        
        return {}
    
    def _track_memory(self):
        """Background thread that periodically samples memory usage"""
        start_time = time.time()
        process = psutil.Process()
        
        while self.is_tracking:
            try:
                # Get current memory usage in MB
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # Get elapsed time
                elapsed = time.time() - start_time
                
                # Store data
                self.memory_data[self.current_label]['timestamps'].append(elapsed)
                self.memory_data[self.current_label]['memory_usage'].append(memory_mb)
                
                # Sleep for the specified interval
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"Error in memory tracking: {e}")
                break
    
    def plot_memory_usage(self, label: str = None, save: bool = True) -> Optional[plt.Figure]:
        """
        Generate plot of memory usage over time
        
        Args:
            label: Specific tracking session to plot (None for all)
            save: Whether to save the plot to disk
            
        Returns:
            Matplotlib figure object, or None if no data
        """
        if label:
            if label not in self.memory_data:
                print(f"No memory data found for '{label}'")
                return None
            
            labels_to_plot = [label]
        else:
            labels_to_plot = list(self.memory_data.keys())
        
        if not labels_to_plot:
            print("No memory data to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for label in labels_to_plot:
            data = self.memory_data[label]
            timestamps = data['timestamps']
            memory_usage = data['memory_usage']
            
            if not timestamps or not memory_usage:
                continue
            
            # Plot memory usage over time
            ax.plot(timestamps, memory_usage, label=label)
            
            # Add statistics as text annotation if available
            if 'stats' in data:
                stats = data['stats']
                max_time = timestamps[-1]
                max_mem = stats['max']
                ax.annotate(
                    f"Peak: {max_mem:.1f} MB",
                    xy=(timestamps[memory_usage.index(max_mem)], max_mem),
                    xytext=(5, 5),
                    textcoords='offset points'
                )
        
        ax.set_title('Memory Usage Over Time')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory Usage (MB)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if len(labels_to_plot) > 1:
            ax.legend()
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if len(labels_to_plot) == 1:
                filename = f"memory_usage_{labels_to_plot[0].replace(' ', '_')}.png"
            else:
                filename = "memory_usage_comparison.png"
                
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        
        return fig
    
    def plot_peak_comparison(self, save: bool = True) -> Optional[plt.Figure]:
        """
        Generate bar chart comparing peak memory usage across all tracked sessions
        
        Args:
            save: Whether to save the plot to disk
            
        Returns:
            Matplotlib figure object, or None if no data
        """
        if not self.memory_data:
            print("No memory data to plot")
            return None
        
        # Extract peak memory usage for each label
        labels = []
        peaks = []
        increases = []
        
        for label, data in self.memory_data.items():
            if 'stats' in data:
                labels.append(label)
                peaks.append(data['stats']['max'])
                increases.append(data['stats']['peak_increase'])
        
        if not labels:
            print("No memory statistics available")
            return None
        
        # Sort by peak memory usage
        sorted_indices = np.argsort(peaks)
        labels = [labels[i] for i in sorted_indices]
        peaks = [peaks[i] for i in sorted_indices]
        increases = [increases[i] for i in sorted_indices]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot peak memory
        ax1.barh(labels, peaks, color='skyblue')
        ax1.set_title('Peak Memory Usage')
        ax1.set_xlabel('Memory Usage (MB)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot memory increase
        ax2.barh(labels, increases, color='coral')
        ax2.set_title('Peak Memory Increase')
        ax2.set_xlabel('Memory Increase (MB)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            plt.savefig(os.path.join(self.output_dir, "memory_peak_comparison.png"), dpi=300)
        
        return fig
    
    def generate_report(self) -> str:
        """
        Generate a text report of memory usage statistics
        
        Returns:
            Formatted report string
        """
        if not self.memory_data:
            return "No memory data collected"
        
        # Collect data for report
        report_data = []
        for label, data in self.memory_data.items():
            if 'stats' in data:
                stats = data['stats']
                report_data.append({
                    'label': label,
                    'max': stats['max'],
                    'min': stats['min'],
                    'mean': stats['mean'],
                    'increase': stats['peak_increase']
                })
        
        # Sort by max memory usage
        report_data.sort(key=lambda x: x['max'])
        
        # Format report
        report = "Memory Usage Report\n"
        report += "===================\n\n"
        
        for item in report_data:
            report += f"Model: {item['label']}\n"
            report += f"  Peak Memory: {item['max']:.2f} MB\n"
            report += f"  Minimum Memory: {item['min']:.2f} MB\n"
            report += f"  Mean Memory: {item['mean']:.2f} MB\n"
            report += f"  Peak Increase: {item['increase']:.2f} MB\n\n"
        
        # Save report
        report_path = os.path.join(self.output_dir, "memory_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report


# Example usage
if __name__ == "__main__":
    tracker = MemoryTracker()
    
    # Simulate tracking for a model
    tracker.start_tracking("YOLO_small + CLIP_small")
    
    # Simulate memory-intensive operation
    large_arrays = []
    for i in range(10):
        large_arrays.append(np.random.rand(1000, 1000))
        time.sleep(0.5)
    
    # Stop tracking
    stats = tracker.stop_tracking()
    print(f"Memory stats: {stats}")
    
    # Generate plots
    tracker.plot_memory_usage()
    
    # Clean up
    del large_arrays