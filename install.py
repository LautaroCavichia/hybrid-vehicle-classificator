"""
install.py - Script to download models and install dependencies

This script handles the installation of all required packages and downloads
pre-trained models for the vehicle classification system.
"""

import os
import argparse
import subprocess
import sys
from tqdm import tqdm
import torch
import requests
import zipfile
import shutil
from pathlib import Path


def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)


def install_packages():
    """Install required packages using pip"""
    print("Installing required packages...")
    
    # Define required packages based on components
    requirements = [
        # Base requirements
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "matplotlib",
        "opencv-python",
        "tqdm",
        "tabulate",
        
        # YOLO requirements
        "ultralytics",
        
        # Supervision requirements
        "supervision",
        
        # CLIP requirements
        "ftfy",
        "regex",
        "git+https://github.com/openai/CLIP.git",
        
        # DINO requirements
        "transformers",
        
        # GLIP requirements
        "transformers"
    ]
    
    # Install each package
    for package in tqdm(requirements, desc="Installing packages"):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            print("Continuing with installation...")
    
    print("Package installation complete.")


def download_models(model_dir="./models/weights", detectors=None, classifiers=None):
    """Download pre-trained models"""
    if detectors is None:
        detectors = ["yolo"]
    if classifiers is None:
        classifiers = ["clip"]
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Dictionary of model download URLs
    model_urls = {
        "yolo": {
            "nano": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "small": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
            "medium": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
            "large": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
            "xlarge": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
        }
    }
    
    # Download selected detector models
    for detector in detectors:
        if detector.lower() == "yolo":
            print("Downloading YOLO models...")
            for size, url in model_urls["yolo"].items():
                dest_path = os.path.join(model_dir, f"yolov8{size[0]}.pt")
                if not os.path.exists(dest_path):
                    print(f"Downloading YOLOv8 {size} model...")
                    download_file(url, dest_path)
                else:
                    print(f"YOLOv8 {size} model already exists.")
    
    # For CLIP, DINO, and GLIP, we'll use the transformers and CLIP libraries 
    # which download models automatically
    
    print("Model download complete.")


def download_test_data(data_dir="./test_data"):
    """Download some test images for verification"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Sample URLs of vehicle images (replace with actual URLs in production)
    image_urls = [
        "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?q=80&w=1000",  # Car
        "https://images.unsplash.com/photo-1601171293061-fda9c68b1fce?q=80&w=1000",  # Van
        "https://images.unsplash.com/photo-1542319465-68c39d98adb1?q=80&w=1000",    # Truck
        "https://images.unsplash.com/photo-1570125909232-eb263c188f7e?q=80&w=1000",  # Bus
        "https://images.unsplash.com/photo-1635263825462-31a1a444114b?q=80&w=1000"   # Emergency vehicle
    ]
    
    print("Downloading test images...")
    for i, url in enumerate(image_urls):
        dest_path = os.path.join(data_dir, f"vehicle_{i+1}.jpg")
        if not os.path.exists(dest_path):
            download_file(url, dest_path)
    
    print(f"Test images downloaded to {data_dir}")


def setup_project_structure():
    """Create the project directory structure"""
    # Define the directory structure based on the architecture
    directories = [
        "./core",
        "./models/detectors",
        "./models/classifiers",
        "./models/refiners",
        "./pipeline",
        "./data",
        "./output",
        "./benchmark_results"
    ]
    
    print("Setting up project structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create __init__.py files in each Python package directory
    for directory in directories:
        if directory != "./output" and directory != "./benchmark_results":
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    pass  # Create an empty file
    
    print("Project structure setup complete.")


def main():
    """Main installation function"""
    parser = argparse.ArgumentParser(description="Install Vehicle Classification System")
    
    parser.add_argument("--skip-packages", action="store_true",
                      help="Skip installing packages (use if already installed)")
    parser.add_argument("--skip-models", action="store_true",
                      help="Skip downloading pre-trained models")
    parser.add_argument("--skip-test-data", action="store_true",
                      help="Skip downloading test images")
    parser.add_argument("--detectors", nargs="+", choices=["yolo", "all"],
                      default=["all"], help="Detectors to download models for")
    parser.add_argument("--classifiers", nargs="+", choices=["clip", "dino", "glip", "all"],
                      default=["all"], help="Classifiers to download models for")
    
    args = parser.parse_args()
    
    # Print welcome message
    print("=" * 80)
    print("Vehicle Classification System - Installation")
    print("=" * 80)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Warning: CUDA is not available. The system will run on CPU, which may be slow.")
    
    # Setup project structure
    setup_project_structure()
    
    # Install packages if not skipped
    if not args.skip_packages:
        install_packages()
    else:
        print("Skipping package installation.")
    
    # Download models if not skipped
    if not args.skip_models:
        # Process detector and classifier selections
        detectors = []
        if "all" in args.detectors:
            detectors = ["yolo", "supervision", "glip"]
        else:
            detectors = args.detectors
        
        classifiers = []
        if "all" in args.classifiers:
            classifiers = ["clip", "dino", "glip"]
        else:
            classifiers = args.classifiers
        
        download_models(detectors=detectors, classifiers=classifiers)
    else:
        print("Skipping model downloads.")
    
    # Download test data if not skipped
    if not args.skip_test_data:
        download_test_data()
    else:
        print("Skipping test data download.")
    
    print("\nInstallation complete!")
    print("\nTo verify the installation, run:")
    print("python main.py --mode single --input ./test_data/vehicle_1.jpg --output ./output/test_result.jpg")


if __name__ == "__main__":
    main()