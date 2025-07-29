"""
Streamlit deployment helper
This script ensures proper path setup for deployment
"""
import sys
import os
from pathlib import Path

def setup_deployment_paths():
    """Setup paths for Streamlit Cloud deployment"""
    
    # Get the current directory (where this script is located)
    current_dir = Path(__file__).parent.absolute()
    
    # Add current directory to Python path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Add image_processing directory to path
    image_processing_dir = current_dir / "image_processing"
    if image_processing_dir.exists() and str(image_processing_dir) not in sys.path:
        sys.path.insert(0, str(image_processing_dir))
    
    print(f"✅ Deployment paths configured:")
    print(f"   📁 Current directory: {current_dir}")
    print(f"   📁 Image processing: {image_processing_dir}")
    print(f"   🐍 Python path: {sys.path[:3]}...")

# Run setup when imported
setup_deployment_paths()
