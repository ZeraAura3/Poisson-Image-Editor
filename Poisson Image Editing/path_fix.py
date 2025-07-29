# Fix for deployment issues with spaces in directory names
import os
import sys

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add to Python path if not already there
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add image_processing directory to path
image_processing_dir = os.path.join(current_dir, 'image_processing')
if image_processing_dir not in sys.path:
    sys.path.insert(0, image_processing_dir)

print(f"✅ Python path configured for: {current_dir}")
print(f"✅ Image processing path: {image_processing_dir}")
