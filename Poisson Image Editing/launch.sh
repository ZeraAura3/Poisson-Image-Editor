#!/bin/bash

# Enhanced Poisson Image Editor - Launch Script
# This script sets up and launches the application

echo "🎨 Enhanced Poisson Image Editor - Launch Script"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python found: $python_version"
else
    echo "❌ Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r ./requirements.txt

# Check if installation was successful
if [[ $? -eq 0 ]]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p pics uploads results temp

# Check for sample images
if [ -d "pics" ] && [ "$(ls -A pics)" ]; then
    echo "✅ Sample images found in pics directory"
else
    echo "ℹ️  No sample images found. You can add sample images to the pics/ directory"
fi

# Launch application
echo ""
echo "🚀 Launching Enhanced Poisson Image Editor..."
echo "   Application will be available at: http://localhost:8501"
echo "   Press Ctrl+C to stop the application"
echo ""

# Start Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
