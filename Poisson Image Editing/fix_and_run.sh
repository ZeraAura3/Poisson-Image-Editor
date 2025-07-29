#!/bin/bash

# Deployment Fix Script for Enhanced Poisson Image Editor
# This script resolves common deployment issues

echo "🔧 Enhanced Poisson Image Editor - Deployment Fix"
echo "================================================="

# Fix 1: Handle spaces in directory names
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "✅ Working directory: $SCRIPT_DIR"

# Fix 2: Ensure Python path is correct
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/image_processing:$PYTHONPATH"
echo "✅ Python path configured"

# Fix 3: Check for required files
if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "❌ requirements.txt not found in $SCRIPT_DIR"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/app.py" ]; then
    echo "❌ app.py not found in $SCRIPT_DIR"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/image_processing" ]; then
    echo "❌ image_processing directory not found"
    exit 1
fi

echo "✅ All required files found"

# Fix 4: Install dependencies with proper path handling
echo "📦 Installing dependencies..."
cd "$SCRIPT_DIR"
pip install -r ./requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Fix 5: Launch application
echo "🚀 Starting Enhanced Poisson Image Editor..."
echo "   Application will be available at: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo

# Use absolute path for streamlit
cd "$SCRIPT_DIR"
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
