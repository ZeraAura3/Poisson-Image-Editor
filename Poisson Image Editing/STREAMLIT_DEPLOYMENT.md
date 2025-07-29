# Streamlit Cloud Deployment Guide

## 🚀 Quick Deployment Steps

### 1. Repository Setup

Make sure your GitHub repository has this structure:
```
your-repo/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── packages.txt             # System packages (for OpenCV)
├── .streamlit/
│   ├── config.toml          # Streamlit configuration
│   └── secrets.toml         # Environment variables
├── image_processing/        # Your custom modules
│   └── enhanced_blender.py
└── pics/                    # Sample images
```

### 2. Streamlit Cloud Settings

When deploying on Streamlit Cloud:

1. **Repository**: `your-username/your-repo-name`
2. **Branch**: `main` (or your default branch)
3. **Main file path**: `app.py` ⚠️ **IMPORTANT: NOT `Poisson Image Editing/app.py`**
4. **Python version**: 3.9

### 3. Fix Current Deployment Issue

The current error is because Streamlit Cloud is looking for:
- Main module: `Poisson Image Editing/app.py` ❌
- Requirements: `Editing/requirements.txt` ❌

**Solution**: Update the deployment settings to:
- Main module: `app.py` ✅
- This will automatically find `requirements.txt` in the root ✅

### 4. Repository Structure Fix

If your files are in a subfolder called "Poisson Image Editing", you have two options:

#### Option A: Move files to root (Recommended)
```bash
# Move all files from subfolder to root
mv "Poisson Image Editing"/* .
```

#### Option B: Update deployment path
- Set main file path to: `Poisson Image Editing/app.py`
- But this requires requirements.txt to be in the subfolder

### 5. Requirements File

Ensure your `requirements.txt` contains:
```
numpy>=1.21.0,<2.0.0
opencv-python-headless>=4.5.0,<5.0.0
scipy>=1.7.0,<2.0.0
matplotlib>=3.3.0,<4.0.0
scikit-image>=0.18.0,<1.0.0
Pillow>=9.5.0,<11.0.0
streamlit>=1.28.0,<2.0.0
plotly>=5.15.0,<6.0.0
pandas>=1.3.0,<3.0.0
```

### 6. System Packages File

Create `packages.txt` for OpenCV dependencies:
```
libgl1-mesa-glx
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgomp1
```

### 7. Troubleshooting

**Error: "Failed to parse requirements.txt"**
- Check file path in deployment settings
- Ensure requirements.txt is in the same directory as app.py

**Error: "Import module not found"**
- Check that all custom modules are in the repository
- Verify import paths in app.py

**Error: "OpenCV not found"**
- Use `opencv-python-headless` instead of `opencv-python`
- Add system packages in `packages.txt`

### 8. Test Locally

Before deploying, test locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Run health check
python health_check.py

# Start Streamlit
streamlit run app.py
```

### 9. Deployment Commands

For other platforms:

**Docker:**
```bash
docker build -t poisson-editor .
docker run -p 8501:8501 poisson-editor
```

**Local:**
```bash
streamlit run app.py --server.port 8501
```

---

## 📞 Support

If you encounter issues:
1. Check the deployment logs in Streamlit Cloud
2. Run the health check script locally
3. Verify all files are in the correct locations
4. Check Python package versions

Good luck with your deployment! 🚀
