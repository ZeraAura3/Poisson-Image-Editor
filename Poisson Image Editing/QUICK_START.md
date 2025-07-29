# Enhanced Poisson Image Editor - Quick Start Guide

Welcome to the Enhanced Poisson Image Editor! This guide will help you get started quickly.

## 🚀 Quick Launch (Windows)

1. **Double-click** `fix_and_run.bat` to automatically set up and start the application (recommended)
   OR
2. **Double-click** `launch.bat` for standard launch
3. The application will be available at: `http://localhost:8501`
4. Open your web browser and navigate to the URL above

## 🚀 Quick Launch (Linux/Mac)

1. Run the fix script (recommended):
   ```bash
   chmod +x fix_and_run.sh
   ./fix_and_run.sh
   ```
2. OR run the standard launch script:
   ```bash
   chmod +x launch.sh
   ./launch.sh
   ```
3. The application will be available at: `http://localhost:8501`

## 📋 Manual Setup (If needed)

If the launch scripts don't work, follow these steps:

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r streamlit_requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 🎯 How to Use

1. **Upload Images**: Use the sidebar to upload source and target images
2. **Create Mask**: Choose from automatic or manual masking options
3. **Adjust Parameters**: Fine-tune blending parameters for optimal results
4. **Process**: Click "Process Images" to apply Poisson blending
5. **Download**: Save your enhanced composite image

## 📊 Features Available

- **Multi-scale Poisson Blending**: Advanced pyramid-based processing
- **Intelligent Masking**: K-means, GrabCut, and edge-based mask generation
- **Color Correction**: LAB color space processing for natural results
- **Interactive UI**: Real-time parameter adjustment and preview
- **Comparison Tools**: Side-by-side before/after comparisons
- **Export Options**: Multiple format support with quality settings

## 🆘 Troubleshooting

**Issue**: Dependencies fail to install
**Solution**: Update pip and try again:
```bash
python -m pip install --upgrade pip
```

**Issue**: Streamlit doesn't start
**Solution**: Check if port 8501 is available or specify a different port:
```bash
streamlit run app.py --server.port=8502
```

**Issue**: Out of memory errors
**Solution**: Use smaller images or enable memory optimization in settings

## 📁 Project Structure

```
Enhanced Poisson Image Editor/
├── app.py                      # Main Streamlit application
├── image_editing.ipynb         # Core implementation notebook
├── image_processing/           # Image processing modules
├── ui_components.py           # UI components
├── pics/                      # Sample images
├── requirements.txt           # Python dependencies
├── DEPLOYMENT.md             # Deployment guide
└── README.md                 # Project documentation
```

## 🔗 Additional Resources

- **Full Documentation**: See `README.md` for complete project details
- **Deployment Guide**: Check `DEPLOYMENT.md` for hosting options
- **Research Paper**: Read the included PDF for algorithm details

---

**Ready to create stunning image composites? Launch the application and start editing!** 🎨
