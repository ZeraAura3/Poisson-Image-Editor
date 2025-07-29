@echo off
REM Enhanced Poisson Image Editor - Windows Launch Script

echo 🎨 Enhanced Poisson Image Editor - Launch Script
echo ==================================================

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set python_version=%%i
echo ✅ Python found: %python_version%

REM Check if virtual environment exists
if not exist "venv" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully

REM Create necessary directories
echo 📁 Creating directories...
if not exist "pics" mkdir pics
if not exist "uploads" mkdir uploads
if not exist "results" mkdir results
if not exist "temp" mkdir temp

REM Check for sample images
dir /b pics\*.* >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Sample images found in pics directory
) else (
    echo ℹ️  No sample images found. You can add sample images to the pics\ directory
)

REM Launch application
echo.
echo 🚀 Launching Enhanced Poisson Image Editor...
echo    Application will be available at: http://localhost:8501
echo    Press Ctrl+C to stop the application
echo.

REM Start Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
