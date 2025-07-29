@echo off
REM Deployment Fix Script for Enhanced Poisson Image Editor (Windows)
REM This script resolves common deployment issues

echo 🔧 Enhanced Poisson Image Editor - Deployment Fix
echo =================================================

REM Fix 1: Get current directory with proper handling of spaces
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
echo ✅ Working directory: "%SCRIPT_DIR%"

REM Fix 2: Set Python path
set "PYTHONPATH=%SCRIPT_DIR%;%SCRIPT_DIR%\image_processing;%PYTHONPATH%"
echo ✅ Python path configured

REM Fix 3: Check for required files
if not exist "%SCRIPT_DIR%\requirements.txt" (
    echo ❌ requirements.txt not found in "%SCRIPT_DIR%"
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%\app.py" (
    echo ❌ app.py not found in "%SCRIPT_DIR%"
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%\image_processing" (
    echo ❌ image_processing directory not found
    pause
    exit /b 1
)

echo ✅ All required files found

REM Fix 4: Install dependencies with proper path handling
echo 📦 Installing dependencies...
cd /d "%SCRIPT_DIR%"
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully

REM Fix 5: Launch application
echo.
echo 🚀 Starting Enhanced Poisson Image Editor...
echo    Application will be available at: http://localhost:8501
echo    Press Ctrl+C to stop
echo.

REM Use current directory for streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
