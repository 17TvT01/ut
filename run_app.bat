@echo off
REM Lung Nodule Detection Application Launcher
REM This script launches the GUI application

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9 or later from python.org
    pause
    exit /b 1
)

REM Check if required packages are installed
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    echo.
    python -m pip install -q -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install required packages
        pause
        exit /b 1
    )
    echo Packages installed successfully!
    echo.
)

REM Launch the application
python app.py

if errorlevel 1 (
    echo.
    echo Error: Application failed to start
    pause
    exit /b 1
)
