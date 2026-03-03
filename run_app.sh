#!/bin/bash
# Lung Nodule Detection Application Launcher
# This script launches the GUI application on macOS/Linux

cd "$(dirname "$0")"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed"
    echo "Please install Python 3.9 or later"
    exit 1
fi

# Check if required packages are installed
python3 -c "import PyQt6" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    python3 -m pip install -q -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install required packages"
        exit 1
    fi
    echo "Packages installed successfully!"
    echo ""
fi

# Launch the application
python3 app.py
