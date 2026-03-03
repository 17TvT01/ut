# Lung Nodule Detection Application

A desktop application for lung nodule detection and analysis using deep learning.

## Quick Start

### Windows
Double-click `run_app.bat` or run in command prompt:
```bash
run_app.bat
```

### macOS/Linux
Run in terminal:
```bash
bash run_app.sh
```

## Manual Setup

### 1. Install Python
- Download Python 3.9+ from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
python app.py
```

## System Requirements

- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 20.04+)
- **Python**: 3.9 or later
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 2GB free space for models and data

## Features

- Load and analyze DICOM CT scan files
- Load and analyze LUNA16 `.mhd` volumes
- Interactive 3D visualization
- Automated nodule detection and segmentation
- Detailed analysis reports
- Training and fine-tuning capabilities
- Export results in multiple formats

## Using LUNA16 Dataset

- Expected data layout:
    - `LUNA16/subset0 ... subset9` (contains `.mhd` + `.raw`)
    - `LUNA16/annotations.csv` or `LUNA16/CSVFILES/annotations.csv`
- Training tab:
    - Select LUNA16 root folder as training data directory.
    - App auto-detects LUNA16 format and builds masks from `annotations.csv`.
- Analysis tab:
    - You can select a single `.mhd` file as input source.

## Troubleshooting

### "Python not found"
- Ensure Python is installed and added to PATH
- Try `python3` instead of `python`

### "ModuleNotFoundError"
- Run: `pip install -r requirements.txt`
- On Linux/macOS: `pip3 install -r requirements.txt`

### Performance Issues
- Close other applications to free up RAM
- Install GPU drivers for faster processing
- Check system requirements above

## Application Structure

```
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── run_app.bat           # Windows launcher
├── run_app.sh            # macOS/Linux launcher
└── nodule_ai/            # Core package
    ├── model.py          # Neural network models
    ├── inference.py      # Inference pipeline
    ├── training.py       # Training utilities
    └── ...
```

## License

This application is provided as-is for medical imaging analysis purposes.

## Support

For issues or questions, please check the documentation or contact the development team.
