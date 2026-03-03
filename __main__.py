"""
Lung Nodule Detection Application
Version 1.0.0

Desktop GUI application for lung nodule detection and analysis.
Run this script to launch the application.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main application
if __name__ == "__main__":
    try:
        from app import main
        main()
    except ImportError as e:
        print(f"Error: Missing required modules: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)
