import torch
import sys
from pathlib import Path
import time
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from nodule_ai.dataset import LIDCDataset
from nodule_ai.model import ComplexUNet3D

def load_model(checkpoint_path, device):
    model = ComplexUNet3D(
        n_channels=1,
        n_classes=1,
        base_filters=16,
        dropout=0.1,
        upsample_mode="trilinear",
    )
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
        
    return model.to(device).eval()

def main():
    print("=== Test Lung Nodule Segmentation Algorithm ===")
    
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    data_dir = Path("data")
    checkpoint_path = Path("checkpoints/complex_unet3d.pt")
    
    # 2. Load Data
    print("\n[1/3] Loading dataset sample...")
    try:
        dataset = LIDCDataset(data_dir, cache=False, target_shape=(160, 160, 160))
        if len(dataset) == 0:
            print("Error: Dataset is empty.")
            return
        
        sample = dataset[0]
        volume = sample["volume"].unsqueeze(0).to(device) # Add batch dim
        print(f"Successfully loaded sample. Volume shape: {volume.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 3. Load Model
    print("\n[2/3] Loading model...")
    try:
        model = load_model(checkpoint_path, device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Run Inference
    print("\n[3/3] Running inference...")
    try:
        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                output = model(volume)
                prob = torch.sigmoid(output)
        
        elapsed = time.time() - start_time
        print(f"Inference completed in {elapsed:.3f}s")
        print(f"Output shape: {prob.shape}")
        print(f"Output range: [{prob.min():.4f}, {prob.max():.4f}]")
        
        # Simple validation
        if prob.shape == volume.shape:
            print("\n✅ TEST PASSED: Input/Output shapes match.")
        else:
            print(f"\n❌ TEST FAILED: Shape mismatch. Input {volume.shape} vs Output {prob.shape}")
            
    except Exception as e:
        print(f"Error during inference: {e}")
        return

if __name__ == "__main__":
    main()
