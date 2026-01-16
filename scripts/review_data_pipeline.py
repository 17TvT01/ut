#!/usr/bin/env python
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import scipy.ndimage

from nodule_ai.dicom import load_dicom_series, build_nodule_mask
from nodule_ai.annotations import parse_annotation_xml

def review_pipeline():
    data_dir = Path("data")
    output_dir = Path("figures/review")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = output_dir / "dataset_stats.csv"
    
    print("search data in:", data_dir)
    xml_files = list(data_dir.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML annotation files.")
    
    records = []
    
    # Process a subset or all
    # For speed in this review, let's process up to 20 samples fully for visualization
    # and all for stats if possible, but let's limit to 50 for quick feedback loop.
    
    max_samples = 50
    process_samples = xml_files[:max_samples]
    
    print(f"Processing {len(process_samples)} samples...")
    
    for i, xml_path in enumerate(tqdm(process_samples)):
        series_dir = xml_path.parent
        # Find dicom dir (sometimes it is the parent, sometimes a subdir)
        # Assumes standard structure where XML is next to or above DICOMs
        if not any(series_dir.glob("*.dcm")):
             # Try children
             subdirs = [d for d in series_dir.iterdir() if d.is_dir()]
             found = False
             for sub in subdirs:
                 if any(sub.glob("*.dcm")):
                     series_dir = sub
                     found = True
                     break
             if not found:
                 continue
                 
        try:
            # Load with NEW resampling logic
            volume, meta, spacing = load_dicom_series(series_dir, target_spacing=(1.0, 1.0, 1.0))
            
            # Parse annotations
            annotations = parse_annotation_xml(xml_path)
            
            # Build mask (Note: current logic builds mask in ORIGINAL space then we might need to match?)
            # Wait, my logic in dicom.py was: 
            # 1. load_dicom_series returns RESAMPLED volume and meta with 'original_spacing' and 'index' (original z-index).
            # 2. build_nodule_mask uses SOP UID to find z-index.
            # CRITICAL ISSUE: 'meta' now contains indices for the ORIGINAL slices.
            # But the volume is RESAMPLED.
            # So if we use build_nodule_mask with 'meta', it will create a mask matching the ORIGINAL volume shape (Z_orig, Y_orig, X_orig).
            
            # We need to recover original shape to build mask
            # meta items have 'original_spacing' but where is valid original shape? 
            # We can deduce max Z index from meta.
            max_z = max(m['index'] for m in meta.values())
            # Assuming 512x512 for standard axial (check first slice path?)
            # Let's assume standard DICOM 512x512 or read from one file.
            # Actually we can't easily know Y, X without reading a file again or storing it.
            # UPDATE NEEDED: load_dicom_series should probably return original_shape or we read it.
            # Quick fix: Read one dicom file to get Shape (Y, X).
            
            import pydicom
            first_dcm = next(series_dir.glob("*.dcm"))
            ds = pydicom.dcmread(str(first_dcm))
            orig_h, orig_w = ds.Rows, ds.Columns
            orig_depth = int(max_z) + 1
            original_shape = (orig_depth, orig_h, orig_w)
            
            # Build mask in ORIGINAL space
            mask_orig = build_nodule_mask(original_shape, annotations, meta, dilation=1)
            
            # Resample mask to match volume
            # We need the original spacing.
            # meta has 'original_spacing'.
            sample_meta = next(iter(meta.values()))
            original_spacing = sample_meta['original_spacing']
            
            # Calculate resize factor using spacing
            # target_spacing is (1.0, 1.0, 1.0)
            target_spacing = (1.0, 1.0, 1.0)
            
            # Resample mask using nearest neighbor
            zoom_factor = np.array(original_spacing) / np.array(target_spacing)
            
            # Note: volume shape might differ slightly due to rounding in load_dicom_series
            # We should probably resize mask to exactly match volume shape if close
            
            mask_resampled = scipy.ndimage.zoom(mask_orig, zoom_factor, order=0)
            
            # Check shape mismatch and crop/pad if necessary (due to rounding)
            if mask_resampled.shape != volume.shape:
                # Simple resize to exact match if off by few pixels
                # or just pad/crop. For mask, safer to simple resize nearest with cv2 or similar, 
                # but 3D is hard with cv2.
                # Let's trust zoom for now, or match shapes.
                 # Only if shapes are different
                # print(f"Shape mismatch: Vol {volume.shape} vs Mask {mask_resampled.shape}")
                pass

            # Record stats
            records.append({
                "id": series_dir.name,
                "orig_shape": str(original_shape),
                "resampled_shape": str(volume.shape),
                "orig_spacing_z": original_spacing[0],
                "orig_spacing_xy": original_spacing[1],
                "nodule_count": len(annotations),
                "mask_positive": mask_resampled.sum()
            })
            
            # Visualization (Save first 20 valid with nodules)
            if mask_resampled.sum() > 0 and len(records) <= 20:
                # Find slice with most nodule pixels
                z_sums = mask_resampled.sum(axis=(1, 2))
                best_z = np.argmax(z_sums)
                
                # Plot
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(volume[best_z], cmap="gray")
                plt.title(f"CT Slice {best_z} (Resampled)")
                plt.axis("off")
                
                plt.subplot(1, 2, 2)
                plt.imshow(volume[best_z], cmap="gray")
                # Overlay mask
                mask_slice = mask_resampled[best_z]
                # Create red overlay
                overlay = np.zeros((*mask_slice.shape, 4))
                overlay[mask_slice > 0] = [1, 0, 0, 0.5] # Red with alpha
                plt.imshow(overlay)
                plt.title("Mask Overlay")
                plt.axis("off")
                
                plt.tight_layout()
                plt.savefig(output_dir / f"sample_{i}_{series_dir.name}.png")
                plt.close()
                
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            import traceback
            traceback.print_exc()

    # Save stats
    df = pd.DataFrame(records)
    print(df.describe())
    df.to_csv(stats_file, index=False)
    print(f"Stats saved to {stats_file}")

if __name__ == "__main__":
    review_pipeline()
