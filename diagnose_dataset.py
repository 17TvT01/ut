#!/usr/bin/env python
"""
Chẩn đoán dataset - Kiểm tra data quality
"""

from pathlib import Path
import torch
from nodule_ai.dataset import LIDCDataset
from nodule_ai.annotations import parse_annotation_xml

def diagnose_dataset():
    data_dir = Path("data")
    
    print("=" * 80)
    print("DATASET DIAGNOSIS")
    print("=" * 80)
    print()
    
    # 1. Kiểm tra XML files
    print("1. XML ANNOTATION FILES:")
    print("-" * 80)
    xml_files = list(data_dir.rglob("*.xml"))
    print(f"Total XML files: {len(xml_files)}")
    
    for xml_path in xml_files:
        print(f"\n  {xml_path.relative_to(data_dir)}")
        try:
            annotations = parse_annotation_xml(xml_path)
            print(f"    ✓ Valid - {len(annotations)} annotations")
            for i, ann in enumerate(annotations[:3]):  # Show first 3
                print(f"      - Nodule {i+1}: {len(ann.points)} points")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print()
    
    # 2. Kiểm tra DICOM files
    print("2. DICOM FILES:")
    print("-" * 80)
    for patient_dir in sorted(data_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        dcm_files = list(patient_dir.glob("*.dcm"))
        print(f"  {patient_dir.name}: {len(dcm_files)} DICOM files")
    
    print()
    
    # 3. Load dataset
    print("3. LOADING DATASET:")
    print("-" * 80)
    try:
        dataset = LIDCDataset(data_dir, cache=False, target_shape=(128, 128, 128))
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print()
        
        # 4. Analyze samples
        print("4. SAMPLE ANALYSIS:")
        print("-" * 80)
        
        total_nodules = 0
        total_volume_voxels = 0
        total_mask_voxels = 0
        
        for i in range(len(dataset)):
            print(f"\nSample {i+1}/{len(dataset)}:")
            try:
                sample = dataset[i]
                volume = sample["volume"]
                mask = sample["mask"]
                
                num_nodules = len(sample.get("annotations", []))
                volume_size = volume.numel()
                mask_positive = (mask > 0.5).sum().item()
                
                print(f"  Volume shape: {tuple(volume.shape)}")
                print(f"  Mask shape: {tuple(mask.shape)}")
                print(f"  Nodules: {num_nodules}")
                print(f"  Mask positive voxels: {mask_positive} / {volume_size} ({100*mask_positive/volume_size:.4f}%)")
                
                total_nodules += num_nodules
                total_volume_voxels += volume_size
                total_mask_voxels += mask_positive
                
            except Exception as e:
                print(f"  ✗ Error loading sample: {e}")
        
        print()
        print("=" * 80)
        print("SUMMARY:")
        print("=" * 80)
        print(f"Total samples: {len(dataset)}")
        print(f"Total nodules: {total_nodules}")
        print(f"Total voxels: {total_volume_voxels:,}")
        print(f"Total positive voxels: {total_mask_voxels:,}")
        if total_volume_voxels > 0:
            ratio = total_mask_voxels / total_volume_voxels
            print(f"Positive ratio: {ratio:.6f} ({100*ratio:.4f}%)")
            print(f"Imbalance ratio: 1:{int(1/ratio)}")
        
        # 5. Kiểm tra class imbalance
        print()
        print("5. CLASS IMBALANCE CHECK:")
        print("-" * 80)
        
        if total_mask_voxels == 0:
            print("✗ WARNING: No positive voxels found in masks!")
            print("  Possible issues:")
            print("  - XML annotations are empty or invalid")
            print("  - build_nodule_mask() function has bugs")
            print("  - DICOM coordinates mismatch with annotations")
        elif ratio < 0.0001:
            print(f"⚠ SEVERE IMBALANCE: Only {100*ratio:.4f}% positive voxels")
            print(f"  Need aggressive techniques:")
            print(f"  - Focal Loss (γ=2-5)")
            print(f"  - Weighted sampling")
            print(f"  - Crop around nodules")
        elif ratio < 0.001:
            print(f"⚠ HIGH IMBALANCE: {100*ratio:.4f}% positive voxels")
            print(f"  Recommended:")
            print(f"  - Focal Loss (γ=2)")
            print(f"  - Weighted Dice")
        else:
            print(f"✓ Reasonable balance: {100*ratio:.4f}% positive")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_dataset()
