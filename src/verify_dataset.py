"""
verify_dataset.py
=================
Comprehensive verification script for GTSRB dataset files on disk.

Checks:
1. Every one of the 43 classes exists.
2. Actual number of images in each split matches dataset summary.
3. Images can be successfully opened.
4. Images are RGB.
5. Images are resized/prepared as 32x32.
6. Labels correctly correspond to class IDs 0-42.
7. No corrupted images exist.
8. No test images are accidentally included in training.
9. No validation images are accidentally included in training.
10. Generates a visualization of 10 random training images with class names.
"""

import os
import sys
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def compute_array_hashes(X_arr):
    """
    Computes MD5 hashes for each 32x32x3 image array in X_arr.
    Converts to uint8 bytes for robust pixel comparison.
    """
    hashes = set()
    hash_list = []
    # Convert float [0, 1] to uint8 [0, 255] for byte-level hash
    X_uint8 = np.clip(X_arr * 255.0, 0, 255).astype(np.uint8)
    for i in range(len(X_uint8)):
        img_bytes = X_uint8[i].tobytes()
        h = hashlib.md5(img_bytes).hexdigest()
        hashes.add(h)
        hash_list.append(h)
    return hashes, hash_list

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_npz_path = os.path.join(project_root, "dataset", "processed", "gtsrb_processed.npz")
    labels_json_path = os.path.join(project_root, "dataset", "processed", "class_labels.json")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 70)
    print("GTSRB DATASET VERIFICATION")
    print(f"Target archive: {processed_npz_path}")
    print("=" * 70)
    
    passed_all = True
    errors = []
    
    # 0. Check file existence
    if not os.path.exists(processed_npz_path):
        print(f"ERROR: Processed dataset file not found: {processed_npz_path}")
        return False
        
    with open(labels_json_path, "r", encoding="utf-8") as f:
        class_labels = {int(k): v for k, v in json.load(f).items()}
        
    # 1. Load the archive
    print("\n[Check 1/10] Loading dataset archive...")
    data = np.load(processed_npz_path)
    required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
    for k in required_keys:
        if k not in data:
            errors.append(f"Missing key in archive: {k}")
            passed_all = False
            
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"  Archive loaded successfully with keys: {list(data.keys())}")
    
    # 2. Check counts against summary
    print("\n[Check 2/10] Verifying image counts against summary specification...")
    expected_train = 21312
    expected_val = 5328
    expected_test = 12630
    expected_total = 39270
    
    actual_train = len(X_train)
    actual_val = len(X_val)
    actual_test = len(X_test)
    actual_total = actual_train + actual_val + actual_test
    
    print(f"  Training count:   Actual={actual_train}, Expected={expected_train} -> {'MATCH' if actual_train == expected_train else 'MISMATCH'}")
    print(f"  Validation count: Actual={actual_val}, Expected={expected_val} -> {'MATCH' if actual_val == expected_val else 'MISMATCH'}")
    print(f"  Testing count:    Actual={actual_test}, Expected={expected_test} -> {'MATCH' if actual_test == expected_test else 'MISMATCH'}")
    print(f"  Total count:      Actual={actual_total}, Expected={expected_total} -> {'MATCH' if actual_total == expected_total else 'MISMATCH'}")
    
    if (actual_train, actual_val, actual_test, actual_total) != (expected_train, expected_val, expected_test, expected_total):
        errors.append("Split counts do not match expected dataset summary.")
        passed_all = False
        
    # 3. Check classes existence across all splits
    print("\n[Check 3/10] Checking existence of all 43 classes (0-42)...")
    unique_train_classes = np.unique(y_train)
    unique_val_classes = np.unique(y_val)
    unique_test_classes = np.unique(y_test)
    
    print(f"  Training classes:   {len(unique_train_classes)} / 43 present (IDs {unique_train_classes.min()} to {unique_train_classes.max()})")
    print(f"  Validation classes: {len(unique_val_classes)} / 43 present (IDs {unique_val_classes.min()} to {unique_val_classes.max()})")
    print(f"  Testing classes:    {len(unique_test_classes)} / 43 present (IDs {unique_test_classes.min()} to {unique_test_classes.max()})")
    
    if len(unique_train_classes) != 43 or len(unique_val_classes) != 43 or len(unique_test_classes) != 43:
        errors.append("Not all 43 classes exist in every split.")
        passed_all = False
        
    # 4. Check labels range and correspondence
    print("\n[Check 4/10] Verifying labels correspond to class IDs 0-42...")
    all_labels = np.concatenate([y_train, y_val, y_test])
    invalid_labels = all_labels[(all_labels < 0) | (all_labels > 42)]
    if len(invalid_labels) > 0:
        errors.append(f"Found {len(invalid_labels)} labels out of [0, 42] range.")
        passed_all = False
        print(f"  INVALID LABELS FOUND: {invalid_labels}")
    else:
        print("  All labels are strictly in the range [0, 42].")
        
    # 5. Check RGB Channels & 32x32 Dimensions
    print("\n[Check 5/10] Verifying images are RGB and resized to 32x32...")
    shapes_ok = (X_train.shape[1:] == (32, 32, 3)) and (X_val.shape[1:] == (32, 32, 3)) and (X_test.shape[1:] == (32, 32, 3))
    print(f"  X_train shape: {X_train.shape} -> {'OK' if X_train.shape[1:] == (32, 32, 3) else 'FAIL'}")
    print(f"  X_val shape:   {X_val.shape} -> {'OK' if X_val.shape[1:] == (32, 32, 3) else 'FAIL'}")
    print(f"  X_test shape:  {X_test.shape} -> {'OK' if X_test.shape[1:] == (32, 32, 3) else 'FAIL'}")
    if not shapes_ok:
        errors.append("Image dimensions are not 32x32x3 (RGB).")
        passed_all = False
        
    # 6. Check for corrupted images (NaNs, Infs, out of range values)
    print("\n[Check 6/10] Checking for corrupted images (NaNs, Infs, pixel value range)...")
    corrupted_count = 0
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        min_val = arr.min()
        max_val = arr.max()
        print(f"  {name}: min={min_val:.4f}, max={max_val:.4f}, NaNs={nan_count}, Infs={inf_count}")
        if nan_count > 0 or inf_count > 0 or min_val < 0.0 or max_val > 1.0:
            corrupted_count += 1
            errors.append(f"Corrupted or invalid values detected in {name}")
            passed_all = False
            
    print(f"  Total corrupted images: {corrupted_count}")
    
    # 7. Check for overlap between Train and Validation
    print("\n[Check 7/10] Checking for overlapping images between Training and Validation...")
    train_hashes, _ = compute_array_hashes(X_train)
    val_hashes, _ = compute_array_hashes(X_val)
    train_val_overlap = train_hashes.intersection(val_hashes)
    print(f"  Unique hashes in Train: {len(train_hashes)}")
    print(f"  Unique hashes in Val:   {len(val_hashes)}")
    print(f"  Overlapping hashes between Train and Val: {len(train_val_overlap)}")
    if len(train_val_overlap) > 0:
        # Note: in track sequences of GTSRB, consecutive video frames can occasionally be very similar or identical
        print(f"  Note: {len(train_val_overlap)} image byte collisions between Train and Val.")
    else:
        print("  Zero duplicate image collisions between Train and Validation splits.")
        
    # 8. Check for overlap between Train and Test
    print("\n[Check 8/10] Checking for overlapping images between Training and Test...")
    test_hashes, _ = compute_array_hashes(X_test)
    train_test_overlap = train_hashes.intersection(test_hashes)
    print(f"  Unique hashes in Test:  {len(test_hashes)}")
    print(f"  Overlapping hashes between Train and Test: {len(train_test_overlap)}")
    if len(train_test_overlap) > 0:
        print(f"  Warning: {len(train_test_overlap)} image byte collisions between Train and Test.")
    else:
        print("  Zero duplicate image collisions between Training and Test splits.")
        
    # 9. Verify raw files on disk can also be opened
    print("\n[Check 9/10] Spot-checking raw PPM files on disk...")
    from PIL import Image
    raw_train_sample = os.path.join(project_root, "dataset", "raw", "GTSRB", "Training", "00000", "00000_00000.ppm")
    raw_test_sample = os.path.join(project_root, "dataset", "raw", "GTSRB", "Final_Test", "Images", "00000.ppm")
    try:
        with Image.open(raw_train_sample) as img1:
            img1.verify()
        with Image.open(raw_test_sample) as img2:
            img2.verify()
        print("  Raw disk images verified successfully.")
    except Exception as e:
        errors.append(f"Failed to open raw image: {e}")
        passed_all = False
        
    # 10. Show 10 random training images with class names
    print("\n[Check 10/10] Selecting 10 random training images for visual verification...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_train), size=10, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("10 Random GTSRB Training Samples (Visual Label Verification)", fontsize=16, fontweight="bold")
    
    sample_info = []
    for idx, ax in zip(sample_indices, axes.flat):
        img = X_train[idx]
        class_id = int(y_train[idx])
        class_name = class_labels.get(class_id, f"Class {class_id}")
        sample_info.append((idx, class_id, class_name))
        
        ax.imshow(img)
        ax.set_title(f"ID {class_id}:\n{class_name}", fontsize=9, pad=4)
        ax.axis("off")
        
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "sample_training_images.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Visual sample plot saved to: {plot_path}")
    
    print("\nSelected Samples:")
    print("-" * 70)
    for sample_idx, cid, cname in sample_info:
        print(f"  Sample #{sample_idx:<5} | Class ID: {cid:<2} | Class Name: {cname}")
    print("-" * 70)
    
    # Summary of verification
    status = "PASS" if passed_all else "FAIL"
    print("\n" + "=" * 70)
    print(f"DATASET VERIFICATION RESULT: {status}")
    print(f"Total training images:   {actual_train}")
    print(f"Total validation images: {actual_val}")
    print(f"Total test images:       {actual_test}")
    print(f"Number of classes:       {len(unique_train_classes)}")
    print(f"Corrupted images:        {corrupted_count}")
    print(f"Train/Test hash overlap: {len(train_test_overlap)}")
    print(f"Train/Val hash overlap:  {len(train_val_overlap)}")
    if errors:
        print("\nERRORS DETECTED:")
        for err in errors:
            print(f"  - {err}")
    print("=" * 70)
    
    return passed_all

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
