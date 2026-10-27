"""
prepare_dataset.py
==================
High-performance preprocessing pipeline for the German Traffic Sign Recognition Benchmark (GTSRB).

Features:
- Reads official GTSRB training and test PPM images using OpenCV and multithreaded I/O
- Converts images to RGB color space
- Resizes images to 32x32 pixels
- Normalizes pixel values to [0.0, 1.0] as float32
- Preserves all 43 class labels (0 to 42)
- Creates stratified Train / Validation splits using reproducible random seed
- Evaluates official Test set annotations from GT-final_test.csv
- Saves compressed preprocessed dataset in `dataset/processed/gtsrb_processed.npz`
- Exports class label mappings to `dataset/processed/class_labels.json`
- Generates `dataset_summary.txt` with comprehensive statistics
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Official GTSRB 43 Class Label Descriptions
SIGN_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

def load_and_preprocess_single_image(args):
    """
    Worker function to read an image from disk, convert BGR -> RGB,
    resize to target_size, and normalize to [0.0, 1.0] (float32).
    """
    image_path, target_size = args
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    return img_norm

def load_dataset_split(file_list, labels, target_size=(32, 32), max_workers=8, desc="Processing images"):
    """
    Multithreaded loader for image files.
    """
    tasks = [(path, target_size) for path in file_list]
    print(f"Loading {len(tasks)} images ({desc})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(tqdm(executor.map(load_and_preprocess_single_image, tasks), total=len(tasks), desc=desc))
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

def collect_training_metadata(train_dir: str):
    """
    Gathers image file paths and class labels from class subfolders 00000 to 00042.
    """
    file_paths = []
    class_labels = []
    
    for class_id in range(43):
        class_str = f"{class_id:05d}"
        class_folder = os.path.join(train_dir, class_str)
        csv_path = os.path.join(class_folder, f"GT-{class_str}.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep=";")
            for row in df.itertuples(index=False):
                img_path = os.path.join(class_folder, row.Filename)
                if os.path.exists(img_path):
                    file_paths.append(img_path)
                    class_labels.append(int(row.ClassId))
                else:
                    print(f"Warning: Missing training image {img_path}")
        else:
            raise FileNotFoundError(f"Missing training annotation file: {csv_path}")
            
    return file_paths, class_labels

def collect_test_metadata(test_dir: str, test_csv_path: str):
    """
    Gathers test image file paths and class labels from GT-final_test.csv.
    """
    file_paths = []
    class_labels = []
    
    df_test = pd.read_csv(test_csv_path, sep=";")
    for row in df_test.itertuples(index=False):
        filename = row.Filename
        img_path = os.path.join(test_dir, filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(test_dir, os.path.basename(filename))
        if os.path.exists(img_path):
            file_paths.append(img_path)
            class_labels.append(int(row.ClassId))
        else:
            raise FileNotFoundError(f"Missing test image: {img_path}")
            
    return file_paths, class_labels

def generate_summary(total_images, num_classes, class_counts, train_count, val_count, test_count, output_paths):
    """
    Generates dataset_summary.txt containing formatted statistics and table.
    """
    lines = []
    lines.append("================================================================================")
    lines.append("           GTSRB DATASET PREPROCESSING AND SPLIT SUMMARY                        ")
    lines.append("================================================================================")
    lines.append("")
    lines.append(f"Total Images:      {total_images}")
    lines.append(f"Number of Classes: {num_classes}")
    lines.append(f"Training Count:    {train_count}")
    lines.append(f"Validation Count:  {val_count}")
    lines.append(f"Testing Count:     {test_count}")
    lines.append("")
    lines.append("Image Properties:")
    lines.append("  - Target Dimensions: 32x32 pixels")
    lines.append("  - Color Space:       RGB (3 channels)")
    lines.append("  - Normalization:     Pixel values normalized to [0.0, 1.0] (float32)")
    lines.append("  - Splits:            Stratified 80% Train / 20% Validation, Official Test Set")
    lines.append("")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"{'Class ID':<10} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8} {'Class Name'}")
    lines.append("--------------------------------------------------------------------------------")
    
    for class_id in range(num_classes):
        c_train = class_counts["train"].get(class_id, 0)
        c_val = class_counts["val"].get(class_id, 0)
        c_test = class_counts["test"].get(class_id, 0)
        c_total = c_train + c_val + c_test
        c_name = SIGN_NAMES.get(class_id, f"Class {class_id}")
        lines.append(f"{class_id:<10} {c_train:<8} {c_val:<8} {c_test:<8} {c_total:<8} {c_name}")
        
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"{'TOTAL':<10} {train_count:<8} {val_count:<8} {test_count:<8} {total_images:<8}")
    lines.append("================================================================================")
    
    summary_text = "\n".join(lines) + "\n"
    
    for path in output_paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"Saved dataset summary to: {path}")
        
    return summary_text

def main():
    parser = argparse.ArgumentParser(description="Prepare and preprocess the GTSRB dataset.")
    parser.add_argument("--raw-dir", type=str, default=r"dataset\raw", help="Path to raw dataset directory")
    parser.add_argument("--output-dir", type=str, default=r"dataset\processed", help="Path to save processed dataset")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio from training data (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split (default: 42)")
    parser.add_argument("--target-size", type=int, default=32, help="Target image size (default: 32 for 32x32)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads for image loading")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.abspath(os.path.join(project_root, args.raw_dir))
    output_dir = os.path.abspath(os.path.join(project_root, args.output_dir))
    
    train_dir = os.path.join(raw_dir, "GTSRB", "Training")
    test_dir = os.path.join(raw_dir, "GTSRB", "Final_Test", "Images")
    test_csv_path = os.path.join(raw_dir, "GT-final_test.csv")
    if not os.path.exists(test_csv_path):
        test_csv_path = os.path.join(raw_dir, "GTSRB", "GT-final_test.csv")
        
    print("=" * 70)
    print("GTSRB Dataset Preparation Starting")
    print(f"Project Root:     {project_root}")
    print(f"Raw Directory:    {raw_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Size:      ({args.target_size}, {args.target_size}) RGB")
    print(f"Random Seed:      {args.seed}")
    print(f"Validation Ratio: {args.val_split:.1%}")
    print("=" * 70)
    
    # 1. Collect Metadata
    train_paths, train_labels = collect_training_metadata(train_dir)
    test_paths, test_labels = collect_test_metadata(test_dir, test_csv_path)
    
    print(f"Found {len(train_paths)} raw training images across 43 classes.")
    print(f"Found {len(test_paths)} raw test images across 43 classes.")
    
    # 2. Stratified Train / Validation Split on Paths & Labels
    train_paths_split, val_paths_split, y_train, y_val = train_test_split(
        train_paths,
        train_labels,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=train_labels
    )
    y_test = np.array(test_labels, dtype=np.int64)
    y_train = np.array(y_train, dtype=np.int64)
    y_val = np.array(y_val, dtype=np.int64)
    
    print(f"Train split count:      {len(train_paths_split)}")
    print(f"Validation split count: {len(val_paths_split)}")
    print(f"Test split count:       {len(test_paths)}")
    
    # 3. Load & Process Images
    target_dim = (args.target_size, args.target_size)
    X_train, y_train = load_dataset_split(train_paths_split, y_train, target_size=target_dim, max_workers=args.workers, desc="Loading Train Images")
    X_val, y_val = load_dataset_split(val_paths_split, y_val, target_size=target_dim, max_workers=args.workers, desc="Loading Val Images")
    X_test, y_test = load_dataset_split(test_paths, y_test, target_size=target_dim, max_workers=args.workers, desc="Loading Test Images")
    
    # 4. Verify Shapes, Formats, and Ranges
    print("\nVerifying processed datasets...")
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}, range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"X_val shape:   {X_val.shape}, dtype: {X_val.dtype}, range: [{X_val.min():.3f}, {X_val.max():.3f}]")
    print(f"X_test shape:  {X_test.shape}, dtype: {X_test.dtype}, range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"Unique classes in train: {len(np.unique(y_train))}")
    print(f"Unique classes in val:   {len(np.unique(y_val))}")
    print(f"Unique classes in test:  {len(np.unique(y_test))}")
    
    assert X_train.shape == (len(train_paths_split), args.target_size, args.target_size, 3), "Invalid X_train shape"
    assert X_val.shape == (len(val_paths_split), args.target_size, args.target_size, 3), "Invalid X_val shape"
    assert X_test.shape == (len(test_paths), args.target_size, args.target_size, 3), "Invalid X_test shape"
    assert 0.0 <= X_train.min() and X_train.max() <= 1.0, "X_train values not normalized to [0, 1]"
    assert 0.0 <= X_val.min() and X_val.max() <= 1.0, "X_val values not normalized to [0, 1]"
    assert 0.0 <= X_test.min() and X_test.max() <= 1.0, "X_test values not normalized to [0, 1]"
    assert len(np.unique(y_train)) == 43, "Not all 43 classes present in training set"
    assert len(np.unique(y_val)) == 43, "Not all 43 classes present in validation set"
    assert len(np.unique(y_test)) == 43, "Not all 43 classes present in test set"
    
    # 5. Save Processed Dataset (.npz)
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "gtsrb_processed.npz")
    print(f"\nSaving processed dataset archive to {npz_path}...")
    np.savez_compressed(
        npz_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )
    npz_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"Compressed archive saved successfully ({npz_size_mb:.2f} MB).")
    
    # 6. Save Class Labels JSON
    labels_json_path = os.path.join(output_dir, "class_labels.json")
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump(SIGN_NAMES, f, indent=2)
    print(f"Saved class labels mapping to {labels_json_path}")
    
    # 7. Compute Counts and Generate Summary
    total_images = len(X_train) + len(X_val) + len(X_test)
    num_classes = 43
    
    class_counts = {
        "train": pd.Series(y_train).value_counts().to_dict(),
        "val": pd.Series(y_val).value_counts().to_dict(),
        "test": pd.Series(y_test).value_counts().to_dict(),
    }
    
    summary_file_root = os.path.join(project_root, "dataset_summary.txt")
    summary_file_processed = os.path.join(output_dir, "dataset_summary.txt")
    
    summary_text = generate_summary(
        total_images=total_images,
        num_classes=num_classes,
        class_counts=class_counts,
        train_count=len(X_train),
        val_count=len(X_val),
        test_count=len(X_test),
        output_paths=[summary_file_root, summary_file_processed]
    )
    
    print("\n" + summary_text)
    print("Dataset preparation completed successfully!")

if __name__ == "__main__":
    main()
