"""
train.py
========
PyTorch Baseline CNN Training Pipeline for Traffic Sign Recognition.

Key features:
1. Environment and Hardware Detection:
   - Prints PyTorch version, CUDA version, selected device, and GPU name.
2. Dataset loading and Preprocessing:
   - Loads preprocessed GTSRB data from `dataset/processed/gtsrb_processed.npz`.
   - Channels-first conversion (3, 32, 32) tensors.
   - Fast GPU batch augmentation on training data only (rotation, translation, brightness/contrast).
   - Evaluation on Validation and Test sets with no augmentation.
3. Class Imbalance Handling:
   - Computes balanced class weights derived strictly from the training set.
   - Applies class-weighted CrossEntropyLoss.
4. Model Architecture:
   - 3 Convolutional Blocks (Conv2D -> ReLU -> MaxPool2D).
   - Fully connected classifier with Dropout(0.5) and 43 output logits.
5. Training & Evaluation Lifecycle:
   - Pre-training sanity check batch test before running full 20 epochs.
   - Tracks train_loss, train_acc, val_loss, val_acc per epoch.
   - Checkpoints best model to `models/best_traffic_sign_cnn.pth`.
   - Records full history in `results/training_history.csv`.
   - Generates loss and accuracy curves in `results/`.
   - Final evaluation on Test Set once using the best checkpoint.
   - Exports test predictions to `results/test_predictions.npz`.
   - Logs configuration to `results/training_config.txt`.
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import v2

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class TrafficSignCNN(nn.Module):
    """
    Baseline Convolutional Neural Network for Traffic Sign Recognition.
    Architecture:
      Input: (3, 32, 32)
      Conv2D(3, 32, k=3, p=1) -> ReLU -> MaxPool2D(2)     => (32, 16, 16)
      Conv2D(32, 64, k=3, p=1) -> ReLU -> MaxPool2D(2)    => (64, 8, 8)
      Conv2D(64, 128, k=3, p=1) -> ReLU -> MaxPool2D(2)   => (128, 4, 4)
      Flatten                                             => (2048)
      Linear(2048, 512) -> ReLU -> Dropout(0.5)
      Linear(512, 43)                                     => 43 classes
    """
    def __init__(self, num_classes=43):
        super(TrafficSignCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class FastTrainAugmentation(nn.Module):
    """
    GPU-accelerated batch data augmentation applied strictly during training:
    - Small random rotation: +/- 10 degrees
    - Small translation: +/- 8%
    - Slight brightness/contrast variation: +/- 15%
    """
    def __init__(self):
        super().__init__()
        self.geometric = v2.Compose([
            v2.RandomRotation(degrees=(-10, 10)),
            v2.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        ])

    def forward(self, x):
        # 1. Geometric transforms on batch
        x = self.geometric(x)
        # 2. Photometric jitter on batch
        B = x.size(0)
        contrast_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.3
        brightness_shift = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.2
        x = (x - 0.5) * contrast_factor + 0.5 + brightness_shift
        return torch.clamp(x, 0.0, 1.0)

def compute_class_weights(y_train: np.ndarray, num_classes: int = 43) -> torch.Tensor:
    """
    Computes balanced class weights strictly from training labels:
    weight[c] = total_samples / (num_classes * count[c])
    Normalized so sum(weights) == num_classes.
    """
    counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    weights = total_samples / (num_classes * counts.astype(np.float32))
    weights = weights / np.sum(weights) * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def run_sanity_check(model, device, sanity_loader, criterion, augmentation):
    """
    Runs a single small batch sanity check to verify:
    - Image tensor loading & shapes
    - Forward pass
    - Loss calculation
    - CUDA device execution
    """
    print("\n" + "=" * 60)
    print("RUNNING PRE-TRAINING SANITY CHECK")
    print("=" * 60)
    
    model.eval()
    for batch_idx, (images, targets) in enumerate(sanity_loader):
        images, targets = images.to(device), targets.to(device)
        print(f"  Input batch tensor shape: {images.shape} (dtype: {images.dtype})")
        print(f"  Target batch tensor shape: {targets.shape} (dtype: {targets.dtype})")
        print(f"  Batch device:             {images.device}")
        
        with torch.no_grad():
            aug_images = augmentation(images)
            outputs = model(aug_images)
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)
            correct = (preds == targets).sum().item()
            
        print(f"  Augmented batch shape:    {aug_images.shape}")
        print(f"  Forward pass output shape: {outputs.shape}")
        print(f"  Loss value computed:      {loss.item():.4f}")
        print(f"  Sanity batch accuracy:    {correct / len(targets):.1%}")
        break
        
    print("SANITY CHECK PASSED SUCCESSFULLY! Ready for training.\n")
    return True

def train_one_epoch(model, dataloader, criterion, optimizer, device, augmentation):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, targets in dataloader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        images = augmentation(images)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * targets.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device, return_predictions=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * targets.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            if return_predictions:
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
    avg_loss = running_loss / total
    avg_acc = correct / total
    
    if return_predictions:
        return avg_loss, avg_acc, np.array(all_targets), np.array(all_preds), np.array(all_probs)
    return avg_loss, avg_acc

def plot_curves(history_df, results_dir):
    epochs = history_df["epoch"]
    
    # 1. Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history_df["train_acc"] * 100, label="Training Accuracy", marker="o", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history_df["val_acc"] * 100, label="Validation Accuracy", marker="s", color="#ff7f0e", linewidth=2)
    plt.title("Traffic Sign Recognition - Training & Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()
    acc_path = os.path.join(results_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=200)
    plt.close()
    
    # 2. Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history_df["train_loss"], label="Training Loss", marker="o", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history_df["val_loss"], label="Validation Loss", marker="s", color="#ff7f0e", linewidth=2)
    plt.title("Traffic Sign Recognition - Training & Validation Loss", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (CrossEntropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    loss_path = os.path.join(results_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=200)
    plt.close()
    
    print(f"Accuracy curve saved to: {acc_path}")
    print(f"Loss curve saved to:     {loss_path}")

def main():
    parser = argparse.ArgumentParser(description="Train baseline CNN for GTSRB.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    npz_path = os.path.join(project_root, "dataset", "processed", "gtsrb_processed.npz")
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    # Hardware & Environment Detection
    print("=" * 70)
    print("TRAFFIC SIGN RECOGNITION USING NEURAL NETWORKS")
    print("=" * 70)
    pytorch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A (CPU only)"
    
    print(f"PyTorch Version:  {pytorch_version}")
    print(f"CUDA Version:     {cuda_version}")
    print(f"Selected Device:  {device}")
    print(f"GPU Name:         {gpu_name}")
    print("=" * 70)
    
    # Load Processed Dataset
    print(f"Loading processed dataset from {npz_path}...")
    dataset_npz = np.load(npz_path)
    X_train_raw = dataset_npz["X_train"]  # (21312, 32, 32, 3)
    y_train_raw = dataset_npz["y_train"]
    X_val_raw = dataset_npz["X_val"]      # (5328, 32, 32, 3)
    y_val_raw = dataset_npz["y_val"]
    X_test_raw = dataset_npz["X_test"]    # (12630, 32, 32, 3)
    y_test_raw = dataset_npz["y_test"]
    
    # Convert from (N, 32, 32, 3) to (N, 3, 32, 32) tensors
    print("Converting arrays to PyTorch tensors (CHW channels-first)...")
    X_train = torch.from_numpy(X_train_raw).permute(0, 3, 1, 2).float()
    y_train = torch.from_numpy(y_train_raw).long()
    X_val = torch.from_numpy(X_val_raw).permute(0, 3, 1, 2).float()
    y_val = torch.from_numpy(y_val_raw).long()
    X_test = torch.from_numpy(X_test_raw).permute(0, 3, 1, 2).float()
    y_test = torch.from_numpy(y_test_raw).long()
    
    print(f"Training split:   {X_train.shape[0]} images, shape: {X_train.shape}")
    print(f"Validation split: {X_val.shape[0]} images, shape: {X_val.shape}")
    print(f"Testing split:    {X_test.shape[0]} images, shape: {X_test.shape} (reserved for final evaluation)")
    
    # Compute Class Weights (ONLY from Training Set)
    class_weights = compute_class_weights(y_train_raw, num_classes=43).to(device)
    print("Computed class weights from training set (min: {:.2f}, max: {:.2f})".format(
        class_weights.min().item(), class_weights.max().item()
    ))
    
    # GPU Data Augmentation Module (TRAIN ONLY)
    train_augmentation = FastTrainAugmentation().to(device)
    
    # Datasets and Loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=cuda_available)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=cuda_available)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=cuda_available)
    
    # Sanity Loader (unshuffled batch of 16)
    sanity_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    
    # Initialize Model, Loss, Optimizer
    model = TrafficSignCNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Sanity check
    run_sanity_check(model, device, sanity_loader, criterion, train_augmentation)
    
    # Training Loop
    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(models_dir, "best_traffic_sign_cnn.pth")
    history = []
    
    print("=" * 70)
    print(f"STARTING FULL TRAINING RUN ({args.epochs} EPOCHS, BATCH SIZE {args.batch_size})")
    print("=" * 70)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Best':<6} {'Time'}")
    print("-" * 70)
    
    start_total_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, train_augmentation)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        is_best = val_acc > best_val_acc
        
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "class_weights": class_weights.cpu()
            }, best_model_path)
            
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time
        })
        
        star = "*" if is_best else " "
        print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.2%} {val_loss:<12.4f} {val_acc:<12.2%} {star:<6} {epoch_time:.2f}s")
        
    total_train_time = time.time() - start_total_time
    print("-" * 70)
    print(f"Training completed in {total_train_time:.2f} seconds ({total_train_time/60:.2f} mins).")
    print(f"Best Validation Accuracy: {best_val_acc:.2%} (Epoch {best_epoch})")
    print(f"Best model checkpoint saved to: {best_model_path}")
    
    # Save History CSV
    history_df = pd.DataFrame(history)
    history_path = os.path.join(results_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")
    
    # Generate Training Curves
    plot_curves(history_df, results_dir)
    
    # Evaluate ONCE on Test Set using Best Checkpoint
    print("\n" + "=" * 70)
    print("EVALUATING BEST MODEL ON THE UNSEEN TEST SET")
    print("=" * 70)
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(
        model, test_loader, criterion, device, return_predictions=True
    )
    
    print(f"Test Set Loss:     {test_loss:.4f}")
    print(f"Test Set Accuracy: {test_acc:.2%}")
    print("=" * 70)
    
    # Save Test Predictions for future confusion matrix
    pred_path = os.path.join(results_dir, "test_predictions.npz")
    np.savez_compressed(
        pred_path,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=y_prob
    )
    print(f"Saved test predictions and true labels to: {pred_path}")
    
    # Final training accuracy from last epoch
    final_train_acc = history_df["train_acc"].iloc[-1]
    
    # Save Complete Config & Parameters
    config_path = os.path.join(results_dir, "training_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("TRAFFIC SIGN RECOGNITION - TRAINING CONFIGURATION & PARAMETERS\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Project Title:            Traffic Sign Recognition Using Neural Networks\n")
        f.write(f"PyTorch Version:          {pytorch_version}\n")
        f.write(f"CUDA Version:             {cuda_version}\n")
        f.write(f"Device:                   {device}\n")
        f.write(f"GPU Name:                 {gpu_name}\n\n")
        f.write(f"Architecture:\n")
        f.write(f"  Input:                  (3, 32, 32)\n")
        f.write(f"  Conv Block 1:           Conv2D(3, 32, k=3, p=1) -> ReLU -> MaxPool2D(2)\n")
        f.write(f"  Conv Block 2:           Conv2D(32, 64, k=3, p=1) -> ReLU -> MaxPool2D(2)\n")
        f.write(f"  Conv Block 3:           Conv2D(64, 128, k=3, p=1) -> ReLU -> MaxPool2D(2)\n")
        f.write(f"  Classifier:             Flatten (2048) -> Linear(2048, 512) -> ReLU -> Dropout(0.5) -> Linear(512, 43)\n\n")
        f.write(f"Training Hyperparameters:\n")
        f.write(f"  Batch Size:             {args.batch_size}\n")
        f.write(f"  Learning Rate:          {args.lr}\n")
        f.write(f"  Epochs:                 {args.epochs}\n")
        f.write(f"  Optimizer:              Adam (lr={args.lr})\n")
        f.write(f"  Loss Function:          Class-Weighted CrossEntropyLoss (Train Split Weights Only)\n")
        f.write(f"  Random Seed:            {args.seed}\n")
        f.write(f"  Data Augmentation:      RandomRotation(±10°), RandomAffine(translate 0.08), ColorJitter(0.15)\n\n")
        f.write(f"Dataset Counts:\n")
        f.write(f"  Training:               {len(X_train)} images\n")
        f.write(f"  Validation:             {len(X_val)} images\n")
        f.write(f"  Testing:                {len(X_test)} images\n")
        f.write(f"  Total:                  {len(X_train) + len(X_val) + len(X_test)} images\n\n")
        f.write(f"Results Summary:\n")
        f.write(f"  Final Training Acc:     {final_train_acc:.2%}\n")
        f.write(f"  Best Validation Acc:    {best_val_acc:.2%} (Epoch {best_epoch})\n")
        f.write(f"  Test Set Loss:          {test_loss:.4f}\n")
        f.write(f"  Test Set Accuracy:      {test_acc:.2%}\n")
        f.write(f"  Total Training Time:    {total_train_time:.2f} seconds ({total_train_time/60:.2f} minutes)\n")
        f.write(f"  Best Model Checkpoint:  {best_model_path}\n")
        f.write(f"  History CSV:            {history_path}\n")
        f.write(f"  Accuracy Plot:          {os.path.join(results_dir, 'accuracy_curve.png')}\n")
        f.write(f"  Loss Plot:              {os.path.join(results_dir, 'loss_curve.png')}\n")
        f.write(f"  Test Predictions:       {pred_path}\n")
    print(f"Training configuration and parameters saved to: {config_path}")

if __name__ == "__main__":
    main()
