"""
train_improved.py
=================
Improved PyTorch CNN Training Pipeline for Traffic Sign Recognition.

Key Improvements over Baseline:
1. Batch Normalization:
   - Added BatchNorm2d after every Conv2D layer to stabilize internal representations,
     accelerate training, and act as a regularizer.
   - Added BatchNorm1d after the dense feature layer.
2. Regularization & Dropout:
   - Graduated dropout in convolutional blocks (0.1, 0.1, 0.2) to prevent co-adaptation.
   - Retained Dropout(0.5) in the fully-connected classification layer.
   - Added weight decay (L2 regularization, 1e-4) to Adam optimizer.
3. Safe Data Augmentation (TRAIN ONLY):
   - Small random rotation (±12°)
   - Small translation (±10%)
   - Safe scaling/zoom (0.9 to 1.1)
   - Photometric jitter (brightness ±20%, contrast ±20%)
   - NO horizontal or vertical flipping (preserves directional/regulatory meaning).
4. Class-Weighted CrossEntropyLoss:
   - Derived strictly from training set labels to handle class imbalance.
5. Learning Rate Scheduling:
   - CosineAnnealingLR smoothly decays learning rate from 0.001 to 1e-5 over 20 epochs.
6. Checkpointing:
   - Saves model with the highest validation accuracy to `models/best_traffic_sign_cnn_improved.pth`.
7. Evaluation on Unseen Test Set:
   - Evaluated strictly ONCE using the best saved checkpoint.
   - Generates predictions, confusion matrix, classification report, and per-class accuracy.
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
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ImprovedTrafficSignCNN(nn.Module):
    """
    Improved CNN architecture with Batch Normalization and regularized Dropout.
    
    Structure:
      Input: (3, 32, 32)
      Block 1: Conv2D(3, 32, k=3, p=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2D(2) -> Dropout(0.1)
      Block 2: Conv2D(32, 64, k=3, p=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2D(2) -> Dropout(0.1)
      Block 3: Conv2D(64, 128, k=3, p=1) -> BatchNorm2d(128) -> ReLU -> MaxPool2D(2) -> Dropout(0.2)
      Flatten (2048)
      Dense: Linear(2048, 512) -> BatchNorm1d(512) -> ReLU -> Dropout(0.5)
      Output: Linear(512, 43)
    """
    def __init__(self, num_classes=43):
        super(ImprovedTrafficSignCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SafeTrafficSignAugmentation(nn.Module):
    """
    Safe GPU batch data augmentation:
    - Random rotation (±12°)
    - Random translation (±10%)
    - Random scale (0.90 to 1.10)
    - Brightness and contrast variation (±20%)
    - NO horizontal/vertical flipping (preserves sign meaning)
    """
    def __init__(self):
        super().__init__()
        self.geometric = v2.Compose([
            v2.RandomRotation(degrees=(-12, 12)),
            v2.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.90, 1.10)),
        ])

    def forward(self, x):
        x = self.geometric(x)
        B = x.size(0)
        contrast_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.4
        brightness_shift = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.3
        x = (x - 0.5) * contrast_factor + 0.5 + brightness_shift
        return torch.clamp(x, 0.0, 1.0)

def compute_class_weights(y_train: np.ndarray, num_classes: int = 43) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    weights = total_samples / (num_classes * counts.astype(np.float32))
    weights = weights / np.sum(weights) * num_classes
    return torch.tensor(weights, dtype=torch.float32)

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
        
    return running_loss / total, correct / total

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
    plt.plot(epochs, history_df["train_acc"] * 100, label="Training Accuracy", marker="o", color="#2ca02c", linewidth=2)
    plt.plot(epochs, history_df["val_acc"] * 100, label="Validation Accuracy", marker="s", color="#1f77b4", linewidth=2)
    plt.title("Improved CNN - Training & Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()
    acc_path = os.path.join(results_dir, "improved_accuracy_curve.png")
    plt.savefig(acc_path, dpi=200)
    plt.close()
    
    # 2. Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history_df["train_loss"], label="Training Loss", marker="o", color="#2ca02c", linewidth=2)
    plt.plot(epochs, history_df["val_loss"], label="Validation Loss", marker="s", color="#1f77b4", linewidth=2)
    plt.title("Improved CNN - Training & Validation Loss", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (CrossEntropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    loss_path = os.path.join(results_dir, "improved_loss_curve.png")
    plt.savefig(loss_path, dpi=200)
    plt.close()
    
    print(f"Saved: {acc_path}")
    print(f"Saved: {loss_path}")

def plot_confusion_matrix(cm, class_ids, output_path, title):
    plt.figure(figsize=(14, 12))
    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Image Count", fontsize=11)
    tick_marks = np.arange(len(class_ids))
    plt.xticks(tick_marks, class_ids, fontsize=8, rotation=90)
    plt.yticks(tick_marks, class_ids, fontsize=8)
    plt.xlabel("Predicted Class ID", fontsize=13, labelpad=10)
    plt.ylabel("Actual Class ID", fontsize=13, labelpad=10)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train improved CNN for GTSRB.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    npz_path = os.path.join(project_root, "dataset", "processed", "gtsrb_processed.npz")
    labels_json_path = os.path.join(project_root, "dataset", "processed", "class_labels.json")
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    # Hardware & Environment Detection
    print("=" * 70)
    print("IMPROVED TRAFFIC SIGN RECOGNITION (CNN + BATCHNORM + SCHEDULER)")
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
    print(f"Loading dataset from {npz_path}...")
    dataset_npz = np.load(npz_path)
    X_train_raw = dataset_npz["X_train"]
    y_train_raw = dataset_npz["y_train"]
    X_val_raw = dataset_npz["X_val"]
    y_val_raw = dataset_npz["y_val"]
    X_test_raw = dataset_npz["X_test"]
    y_test_raw = dataset_npz["y_test"]
    
    import json
    with open(labels_json_path, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    target_names = [label_map[i] for i in range(43)]
    
    # Convert arrays to PyTorch tensors (CHW)
    X_train = torch.from_numpy(X_train_raw).permute(0, 3, 1, 2).float()
    y_train = torch.from_numpy(y_train_raw).long()
    X_val = torch.from_numpy(X_val_raw).permute(0, 3, 1, 2).float()
    y_val = torch.from_numpy(y_val_raw).long()
    X_test = torch.from_numpy(X_test_raw).permute(0, 3, 1, 2).float()
    y_test = torch.from_numpy(y_test_raw).long()
    
    print(f"Training split:   {X_train.shape[0]} images")
    print(f"Validation split: {X_val.shape[0]} images")
    print(f"Testing split:    {X_test.shape[0]} images (strictly reserved for final evaluation)")
    
    # Class weights strictly from training set
    class_weights = compute_class_weights(y_train_raw, num_classes=43).to(device)
    
    # Augmentation and Loaders
    train_augmentation = SafeTrafficSignAugmentation().to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=cuda_available)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=cuda_available)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=cuda_available)
    
    # Model, Criterion, Optimizer, Scheduler
    model = ImprovedTrafficSignCNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # Training Loop
    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(models_dir, "best_traffic_sign_cnn_improved.pth")
    history = []
    
    print("\n" + "=" * 70)
    print(f"STARTING IMPROVED TRAINING RUN ({args.epochs} EPOCHS, BATCH SIZE {args.batch_size})")
    print("=" * 70)
    print(f"{'Epoch':<7} {'Train Loss':<11} {'Train Acc':<11} {'Val Loss':<11} {'Val Acc':<11} {'LR':<10} {'Best':<5} {'Time'}")
    print("-" * 75)
    
    start_total_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, train_augmentation)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
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
            "lr": current_lr,
            "epoch_time_sec": epoch_time
        })
        
        star = "*" if is_best else " "
        print(f"{epoch:<7} {train_loss:<11.4f} {train_acc:<11.2%} {val_loss:<11.4f} {val_acc:<11.2%} {current_lr:<10.6f} {star:<5} {epoch_time:.2f}s")
        
    total_train_time = time.time() - start_total_time
    print("-" * 75)
    print(f"Training completed in {total_train_time:.2f} seconds ({total_train_time/60:.2f} mins).")
    print(f"Best Validation Accuracy: {best_val_acc:.2%} (Epoch {best_epoch})")
    print(f"Best improved model checkpoint saved to: {best_model_path}")
    
    # Save History CSV
    history_df = pd.DataFrame(history)
    history_path = os.path.join(results_dir, "improved_training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Saved: {history_path}")
    
    # Generate Training Curves
    plot_curves(history_df, results_dir)
    
    # Evaluate ONCE on Unseen Test Set using Best Checkpoint
    print("\n" + "=" * 70)
    print("EVALUATING IMPROVED MODEL ON UNSEEN TEST SET")
    print("=" * 70)
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(
        model, test_loader, criterion, device, return_predictions=True
    )
    
    # Metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Save Test Predictions
    pred_path = os.path.join(results_dir, "improved_test_predictions.npz")
    np.savez_compressed(
        pred_path,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=y_prob
    )
    print(f"Saved: {pred_path}")
    
    # Save Classification Report
    report_str = classification_report(
        y_true, y_pred, labels=list(range(43)), target_names=target_names, digits=4, zero_division=0
    )
    report_output_path = os.path.join(results_dir, "improved_classification_report.txt")
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write("              IMPROVED GTSRB CLASSIFICATION REPORT (UNSEEN TEST SET)\n")
        f.write("=" * 85 + "\n\n")
        f.write(report_str + "\n\n")
        f.write("-" * 85 + "\n")
        f.write("GLOBAL SUMMARY METRICS\n")
        f.write("-" * 85 + "\n")
        f.write(f"Total Test Samples:       {len(y_true)}\n")
        f.write(f"Overall Test Accuracy:    {test_acc:.4f} ({test_acc * 100:.2f}%)\n")
        f.write(f"Test Loss:                {test_loss:.4f}\n")
        f.write(f"Macro Precision:          {precision_macro:.4f}\n")
        f.write(f"Macro Recall:             {recall_macro:.4f}\n")
        f.write(f"Macro F1-Score:           {f1_macro:.4f}\n")
        f.write(f"Weighted Precision:       {precision_weighted:.4f}\n")
        f.write(f"Weighted Recall:          {recall_weighted:.4f}\n")
        f.write(f"Weighted F1-Score:        {f1_weighted:.4f}\n")
        f.write("=" * 85 + "\n")
    print(f"Saved: {report_output_path}")
    
    # Save Per-Class Accuracy CSV
    per_class_records = []
    for c in range(43):
        mask = (y_true == c)
        c_support = int(mask.sum())
        c_correct = int((y_pred[mask] == c).sum())
        c_acc = (c_correct / c_support) if c_support > 0 else 0.0
        c_prec = ((y_pred == c) & (y_true == c)).sum() / ((y_pred == c).sum() + 1e-9)
        c_f1 = (2 * c_prec * c_acc) / (c_prec + c_acc + 1e-9)
        per_class_records.append({
            "class_id": c,
            "class_name": label_map[c],
            "support": c_support,
            "correct": c_correct,
            "accuracy": round(c_acc, 4),
            "precision": round(c_prec, 4),
            "f1_score": round(c_f1, 4)
        })
    df_per_class = pd.DataFrame(per_class_records)
    per_class_path = os.path.join(results_dir, "improved_per_class_accuracy.csv")
    df_per_class.to_csv(per_class_path, index=False)
    print(f"Saved: {per_class_path}")
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(43)))
    cm_path = os.path.join(results_dir, "improved_confusion_matrix.png")
    plot_confusion_matrix(cm, [str(i) for i in range(43)], cm_path, "Improved GTSRB Test Set Confusion Matrix")
    
    # Save Config
    final_train_acc = history_df["train_acc"].iloc[-1]
    config_path = os.path.join(results_dir, "improved_training_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("IMPROVED TRAFFIC SIGN RECOGNITION - TRAINING CONFIGURATION & PARAMETERS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Project Title:            Traffic Sign Recognition Using Neural Networks\n")
        f.write(f"Model Type:               Improved CNN (BatchNorm + Dropout + CosineAnnealingLR)\n")
        f.write(f"PyTorch Version:          {pytorch_version}\n")
        f.write(f"CUDA Version:             {cuda_version}\n")
        f.write(f"Device:                   {device}\n")
        f.write(f"GPU Name:                 {gpu_name}\n\n")
        f.write("Architecture:\n")
        f.write("  Block 1: Conv2D(3, 32, k=3, p=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2D(2) -> Dropout(0.1)\n")
        f.write("  Block 2: Conv2D(32, 64, k=3, p=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2D(2) -> Dropout(0.1)\n")
        f.write("  Block 3: Conv2D(64, 128, k=3, p=1) -> BatchNorm2d(128) -> ReLU -> MaxPool2D(2) -> Dropout(0.2)\n")
        f.write("  Classifier: Flatten (2048) -> Linear(2048, 512) -> BatchNorm1d(512) -> ReLU -> Dropout(0.5) -> Linear(512, 43)\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"  Batch Size:             {args.batch_size}\n")
        f.write(f"  Learning Rate:          {args.lr}\n")
        f.write(f"  Weight Decay:           1e-4\n")
        f.write(f"  Epochs:                 {args.epochs}\n")
        f.write(f"  Optimizer:              Adam (lr={args.lr}, weight_decay=1e-4)\n")
        f.write(f"  LR Scheduler:           CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-5)\n")
        f.write(f"  Loss Function:          Class-Weighted CrossEntropyLoss (Train Split Only)\n")
        f.write(f"  Random Seed:            {args.seed}\n")
        f.write("  Data Augmentation:      RandomRotation(+/-12 deg), RandomAffine(translate 0.10, scale 0.9-1.1), ColorJitter(0.20)\n\n")
        f.write("Measured Results Summary:\n")
        f.write(f"  Final Training Acc:     {final_train_acc:.2%}\n")
        f.write(f"  Best Validation Acc:    {best_val_acc:.2%} (Epoch {best_epoch})\n")
        f.write(f"  Test Set Loss:          {test_loss:.4f}\n")
        f.write(f"  Test Set Accuracy:      {test_acc:.2%}\n")
        f.write(f"  Macro Precision:        {precision_macro:.4f}\n")
        f.write(f"  Macro Recall:           {recall_macro:.4f}\n")
        f.write(f"  Macro F1-Score:         {f1_macro:.4f}\n")
        f.write(f"  Weighted Precision:     {precision_weighted:.4f}\n")
        f.write(f"  Weighted Recall:        {recall_weighted:.4f}\n")
        f.write(f"  Weighted F1-Score:      {f1_weighted:.4f}\n")
        f.write(f"  Total Training Time:    {total_train_time:.2f} seconds ({total_train_time/60:.2f} minutes)\n")
    print(f"Saved: {config_path}")
    
    # Print Required Metrics
    print("\n" + "=" * 70)
    print("IMPROVED MODEL FINAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"Final training accuracy:          {final_train_acc:.2%}")
    print(f"Best validation accuracy and epoch: {best_val_acc:.2%} (Epoch {best_epoch})")
    print(f"Test accuracy:                    {test_acc:.2%}")
    print(f"Test loss:                        {test_loss:.4f}")
    print(f"Macro precision:                  {precision_macro:.4f}")
    print(f"Macro recall:                     {recall_macro:.4f}")
    print(f"Macro F1:                         {f1_macro:.4f}")
    print(f"Weighted precision:               {precision_weighted:.4f}")
    print(f"Weighted recall:                  {recall_weighted:.4f}")
    print(f"Weighted F1:                      {f1_weighted:.4f}")
    print(f"Training time:                    {total_train_time:.2f}s ({total_train_time/60:.2f} mins)")
    print("=" * 70)

if __name__ == "__main__":
    main()
