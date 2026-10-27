"""
evaluate_improved.py
====================
Detailed error analysis and evaluation pipeline for the IMPROVED CNN model.

Uses ONLY:
- Saved checkpoint: models/best_traffic_sign_cnn_improved.pth
- Unseen test predictions: results/improved_test_predictions.npz
- Unseen test images: dataset/processed/gtsrb_processed.npz ("X_test")
- Label names: dataset/processed/class_labels.json

Generates:
1. results/improved_confusion_matrix_normalized.png (normalized by actual class)
2. results/improved_top_confusions.csv (top 10 actual -> predicted pairs, count, % of actual)
3. results/improved_misclassified_samples.png (representative incorrect test images with actual, pred, confidence)
4. results/improved_per_class_accuracy.csv (class ID, class name, correct predictions, total samples, accuracy)
5. results/improved_classification_report.txt (precision, recall, F1-score, support for all 43 classes + summary)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

def plot_normalized_confusion_matrix(cm_norm, class_ids, output_path, title="GTSRB Improved CNN - Normalized Confusion Matrix (Recall)"):
    """
    Renders and saves a high-resolution normalized confusion matrix heatmap.
    Normalized by actual class (row-normalized).
    """
    plt.figure(figsize=(15, 13))
    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    
    im = plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Proportion of True Class (Recall)", fontsize=12)
    
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

def plot_misclassified_samples(X_test, y_true, y_pred, y_prob, label_map, output_path, n_samples=16):
    """
    Plots a high-quality grid of representative misclassified test images showing Actual, Predicted, and Confidence.
    """
    mis_indices = np.where(y_true != y_pred)[0]
    
    # Pick diverse misclassifications across different (actual, predicted) pairs
    seen_pairs = set()
    selected_indices = []
    
    for idx in mis_indices:
        pair = (int(y_true[idx]), int(y_pred[idx]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            selected_indices.append(idx)
        if len(selected_indices) == n_samples:
            break
            
    if len(selected_indices) < n_samples:
        for idx in mis_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) == n_samples:
                break
                
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    fig.suptitle("Representative Misclassified Test Images - Improved CNN\n(Actual Class vs. Predicted Class & Model Confidence)",
                 fontsize=16, fontweight="bold", y=0.98)
    
    for ax, idx in zip(axes.flat, selected_indices):
        img = X_test[idx]
        actual_id = int(y_true[idx])
        pred_id = int(y_pred[idx])
        conf = float(y_prob[idx, pred_id]) * 100.0
        
        actual_name = label_map.get(actual_id, f"Class {actual_id}")
        pred_name = label_map.get(pred_id, f"Class {pred_id}")
        
        if len(actual_name) > 23:
            actual_name = actual_name[:21] + "..."
        if len(pred_name) > 23:
            pred_name = pred_name[:21] + "..."
            
        ax.imshow(img)
        title_text = f"Actual [{actual_id}]: {actual_name}\nPred [{pred_id}]: {pred_name}\nConf: {conf:.1f}%"
        ax.set_title(title_text, fontsize=9.5, pad=6, color="#b30000", fontweight="semibold")
        ax.axis("off")
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pred_path = os.path.join(project_root, "results", "improved_test_predictions.npz")
    npz_data_path = os.path.join(project_root, "dataset", "processed", "gtsrb_processed.npz")
    labels_json_path = os.path.join(project_root, "dataset", "processed", "class_labels.json")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 75)
    print("IMPROVED CNN: DETAILED ERROR ANALYSIS & EVALUATION ON UNSEEN TEST SET")
    print("=" * 75)
    print(f"Loading predictions from: {pred_path}")
    
    # 1. Load predictions
    pred_data = np.load(pred_path)
    y_true = pred_data["y_true"]
    y_pred = pred_data["y_pred"]
    y_prob = pred_data["probabilities"]
    
    # 2. Load test images
    dataset_npz = np.load(npz_data_path)
    X_test = dataset_npz["X_test"]
    
    # 3. Load class labels
    with open(labels_json_path, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
        
    num_classes = 43
    class_ids = [str(i) for i in range(num_classes)]
    target_names = [label_map[i] for i in range(num_classes)]
    
    # Core Overall Metrics
    overall_acc = accuracy_score(y_true, y_pred)
    total_test = len(y_true)
    correct_count = int((y_true == y_pred).sum())
    incorrect_count = total_test - correct_count
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    print(f"Total Test Samples:       {total_test}")
    print(f"Correct Predictions:      {correct_count}")
    print(f"Incorrect Predictions:    {incorrect_count}")
    print(f"Overall Test Accuracy:    {overall_acc:.4%} ({overall_acc * 100:.2f}%)")
    print(f"Macro F1-Score:           {f1_macro:.4f} ({f1_macro * 100:.2f}%)")
    print(f"Weighted F1-Score:        {f1_weighted:.4f} ({f1_weighted * 100:.2f}%)")
    print("-" * 75)
    
    # 4. Generate & Save Normalized Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    norm_cm_path = os.path.join(results_dir, "improved_confusion_matrix_normalized.png")
    plot_normalized_confusion_matrix(cm_norm, class_ids, norm_cm_path)
    
    # 5. Generate & Save Top 10 Confusion Pairs CSV
    confusions = []
    for t, p in zip(y_true, y_pred):
        if t != p:
            confusions.append((int(t), int(p)))
            
    top_counts = Counter(confusions).most_common(10)
    top_records = []
    for rank, ((t, p), count) in enumerate(top_counts, 1):
        support_t = int((y_true == t).sum())
        pct_of_true = (count / support_t) * 100.0 if support_t > 0 else 0.0
        top_records.append({
            "rank": rank,
            "actual_class_id": t,
            "actual_class_name": label_map[t],
            "predicted_class_id": p,
            "predicted_class_name": label_map[p],
            "count": count,
            "percentage_of_actual_class": round(pct_of_true, 2)
        })
        
    df_top = pd.DataFrame(top_records)
    top_conf_path = os.path.join(results_dir, "improved_top_confusions.csv")
    df_top.to_csv(top_conf_path, index=False)
    print(f"Saved: {top_conf_path}")
    
    # 6. Generate & Save Representative Misclassified Samples Grid
    mis_samples_path = os.path.join(results_dir, "improved_misclassified_samples.png")
    plot_misclassified_samples(X_test, y_true, y_pred, y_prob, label_map, mis_samples_path, n_samples=16)
    
    # 7. Generate & Save Per-Class Accuracy CSV
    per_class_records = []
    for c in range(num_classes):
        mask = (y_true == c)
        c_support = int(mask.sum())
        c_correct = int((y_pred[mask] == c).sum())
        c_acc = (c_correct / c_support) if c_support > 0 else 0.0
        
        c_pred_mask = (y_pred == c)
        c_prec = (c_correct / c_pred_mask.sum()) if c_pred_mask.sum() > 0 else 0.0
        c_f1 = (2 * c_prec * c_acc) / (c_prec + c_acc) if (c_prec + c_acc) > 0 else 0.0
        
        per_class_records.append({
            "class_id": c,
            "class_name": label_map[c],
            "correct_predictions": c_correct,
            "total_samples": c_support,
            "accuracy": round(c_acc, 4),
            "precision": round(c_prec, 4),
            "f1_score": round(c_f1, 4)
        })
        
    df_per_class = pd.DataFrame(per_class_records)
    per_class_path = os.path.join(results_dir, "improved_per_class_accuracy.csv")
    df_per_class.to_csv(per_class_path, index=False)
    print(f"Saved: {per_class_path}")
    
    # 8. Update Classification Report TXT
    report_str = classification_report(
        y_true, y_pred, labels=list(range(num_classes)), target_names=target_names, digits=4, zero_division=0
    )
    report_output_path = os.path.join(results_dir, "improved_classification_report.txt")
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write("=" * 88 + "\n")
        f.write("          IMPROVED CNN: GTSRB CLASSIFICATION REPORT (UNSEEN TEST SET)\n")
        f.write("=" * 88 + "\n\n")
        f.write(report_str + "\n\n")
        f.write("-" * 88 + "\n")
        f.write("GLOBAL SUMMARY METRICS\n")
        f.write("-" * 88 + "\n")
        f.write(f"Total Test Samples:       {total_test}\n")
        f.write(f"Correct Predictions:      {correct_count}\n")
        f.write(f"Incorrect Predictions:    {incorrect_count}\n")
        f.write(f"Overall Test Accuracy:    {overall_acc:.4f} ({overall_acc * 100:.2f}%)\n")
        f.write(f"Macro Precision:          {precision_macro:.4f}\n")
        f.write(f"Macro Recall:             {recall_macro:.4f}\n")
        f.write(f"Macro F1-Score:           {f1_macro:.4f}\n")
        f.write(f"Weighted Precision:       {precision_weighted:.4f}\n")
        f.write(f"Weighted Recall:          {recall_weighted:.4f}\n")
        f.write(f"Weighted F1-Score:        {f1_weighted:.4f}\n")
        f.write("=" * 88 + "\n")
    print(f"Saved: {report_output_path}")
    
    # 9. Print Required Outputs
    print("\n" + "=" * 75)
    print("DETAILED ERROR ANALYSIS & METRICS REPORT")
    print("=" * 75)
    
    # 5 best-performing classes
    df_sorted_acc = df_per_class.sort_values(by=["accuracy", "f1_score"], ascending=[False, False])
    print("\n[1] TOP 5 BEST-PERFORMING CLASSES:")
    print(f"{'Class ID':<10} {'Class Name':<35} {'Correct/Total':<15} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 80)
    for _, r in df_sorted_acc.head(5).iterrows():
        print(f"Class {int(r['class_id']):02d}    {r['class_name'][:33]:<35} {int(r['correct_predictions'])}/{int(r['total_samples']):<11} {r['accuracy']:.2%}     {r['f1_score']:.4f}")
        
    # 5 weakest classes
    df_weakest = df_per_class.sort_values(by=["accuracy", "f1_score"], ascending=[True, True])
    print("\n[2] TOP 5 WEAKEST CLASSES:")
    print(f"{'Class ID':<10} {'Class Name':<35} {'Correct/Total':<15} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 80)
    for _, r in df_weakest.head(5).iterrows():
        print(f"Class {int(r['class_id']):02d}    {r['class_name'][:33]:<35} {int(r['correct_predictions'])}/{int(r['total_samples']):<11} {r['accuracy']:.2%}     {r['f1_score']:.4f}")
        
    # Top 10 confusion pairs
    print("\n[3] TOP 10 CONFUSION PAIRS (ACTUAL -> PREDICTED):")
    print(f"{'Rank':<5} {'Actual Class':<35} {'Predicted Class':<35} {'Count':<7} {'% of Actual':<10}")
    print("-" * 92)
    for _, r in df_top.iterrows():
        actual_str = f"[{int(r['actual_class_id']):02d}] {r['actual_class_name'][:28]}"
        pred_str = f"[{int(r['predicted_class_id']):02d}] {r['predicted_class_name'][:28]}"
        print(f"{int(r['rank']):<5} {actual_str:<35} {pred_str:<35} {int(r['count']):<7} {r['percentage_of_actual_class']:.2f}%")
        
    print("\n[4] OVERALL PERFORMANCE SUMMARY:")
    print(f"Overall Test Accuracy:    {overall_acc:.4%} ({overall_acc * 100:.2f}%)")
    print(f"Macro F1:                 {f1_macro:.4f} ({f1_macro * 100:.2f}%)")
    print(f"Weighted F1:              {f1_weighted:.4f} ({f1_weighted * 100:.2f}%)")
    print("=" * 75)

if __name__ == "__main__":
    main()
