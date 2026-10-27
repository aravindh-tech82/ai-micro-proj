"""
evaluate.py
===========
Comprehensive evaluation pipeline for the trained GTSRB CNN model.

Uses ONLY the saved unseen test predictions and actual test images:
- predictions: results/test_predictions.npz
- test images: dataset/processed/gtsrb_processed.npz
- labels: dataset/processed/class_labels.json

Generates:
1. results/confusion_matrix.png
2. results/confusion_matrix_normalized.png
3. results/classification_report.txt
4. results/per_class_accuracy.csv
5. results/top_confusions.csv
6. results/misclassified_samples.png
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

def plot_confusion_matrix(cm, class_ids, output_path, title, normalized=False, cmap=plt.cm.Blues):
    """
    Renders and saves a high-resolution confusion matrix heatmap.
    """
    plt.figure(figsize=(14, 12))
    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    
    if normalized:
        im = plt.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    else:
        im = plt.imshow(cm, interpolation="nearest", cmap=cmap)
        
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    if normalized:
        cbar.set_label("Proportion of True Class (Recall)", fontsize=11)
    else:
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

def plot_misclassified_samples(X_test, y_true, y_pred, y_prob, label_map, output_path, n_samples=12):
    """
    Plots a grid of misclassified test images showing Actual, Predicted, and Confidence.
    """
    # Find all misclassified indices
    mis_indices = np.where(y_true != y_pred)[0]
    
    # Pick diverse misclassified samples
    # Group by (true, pred) to get a variety of error types
    seen_pairs = set()
    selected_indices = []
    
    # First pick unique error pairs
    for idx in mis_indices:
        pair = (int(y_true[idx]), int(y_pred[idx]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            selected_indices.append(idx)
        if len(selected_indices) == n_samples:
            break
            
    # Fallback if needed
    if len(selected_indices) < n_samples:
        remaining = [i for i in mis_indices if i not in selected_indices]
        selected_indices.extend(remaining[:(n_samples - len(selected_indices))])
        
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    fig.suptitle("Sample Misclassified Test Images (Actual vs. Predicted)", fontsize=16, fontweight="bold", y=0.98)
    
    for ax, idx in zip(axes.flat, selected_indices):
        img = X_test[idx]
        actual_id = int(y_true[idx])
        pred_id = int(y_pred[idx])
        conf = float(y_prob[idx, pred_id]) * 100.0
        
        actual_name = label_map.get(actual_id, f"Class {actual_id}")
        pred_name = label_map.get(pred_id, f"Class {pred_id}")
        
        # Shorten very long names for readability
        if len(actual_name) > 24:
            actual_name = actual_name[:22] + "..."
        if len(pred_name) > 24:
            pred_name = pred_name[:22] + "..."
            
        ax.imshow(img)
        title_text = f"True [{actual_id}]: {actual_name}\nPred [{pred_id}]: {pred_name}\nConf: {conf:.1f}%"
        ax.set_title(title_text, fontsize=9.5, pad=6, color="#b30000", fontweight="semibold")
        ax.axis("off")
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pred_path = os.path.join(project_root, "results", "test_predictions.npz")
    npz_data_path = os.path.join(project_root, "dataset", "processed", "gtsrb_processed.npz")
    labels_json_path = os.path.join(project_root, "dataset", "processed", "class_labels.json")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 70)
    print("GTSRB MODEL EVALUATION ON UNSEEN TEST PREDICTIONS")
    print(f"Loading predictions: {pred_path}")
    print("=" * 70)
    
    # 1. Load predictions
    pred_data = np.load(pred_path)
    y_true = pred_data["y_true"]
    y_pred = pred_data["y_pred"]
    y_prob = pred_data["probabilities"]
    
    # 2. Load test images for visual error inspection
    dataset_npz = np.load(npz_data_path)
    X_test = dataset_npz["X_test"]
    
    # 3. Load class labels
    with open(labels_json_path, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
        
    num_classes = 43
    class_ids = [str(i) for i in range(num_classes)]
    target_names = [label_map[i] for i in range(num_classes)]
    
    # Overall Accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    total_test = len(y_true)
    correct_count = int((y_true == y_pred).sum())
    incorrect_count = total_test - correct_count
    
    print(f"Total Test Samples:      {total_test}")
    print(f"Correct Predictions:     {correct_count}")
    print(f"Incorrect Predictions:   {incorrect_count}")
    print(f"Overall Test Accuracy:   {overall_acc:.4%} ({overall_acc * 100:.2f}%)")
    
    # Global Macro & Weighted metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # 4. Generate Classification Report Text File
    report_str = classification_report(
        y_true, y_pred, labels=list(range(num_classes)), target_names=target_names, digits=4, zero_division=0
    )
    
    report_output_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write("                  GTSRB CLASSIFICATION REPORT (UNSEEN TEST SET)\n")
        f.write("=" * 85 + "\n\n")
        f.write(report_str + "\n\n")
        f.write("-" * 85 + "\n")
        f.write("GLOBAL SUMMARY METRICS\n")
        f.write("-" * 85 + "\n")
        f.write(f"Total Test Samples:       {total_test}\n")
        f.write(f"Overall Accuracy:         {overall_acc:.4f} ({overall_acc * 100:.2f}%)\n")
        f.write(f"Macro Precision:          {precision_macro:.4f}\n")
        f.write(f"Macro Recall:             {recall_macro:.4f}\n")
        f.write(f"Macro F1-Score:           {f1_macro:.4f}\n")
        f.write(f"Weighted Precision:       {precision_weighted:.4f}\n")
        f.write(f"Weighted Recall:          {recall_weighted:.4f}\n")
        f.write(f"Weighted F1-Score:        {f1_weighted:.4f}\n")
        f.write("=" * 85 + "\n")
    print(f"Saved: {report_output_path}")
    
    # 5. Per-Class Accuracy CSV
    per_class_records = []
    for c in range(num_classes):
        mask = (y_true == c)
        c_support = int(mask.sum())
        c_correct = int((y_pred[mask] == c).sum())
        c_acc = (c_correct / c_support) if c_support > 0 else 0.0
        
        # Per-class precision, recall, f1
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
    per_class_path = os.path.join(results_dir, "per_class_accuracy.csv")
    df_per_class.to_csv(per_class_path, index=False)
    print(f"Saved: {per_class_path}")
    
    # 6. Top 10 Most Confused Class Pairs CSV
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
            "true_class_id": t,
            "true_class_name": label_map[t],
            "pred_class_id": p,
            "pred_class_name": label_map[p],
            "count": count,
            "pct_of_true_class": round(pct_of_true, 2)
        })
        
    df_top = pd.DataFrame(top_records)
    top_conf_path = os.path.join(results_dir, "top_confusions.csv")
    df_top.to_csv(top_conf_path, index=False)
    print(f"Saved: {top_conf_path}")
    
    # 7. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        cm, class_ids, cm_path,
        title="GTSRB Test Set Confusion Matrix (Raw Counts)",
        normalized=False, cmap=plt.cm.Blues
    )
    
    # 8. Normalized Confusion Matrix (Row-Normalized: Recall per class)
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    cm_norm_path = os.path.join(results_dir, "confusion_matrix_normalized.png")
    plot_confusion_matrix(
        cm_norm, class_ids, cm_norm_path,
        title="GTSRB Test Set Normalized Confusion Matrix (Recall)",
        normalized=True, cmap=plt.cm.Blues
    )
    
    # 9. Misclassified Samples Plot
    mis_samples_path = os.path.join(results_dir, "misclassified_samples.png")
    plot_misclassified_samples(
        X_test, y_true, y_pred, y_prob, label_map, mis_samples_path, n_samples=12
    )
    
    # 10. Print Concise Evaluation Summary
    print("\n" + "=" * 70)
    print("CONCISE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Images:        {total_test}")
    print(f"Correct Predictions:     {correct_count}")
    print(f"Misclassified:           {incorrect_count}")
    print(f"Overall Test Accuracy:   {overall_acc:.2%}")
    print(f"Macro Precision:          {precision_macro:.4f}")
    print(f"Macro Recall:             {recall_macro:.4f}")
    print(f"Macro F1-Score:           {f1_macro:.4f}")
    print(f"Weighted Precision:       {precision_weighted:.4f}")
    print(f"Weighted Recall:          {recall_weighted:.4f}")
    print(f"Weighted F1-Score:        {f1_weighted:.4f}")
    print("-" * 70)
    
    print("\nTop 5 Highest Accuracy Classes:")
    df_sorted_acc = df_per_class.sort_values(by="accuracy", ascending=False)
    for _, r in df_sorted_acc.head(5).iterrows():
        print(f"  Class {int(r['class_id']):02d} ({r['class_name'][:28]:<28}): {r['accuracy']:.1%} ({int(r['correct'])}/{int(r['support'])})")
        
    print("\nTop 5 Lowest Accuracy Classes:")
    for _, r in df_sorted_acc.tail(5).iterrows():
        print(f"  Class {int(r['class_id']):02d} ({r['class_name'][:28]:<28}): {r['accuracy']:.1%} ({int(r['correct'])}/{int(r['support'])})")
        
    print("\nTop 5 Most Frequent Confusions:")
    for _, r in df_top.head(5).iterrows():
        print(f"  Actual: [{int(r['true_class_id']):02d}] {r['true_class_name'][:22]:<22} -> Pred: [{int(r['pred_class_id']):02d}] {r['pred_class_name'][:22]:<22} | Count: {int(r['count']):3d} ({r['pct_of_true_class']:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()
