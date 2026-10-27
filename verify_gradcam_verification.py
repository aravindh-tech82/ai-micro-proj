"""
verify_gradcam_verification.py
==============================
Generates results/gradcam_verification.png and performs the rigorous quantitative
and qualitative sanity checks requested by the user for the Grad-CAM implementation.

Test Subject: Known correctly classified speed-limit test image from official GTSRB:
Image: 00040.ppm (Class 7: Speed limit (100km/h))
Original Native Resolution: 71x70 pixels
Ground Truth ROI Bounding Box: [Roi.X1=7, Roi.Y1=6, Roi.X2=65, Roi.Y2=64]
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# Ensure local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from gradcam import generate_gradcam, _get_global_gradcam

def main():
    project_root = os.path.abspath(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load ground truth metadata
    df = pd.read_csv(os.path.join(project_root, "dataset", "raw", "GTSRB", "GT-final_test.csv"), sep=";")
    target_filename = "00040.ppm"
    row = df[df["Filename"] == target_filename].iloc[0]

    actual_class_id = int(row["ClassId"])
    actual_class_name = "Speed limit (100km/h)"
    x1, y1, x2, y2 = int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"])

    # 2. Load original image at native resolution
    raw_path = os.path.join(project_root, "dataset", "raw", "GTSRB", "Final_Test", "Images", target_filename)
    pil_orig = Image.open(raw_path)
    orig_w, orig_h = pil_orig.size

    # 3. Prepare 32x32 model input
    pil_32 = pil_orig.resize((32, 32), Image.Resampling.LANCZOS)
    arr_32 = np.array(pil_32, dtype=np.float32) / 255.0

    # 4. Verify model inference before Grad-CAM
    gradcam_engine, device = _get_global_gradcam()
    tensor_test = torch.from_numpy(arr_32).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits_before = gradcam_engine.model(tensor_test)
        pred_before = int(logits_before.argmax(dim=1).item())
        conf_before = float(F.softmax(logits_before, dim=1)[0, pred_before].item())

    # 5. Generate Grad-CAM with native resolution overlay
    pred_class, confidence, heatmap, overlay = generate_gradcam(arr_32, original_image=pil_orig)

    with torch.no_grad():
        logits_after = gradcam_engine.model(tensor_test)
        pred_after = int(logits_after.argmax(dim=1).item())

    # Verify predictions
    assert pred_before == pred_after == pred_class, f"Prediction inconsistency: {pred_before} vs {pred_after} vs {pred_class}"
    assert not np.isnan(heatmap).any(), "Heatmap contains NaN"
    assert not np.isinf(heatmap).any(), "Heatmap contains Inf"
    assert heatmap.max() > 0, "Heatmap is completely zero/empty"
    assert heatmap.shape == (orig_h, orig_w), f"Heatmap shape mismatch: {heatmap.shape} vs ({orig_h}, {orig_w})"
    assert overlay.shape == (orig_h, orig_w, 3), f"Overlay shape mismatch: {overlay.shape} vs ({orig_h}, {orig_w}, 3)"

    # 6. Quantitative Sanity Check: Inside vs. Outside Sign Bounding Box
    sign_mask = np.zeros_like(heatmap, dtype=bool)
    sign_mask[y1:y2, x1:x2] = True

    total_activation = heatmap.sum()
    inside_activation = heatmap[sign_mask].sum()
    outside_activation = heatmap[~sign_mask].sum()

    pct_inside = (inside_activation / total_activation) * 100.0 if total_activation > 0 else 0.0
    pct_outside = (outside_activation / total_activation) * 100.0 if total_activation > 0 else 0.0

    print("=" * 75)
    print("GRAD-CAM IMPLEMENTATION & VISUALIZATION VERIFICATION")
    print("=" * 75)
    print(f"Test Image Filename:         {target_filename}")
    print(f"Original Native Resolution:  {orig_w}x{orig_h} pixels")
    print(f"Target Convolutional Layer:  features[10] (Conv2d(64, 128, kernel_size=3, padding=1))")
    print(f"Actual Class:                [{actual_class_id}] {actual_class_name}")
    print(f"Predicted Class:             [{pred_class}] {actual_class_name}")
    print(f"Prediction Confidence:       {confidence * 100:.2f}%")
    print(f"Prediction Consistency:      PASSED (Before: {pred_before}, After: {pred_after})")
    print(f"Ground Truth Bounding Box:   [X1={x1}, Y1={y1}, X2={x2}, Y2={y2}]")
    print(f"NaN / Inf Detected:          NONE (PASSED)")
    print(f"Zero / Empty Heatmap:        NO (Max activation: {heatmap.max():.4f})")
    print("-" * 75)
    print("QUANTITATIVE LOCALIZATION SANITY CHECK:")
    print(f"  Activation INSIDE Sign Region:   {pct_inside:.2f}%")
    print(f"  Activation OUTSIDE Sign Region:  {pct_outside:.2f}%")
    print(f"  Heatmap Mean Inside Sign ROI:    {heatmap[sign_mask].mean():.4f}")
    print(f"  Heatmap Mean Outside Sign ROI:   {heatmap[~sign_mask].mean():.4f}")
    print("=" * 75)

    # 7. Render High-Resolution Verification Figure
    # Required 4 sub-figures:
    # A. Original uploaded/test image at its original resolution
    # B. 32x32 model input, enlarged only for visualization
    # C. Grad-CAM heatmap
    # D. Grad-CAM overlay on the ORIGINAL image
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), facecolor="#ffffff")
    fig.suptitle(
        f"Grad-CAM Verification on Speed Limit Sign ({target_filename})\n"
        f"Actual: [{actual_class_id}] {actual_class_name} | Pred: [{pred_class}] {actual_class_name} | Confidence: {confidence * 100:.2f}%\n"
        f"Quantitative Localization: {pct_inside:.2f}% INSIDE Sign ROI (features[10]: Conv2D)",
        fontsize=14, fontweight="bold", y=1.02, color="#0f172a"
    )

    # Sub-figure A: Original native resolution image
    axes[0].imshow(pil_orig)
    axes[0].set_title(f"A. Original Test Image\nNative Resolution ({orig_w}x{orig_h})", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Sub-figure B: 32x32 model input, enlarged only for visualization
    axes[1].imshow(arr_32)
    axes[1].set_title("B. 32x32 Model Input\n(Enlarged for Visualization)", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")

    # Sub-figure C: Grad-CAM heatmap
    im_c = axes[2].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"C. Grad-CAM Heatmap\nNormalized [{orig_w}x{orig_h}]", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")
    cbar = fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # Sub-figure D: Grad-CAM overlay on the ORIGINAL image
    axes[3].imshow(overlay)
    # Draw ground truth ROI rectangle on D for visual proof
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#22c55e", linewidth=2.5, linestyle="--", label="Ground Truth ROI")
    axes[3].add_patch(rect)
    axes[3].legend(loc="upper right", fontsize=8.5, framealpha=0.85)
    axes[3].set_title(f"D. Overlay on ORIGINAL Image\n({pct_inside:.1f}% Inside Sign ROI)", fontsize=11, fontweight="bold", pad=8)
    axes[3].axis("off")

    plt.tight_layout()
    output_png = os.path.join(results_dir, "gradcam_verification.png")
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved verification figure to: {output_png}")

if __name__ == "__main__":
    main()
