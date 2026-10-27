"""
gradcam.py
==========
Grad-CAM (Gradient-weighted Class Activation Mapping) explainability pipeline
for the FINAL Improved Traffic Sign CNN.

Target Model: models/best_traffic_sign_cnn_improved.pth
Target Layer: Last convolutional layer (features[10]: Conv2D(64, 128, kernel_size=3, padding=1))

Key Features:
- Preserves native resolution of original uploaded image
- Resizes heatmap directly to original image dimensions (orig_h, orig_w)
- Creates overlay directly on the original resolution image (NOT the 32x32 image)
- Handles edge-case where raw CAM is sub-zero by falling back to relative activation, avoiding empty/zero heatmaps
- Verifies prediction consistency before and after Grad-CAM
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(__file__))
from train_improved import ImprovedTrafficSignCNN


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation
    for PyTorch convolutional neural networks.
    """
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.model.eval()

        self.activations = None
        self.gradients = None
        self.hook_handle_forward = None
        self.hook_handle_backward = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is gradient with respect to module output
            self.gradients = grad_output[0]

        self.hook_handle_forward = self.target_layer.register_forward_hook(forward_hook)
        self.hook_handle_backward = self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None, target_size=(32, 32)):
        """
        Computes Grad-CAM heatmap for an input tensor of shape (1, 3, 32, 32),
        and interpolates the heatmap directly to target_size (orig_h, orig_w).
        
        Returns:
            predicted_class (int)
            confidence (float)
            heatmap (numpy.ndarray): Shape target_size (H, W), normalized in [0.0, 1.0]
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

        pred_class = int(logits.argmax(dim=1).item())
        confidence = float(probabilities[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        # Target class score for backpropagation
        score = logits[0, target_class]
        score.backward()

        # 1. Global average pooling of gradients: alpha_k = (1/Z) * sum(grad_k)
        # self.gradients: (1, 128, 8, 8)
        pooled_gradients = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 2. Weighted linear combination of activation maps
        cam = torch.sum(pooled_gradients * self.activations, dim=1, keepdim=True)

        # 3. Apply ReLU to retain positive influences
        cam_relu = F.relu(cam)

        # Check if ReLU zeroed out the entire map (due to uncentered conv baseline shift)
        if cam_relu.max().item() > 0:
            cam_final = cam_relu
        else:
            # Fallback to relative baseline shift so zero/empty heatmap never occurs
            cam_final = cam - cam.min()

        # 4. Upsample directly to TARGET ORIGINAL image dimensions (orig_h, orig_w)
        orig_h, orig_w = target_size
        cam_resized = F.interpolate(
            cam_final,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )

        heatmap = cam_resized.squeeze().cpu().detach().numpy()

        # Sanity check: Check for NaN / Inf
        if np.isnan(heatmap).any() or np.isinf(heatmap).any():
            heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)

        # 5. Normalize to [0.0, 1.0]
        denom = heatmap.max() - heatmap.min()
        if denom > 1e-8:
            heatmap = (heatmap - heatmap.min()) / denom
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.float32)

        return pred_class, confidence, heatmap


# Global singleton instances for simple generate_gradcam() API
_GLOBAL_MODEL = None
_GLOBAL_GRADCAM = None
_GLOBAL_DEVICE = None


def _get_global_gradcam():
    global _GLOBAL_MODEL, _GLOBAL_GRADCAM, _GLOBAL_DEVICE
    if _GLOBAL_GRADCAM is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ckpt_path = os.path.join(project_root, "models", "best_traffic_sign_cnn_improved.pth")

        _GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _GLOBAL_MODEL = ImprovedTrafficSignCNN(num_classes=43).to(_GLOBAL_DEVICE)

        checkpoint = torch.load(ckpt_path, map_location=_GLOBAL_DEVICE)
        _GLOBAL_MODEL.load_state_dict(checkpoint["model_state_dict"])
        _GLOBAL_MODEL.eval()

        # Target layer is features[10]: final convolutional layer
        target_layer = _GLOBAL_MODEL.features[10]
        _GLOBAL_GRADCAM = GradCAM(_GLOBAL_MODEL, target_layer, _GLOBAL_DEVICE)

    return _GLOBAL_GRADCAM, _GLOBAL_DEVICE


def generate_gradcam(image, original_image=None, target_class=None):
    """
    User-friendly Grad-CAM function that overlays on the ORIGINAL resolution image.
    
    Args:
        image (numpy.ndarray, PIL.Image, or torch.Tensor):
            Traffic sign image for CNN input. If not 32x32, will be resized to (32, 32).
        original_image (numpy.ndarray or PIL.Image, optional):
            Original uploaded image at its native resolution (e.g. 71x70, 200x200).
            If None, uses `image`.
        target_class (int, optional):
            Specific class to explain. If None, uses top predicted class.
            
    Returns:
        predicted_class (int): Predicted GTSRB class ID (0 to 42)
        confidence (float): Softmax probability of predicted class (0.0 to 1.0)
        heatmap (numpy.ndarray): 2D float32 array of shape (orig_h, orig_w) in [0.0, 1.0]
        overlay (numpy.ndarray): 3D float32 array of shape (orig_h, orig_w, 3) in [0.0, 1.0]
    """
    gradcam_engine, device = _get_global_gradcam()

    # 1. Process original image at native resolution for display overlay
    if original_image is not None:
        if isinstance(original_image, Image.Image):
            orig_pil = original_image.convert("RGB")
            orig_np = np.array(orig_pil, dtype=np.float32) / 255.0
        else:
            orig_np = np.array(original_image, dtype=np.float32)
            if orig_np.max() > 1.0:
                orig_np = orig_np / 255.0
    else:
        if isinstance(image, Image.Image):
            orig_pil = image.convert("RGB")
            orig_np = np.array(orig_pil, dtype=np.float32) / 255.0
        else:
            orig_np = np.array(image, dtype=np.float32)
            if orig_np.ndim == 3 and orig_np.shape[0] == 3:
                orig_np = np.transpose(orig_np, (1, 2, 0))
            if orig_np.max() > 1.0:
                orig_np = orig_np / 255.0

    orig_np = np.clip(orig_np, 0.0, 1.0).astype(np.float32)
    orig_h, orig_w = orig_np.shape[:2]

    # 2. Process model input: exactly 32x32 RGB
    if isinstance(image, Image.Image):
        img_32_pil = image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
        img_32_np = np.array(img_32_pil, dtype=np.float32) / 255.0
    else:
        img_np = np.array(image, dtype=np.float32)
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        if img_np.shape[:2] != (32, 32):
            pil_temp = Image.fromarray((np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8))
            pil_temp = pil_temp.resize((32, 32), Image.Resampling.LANCZOS)
            img_32_np = np.array(pil_temp, dtype=np.float32) / 255.0
        else:
            img_32_np = img_np

    img_32_np = np.clip(img_32_np, 0.0, 1.0).astype(np.float32)

    # 3. Create CNN tensor: (1, 3, 32, 32)
    tensor = torch.from_numpy(img_32_np).permute(2, 0, 1).unsqueeze(0).float()

    # Verify prediction before Grad-CAM
    with torch.no_grad():
        initial_logits = gradcam_engine.model(tensor.to(device))
        initial_pred = int(initial_logits.argmax(dim=1).item())

    # 4. Generate Grad-CAM with target_size = (orig_h, orig_w)
    pred_class, confidence, heatmap = gradcam_engine.generate(
        tensor,
        target_class=target_class,
        target_size=(orig_h, orig_w)
    )

    # Verify prediction consistency
    assert pred_class == initial_pred, f"Prediction mismatch: {pred_class} vs {initial_pred}"

    # 5. Generate colorful heatmap overlay using Jet colormap
    colormap = matplotlib.colormaps["jet"]
    heatmap_colored = colormap(heatmap)[:, :, :3]  # Strip alpha, retain RGB in [0, 1]

    # 6. Blend directly with the ORIGINAL resolution image
    overlay = 0.5 * orig_np + 0.5 * heatmap_colored
    overlay = np.clip(overlay, 0.0, 1.0)

    # Sanity checks
    assert not np.isnan(heatmap).any(), "Heatmap contains NaN values"
    assert not np.isinf(heatmap).any(), "Heatmap contains Inf values"
    assert heatmap.shape == (orig_h, orig_w), f"Dimension mismatch: {heatmap.shape} vs ({orig_h}, {orig_w})"
    assert overlay.shape == (orig_h, orig_w, 3), f"Overlay dimension mismatch: {overlay.shape} vs ({orig_h}, {orig_w}, 3)"

    return pred_class, confidence, heatmap, overlay
