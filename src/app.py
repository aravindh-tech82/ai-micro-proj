"""
app.py
======
Streamlit Web Dashboard for:
"Traffic Sign Recognition Using Neural Networks"
CNN-Based Traffic Sign Classification with Explainable AI

Reuses:
- src/predict.py -> predict_traffic_sign()
- src/gradcam.py -> generate_gradcam()
- models/best_traffic_sign_cnn_improved.pth
"""

import os
import sys
import numpy as np
from PIL import Image
import streamlit as st

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import predict_traffic_sign, TRAFFIC_SIGN_KNOWLEDGE_BASE

# Streamlit Page Configuration
st.set_page_config(
    page_title="Traffic Sign Recognition Using Neural Networks",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #475569;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .result-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .action-card {
        background-color: #ffffff;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .gradcam-note {
        font-size: 0.95rem;
        color: #334155;
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        border-left: 4px solid #0284c7;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_test_images():
    """
    Loads a curated selection of test set samples for instant one-click demonstration.
    """
    npz_path = os.path.join(PROJECT_ROOT, "dataset", "processed", "gtsrb_processed.npz")
    if not os.path.exists(npz_path):
        return {}

    data = np.load(npz_path)
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Preselected diverse classes: Stop, Speed limit 100km/h, Yield, Keep right, Bumpy road, Ice/snow
    sample_catalog = {
        "Stop Sign (Class 14)": 93,
        "Speed Limit 100km/h (Class 07)": 11,
        "Keep Right (Class 38)": 8,
        "Yield (Class 13)": 2,
        "Bumpy Road (Class 22)": 88,
        "Beware of Ice/Snow (Class 30)": 41,
    }

    loaded_samples = {}
    for label, idx in sample_catalog.items():
        if idx < len(X_test):
            loaded_samples[label] = {
                "image_array": X_test[idx],
                "actual_class_id": int(y_test[idx]),
                "index": idx
            }
    return loaded_samples


def main():
    # 1. HEADER
    st.markdown('<div class="main-header">Traffic Sign Recognition Using Neural Networks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">CNN-Based Traffic Sign Classification with Explainable AI</div>', unsafe_allow_html=True)

    # 2. SIDEBAR - MODEL PERFORMANCE CARD
    with st.sidebar:
        st.header("📊 Model Information")
        st.caption("Official performance on unseen GTSRB test benchmark.")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Dataset</div>
                <div class="metric-value">GTSRB</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Classes</div>
                <div class="metric-value">43</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test Images</div>
            <div class="metric-value">12,630</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="border-color: #86efac; background-color: #f0fdf4;">
            <div class="metric-label" style="color: #166534;">Final Test Accuracy</div>
            <div class="metric-value" style="color: #15803d;">95.61%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Macro F1-Score</div>
            <div class="metric-value">94.43%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best Validation Accuracy</div>
            <div class="metric-value">99.64%</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        **Architecture Details:**
        - 3 Conv Blocks with BatchNorm2D & Regularized Dropout
        - Classifier: Dense(512) $\\rightarrow$ BatchNorm1D $\\rightarrow$ Dropout(0.5) $\\rightarrow$ Output(43)
        - Optimization: Adam with CosineAnnealingLR
        - Explainability: Grad-CAM on Last Conv Layer
        """)

    # 3. INPUT SELECTION: Upload or Demo Test Sample
    st.subheader("1. Select or Upload Traffic Sign Image")
    
    input_tab1, input_tab2 = st.tabs(["📁 Upload Image File", "🧪 Select Demo Test Sign"])
    input_image_32 = None
    display_image_raw = None
    source_description = ""

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Upload Traffic Sign Image",
            type=["png", "jpg", "jpeg"],
            help="Upload a traffic sign image in PNG, JPG, or JPEG format."
        )
        if uploaded_file is not None:
            try:
                pil_img = Image.open(uploaded_file)
                display_image_raw = pil_img.copy()
                
                # Convert to RGB (handles RGBA or Grayscale)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                    
                # Resize to 32x32 for CNN inference
                pil_resized = pil_img.resize((32, 32), Image.Resampling.LANCZOS)
                input_image_32 = np.array(pil_resized, dtype=np.float32) / 255.0
                source_description = f"Uploaded File: {uploaded_file.name} ({pil_img.size[0]}x{pil_img.size[1]})"
            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}. Please upload a valid image file.")

    with input_tab2:
        samples = load_sample_test_images()
        if samples:
            selected_sample_key = st.selectbox(
                "Choose an unseen GTSRB test set image for demonstration:",
                options=list(samples.keys()),
                index=0
            )
            if st.button("Load Selected Test Sign", type="secondary"):
                chosen = samples[selected_sample_key]
                input_image_32 = chosen["image_array"]
                display_image_raw = Image.fromarray((chosen["image_array"] * 255).astype(np.uint8))
                source_description = f"GTSRB Unseen Test Set: {selected_sample_key} (Index #{chosen['index']})"
        else:
            st.info("Test samples catalog loading from disk...")

    # 4. PREDICTION & EXPLAINABILITY EXECUTION
    if input_image_32 is not None:
        st.divider()
        st.subheader("2. Prediction & Model Inference")

        with st.spinner("Executing CNN inference and computing Grad-CAM activations..."):
            try:
                # Call existing prediction intelligence layer with original image for native resolution overlay
                prediction_result = predict_traffic_sign(input_image_32, original_image=display_image_raw)
            except TypeError:
                # Fallback if called with single parameter signature
                prediction_result = predict_traffic_sign(input_image_32)
            except Exception as ex:
                st.error(f"Model prediction failed: {str(ex)}")
                return

        # Verification of prediction values
        pred_name = prediction_result.get("predicted_class", "Unknown")
        confidence_str = prediction_result.get("confidence", "0.00%")
        category = prediction_result.get("category", "General")
        meaning = prediction_result.get("meaning", "No description available.")
        action = prediction_result.get("recommended_action", "Exercise caution.")
        heatmap = prediction_result.get("gradcam_heatmap")
        overlay = prediction_result.get("gradcam_overlay")

        # Layout: Prediction Details Card
        col_pred, col_details = st.columns([1.2, 1.8])

        with col_pred:
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size: 0.85rem; font-weight: 600; color: #0284c7; text-transform: uppercase; margin-bottom: 4px;">
                    Predicted Traffic Sign
                </div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;">
                    {pred_name}
                </div>
                <div style="display: flex; gap: 20px;">
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 600; color: #64748b;">CONFIDENCE</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #16a34a;">{confidence_str}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 600; color: #64748b;">CATEGORY</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #1e293b; padding-top: 4px;">{category}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if source_description:
                st.caption(source_description)

        with col_details:
            st.markdown(f"""
            <div class="action-card">
                <div style="font-size: 0.85rem; font-weight: 700; color: #1e40af; text-transform: uppercase; margin-bottom: 4px;">
                    Sign Meaning
                </div>
                <div style="font-size: 1.05rem; color: #1e293b; line-height: 1.5; margin-bottom: 14px;">
                    {meaning}
                </div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #166534; text-transform: uppercase; margin-bottom: 4px;">
                    Recommended Driver Action
                </div>
                <div style="font-size: 1.05rem; color: #14532d; font-weight: 500; line-height: 1.5;">
                    {action}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 5. GRAD-CAM EXPLANATION SECTION
        st.divider()
        st.subheader("3. Explainable AI: Grad-CAM Activation Analysis")
        st.markdown(
            '<div class="gradcam-note"><strong>Grad-CAM highlights the image regions that contributed most strongly to the CNN prediction.</strong></div>',
            unsafe_allow_html=True
        )

        col_g1, col_g2, col_g3 = st.columns(3)

        with col_g1:
            st.markdown("#### Original Image")
            orig_w, orig_h = display_image_raw.size if isinstance(display_image_raw, Image.Image) else (display_image_raw.shape[1], display_image_raw.shape[0])
            st.image(display_image_raw, caption=f"Original Input ({orig_w}x{orig_h} RGB)", use_container_width=True)

        with col_g2:
            st.markdown("#### Grad-CAM Heatmap")
            st.image(heatmap, caption=f"Activation Hotspot ({heatmap.shape[1]}x{heatmap.shape[0]})", clamp=True, use_container_width=True)

        with col_g3:
            st.markdown("#### Grad-CAM Overlay")
            overlay_uint8 = (overlay * 255).astype(np.uint8)
            st.image(overlay_uint8, caption=f"Visual Attribution Blend ({overlay_uint8.shape[1]}x{overlay_uint8.shape[0]})", use_container_width=True)

    else:
        st.info("👆 Please upload a traffic sign image (PNG, JPG, JPEG) or choose a test sign from the demo tab above to see predictions.")


if __name__ == "__main__":
    main()
