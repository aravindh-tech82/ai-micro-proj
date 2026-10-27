# Traffic Sign Recognition Using Neural Networks

CNN-Based Traffic Sign Classification with Explainable AI (Grad-CAM) on the German Traffic Sign Recognition Benchmark (GTSRB).

---

## 🚦 Project Overview
This project develops an end-to-end, high-accuracy deep learning pipeline for traffic sign recognition designed for Advanced Driver Assistance Systems (ADAS) and Autonomous Vehicles:
- **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB) with 43 distinct classes.
- **Model Architecture:** 3-block Convolutional Neural Network with Batch Normalization, Dropout regularization, Adam optimizer, and Cosine Annealing learning rate schedule.
- **Explainable AI:** Integrated **Grad-CAM** visual explanations at native image resolution targeting the final convolutional layer.
- **Interactive Dashboard:** Full-featured local web dashboard built with **Streamlit** for real-time sign classification, domain knowledge mapping (Category, Meaning, Driver Action), and visual attribution.

---

## 📊 Key Results

Evaluated strictly on **12,630 unseen official GTSRB test images**:

| Metric | Baseline CNN | Improved CNN | Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Best Validation Accuracy** | 98.67% | **99.64%** | **+0.97%** |
| **Unseen Test Accuracy** | 88.69% | **95.61%** | **+6.92%** |
| **Test Loss** | 0.6233 | **0.1508** | **-0.4725** (75.8% drop) |
| **Macro F1-Score** | 85.85% | **94.43%** | **+8.58%** |
| **Weighted F1-Score** | 88.76% | **95.62%** | **+6.86%** |
| **Generalization Gap** | 9.98% | **4.03%** | **Narrowed by 5.95%** |

---

## 🛠️ Project Structure

```
Traffic-Sign-Recognition/
├── models/
│   ├── best_traffic_sign_cnn.pth             # Baseline trained model weights
│   └── best_traffic_sign_cnn_improved.pth    # Final improved CNN model weights
├── results/
│   ├── improved_accuracy_curve.png           # Training vs validation accuracy curve
│   ├── improved_loss_curve.png               # Training vs validation loss curve
│   ├── improved_confusion_matrix.png         # Raw count confusion matrix (43x43)
│   ├── improved_confusion_matrix_normalized.png # Normalized recall confusion matrix
│   ├── gradcam_examples.png                  # Multi-class Grad-CAM visualization grid
│   ├── gradcam_verification.png              # Quantitative ROI localization verification
│   └── improved_classification_report.txt    # Per-class precision, recall, F1-score
├── src/
│   ├── download_data.py                      # GTSRB dataset downloader
│   ├── prepare_dataset.py                    # Preprocessing, resizing, and train/val split
│   ├── verify_dataset.py                     # Dataset integrity and leakage checks
│   ├── train.py                              # Baseline CNN training script
│   ├── train_improved.py                     # Improved CNN training with BN, Dropout, Scheduler
│   ├── evaluate.py                           # Baseline evaluation suite
│   ├── evaluate_improved.py                  # Improved CNN error analysis and confusion analysis
│   ├── gradcam.py                            # Native-resolution Grad-CAM engine
│   ├── predict.py                            # Prediction intelligence layer & knowledge base
│   └── app.py                                # Streamlit interactive web dashboard
├── test_samples/                             # Sample test signs for demonstration
├── dataset_summary.txt                       # Verified dataset class counts & statistics
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation
Clone this repository and create a virtual environment:
```bash
git clone https://github.com/aravindh-tech82/ai-micro-proj.git
cd ai-micro-proj
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow streamlit
```

### 2. Dataset Setup & Training
```bash
# 1. Download official GTSRB dataset
python src/download_data.py

# 2. Preprocess and split (80% train / 20% validation)
python src/prepare_dataset.py

# 3. Train improved CNN
python src/train_improved.py
```

### 3. Launch the Web Dashboard
```bash
python -m streamlit run src/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to test sign images and inspect Grad-CAM heatmaps.
