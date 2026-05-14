# ♻️ Smart Garbage Classification
### Computer Vision & Image Processing — Final Project

---

## 👥 Team Members

* Salma Ahmad Abdelhamid
* Mohamed Hossam Saeed

---

## 📌 Project Overview

A complete end-to-end deep learning pipeline for classifying garbage images into **5 categories**:
`clothes` · `glass` · `metal` · `paper` · `plastic`

The project covers the full pipeline — from raw data collection and preprocessing, to building two models from scratch and with transfer learning, to evaluation, comparison, and a live web deployment.

---

## 📂 Repository Structure

```
SmartGarbageClassification/
│
├── data/
│   └── dataset_link.txt
│
├── notebooks/
│   └── Final_Attempt.ipynb                          ← main notebook (all stages)
│
├── deployment/
│   ├── app.py                                       ← Streamlit web application
│   ├── requirements.txt                             ← Python dependencies
│   ├── cnn_scratch_best.keras                       ← trained CNN from scratch
│   └── drive_link_transfer_model_final.keras        ← trained Transfer Learning model drive link (file exceeds 25MB)
│      

│
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | [Kaggle — Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) |
| Classes | 5 (clothes, glass, metal, paper, plastic) |
| Total Images | ~4,500 |
| Train Split | 80% (3,600 images) |
| Val Split | 10% (450 images) |
| Test Split | 10% (450 images) |

> Built-in datasets (MNIST, CIFAR, etc.) were NOT used — raw images handled manually as required.

---

## ⚙️ Preprocessing Steps

All preprocessing is documented in the notebook. Key steps:

- ✅ Corrupted and missing image detection and removal
- ✅ Class balance check and visualization
- ✅ Resize all images to **128×128**
- ✅ Normalize pixel values — `[0,1]` for CNN, `[-1,1]` for Transfer Learning
- ✅ Data Augmentation (rotation, flip, zoom, brightness, shift, shear)
- ✅ Train / Validation / Test split

---

## 🧠 Models

### Model 1 — CNN from Scratch
Built layer by layer with no pre-trained weights:
- 4 Convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPooling)
- GlobalAveragePooling2D
- Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Softmax(5)
- Trained for 30 epochs with EarlyStopping and ReduceLROnPlateau

### Model 2 — Transfer Learning (MobileNetV2)
Manual pipeline — no drag and drop:
- Backbone: MobileNetV2 pretrained on ImageNet (no top)
- Custom head: GlobalAveragePooling → Dense(256) → Dense(128) → Softmax(5)
- Phase 1: Frozen backbone — train head only (lr=1e-4, 10 epochs)
- Phase 2: Unfreeze last 30 layers — fine-tune (lr=1e-5, 10 epochs)

---

## 📈 Results

### Model Comparison

| Metric | CNN from Scratch | Transfer Learning |
|--------|:---:|:---:|
| Test Accuracy | 80.89% | 92.00% |
| Precision | 0.8125 | 0.9202 |
| Recall | 0.8089 | 0.9200 |
| F1-Score | 0.8084 | 0.9194 |
| Macro AUC | 0.9653 | 0.9885 |
| Trainable Parameters | 489,477 | 1,888,645 |
| Training Time | ~69 min | ~40 min |

### Key Observations
- Transfer Learning outperformed CNN from Scratch on every metric
- Plastic was the hardest class for both models due to visual similarity with glass
- Transfer Learning converged faster despite being a larger model
- Both models achieved strong AUC scores (>0.96) across all classes

---

## 🚀 Deployment

A fully functional **Streamlit web application** was built for real-time garbage classification.

### Features
- Upload any garbage image and get instant prediction
- Choose between CNN from Scratch or Transfer Learning model
- View confidence scores per class with visual bar chart
- Model comparison table with all metrics
- Recycling tips per predicted class

### How to Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Make sure these files are in the deployment folder:**
```
deployment/
├── app.py
├── requirements.txt
├── cnn_scratch_best.keras
└── transfer_model_final.keras
```

**3. Run the app:**
```bash
streamlit run app.py
```

**4. Open your browser at:**
```
http://localhost:8501
```

---

## 🛠️ Requirements

```
streamlit>=1.32.0
tensorflow>=2.13.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📋 Evaluation Metrics Covered

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-Score
- ✅ Confusion Matrix (heatmap)
- ✅ Loss & Accuracy Curves (both phases for TL)
- ✅ ROC / AUC Curves (One-vs-Rest for multi-class)

---

> **Course:** Image Processing & Computer Vision  
> **Institution:** SUTech  
> **Academic Year:** 2025 / 2026
---



---

## 📜 License

This project is for academic purposes only.

```
