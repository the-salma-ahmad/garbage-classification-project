# ============================================================
# SMART GARBAGE CLASSIFICATION — Streamlit Deployment
# Web Application for Real-Time Image Classification
# Models: CNN from Scratch & Transfer Learning (MobileNetV2)
# ============================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import time

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Garbage Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Clean dark industrial aesthetic
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background-color: #0f1117;
        color: #e8eaf0;
    }

    /* ── Header ── */
    .main-header {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: #4ade80;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: #6b7280;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    /* ── Cards ── */
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3142;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2235 100%);
        border: 1px solid #4ade80;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .pred-class {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #4ade80;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .pred-conf {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-top: 0.3rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #13151f;
        border-right: 1px solid #2d3142;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-family: 'DM Sans', sans-serif;
        color: #c9cdd8;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        background: #1a1d27;
        border: 2px dashed #2d3142;
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #4ade80;
    }

    /* ── Tab style ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1d27;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #6b7280;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4ade80 !important;
        color: #0f1117 !important;
    }

    /* ── Divider ── */
    hr { border-color: #2d3142; }

    /* ── Section titles ── */
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #4ade80;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }

    /* ── Class badge ── */
    .class-badge {
        display: inline-block;
        background: #1e2235;
        border: 1px solid #2d3142;
        border-radius: 20px;
        padding: 4px 14px;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #9ca3af;
        margin: 3px;
    }

    /* ── Table style ── */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
    }
    .comparison-table th {
        background: #1e2235;
        color: #4ade80;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 1px;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 2px solid #4ade80;
    }
    .comparison-table td {
        padding: 10px 16px;
        border-bottom: 1px solid #2d3142;
        color: #c9cdd8;
    }
    .comparison-table tr:hover td {
        background: #1a1d27;
    }
    .winner {
        color: #4ade80;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

CLASSES = ['clothes', 'glass', 'metal', 'paper', 'plastic']
IMG_SIZE = (128, 128)

CLASS_ICONS = {
    'clothes': '👕',
    'glass':   '🍶',
    'metal':   '🔩',
    'paper':   '📄',
    'plastic': '🧴'
}

CLASS_TIPS = {
    'clothes': 'Donate wearable clothes. Recycle damaged textiles at fabric bins.',
    'glass':   'Rinse glass before recycling. Separate by color if required.',
    'metal':   'Crush cans to save space. Remove food residue before recycling.',
    'paper':   'Keep paper dry. Flatten cardboard boxes before disposal.',
    'plastic': 'Check the recycling number. Rinse containers before recycling.'
}

# ============================================================
# MODEL LOADING — cached so it only loads once
# ============================================================

@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("cnn_scratch_best.keras")
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_tl_model():
    try:
        model = load_model("transfer_model_final.keras")
        return model
    except Exception as e:
        return None

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(model, image, model_type):
    """
    Preprocess image and run prediction.
    model_type: 'cnn' or 'tl'
    Returns: predicted class, confidence, all probabilities
    """
    img = image.resize(IMG_SIZE)
    img_array = np.array(img.convert('RGB')).astype(np.float32)

    if model_type == 'tl':
        img_array = preprocess_input(img_array)          # [-1, 1] for MobileNetV2
    else:
        img_array = img_array / 255.0                    # [0, 1] for CNN scratch

    img_array = np.expand_dims(img_array, axis=0)        # add batch dim

    probs = model.predict(img_array, verbose=0)[0]
    pred_idx = np.argmax(probs)

    return CLASSES[pred_idx], float(probs[pred_idx]), probs

# ============================================================
# CONFIDENCE BAR CHART
# ============================================================

def plot_confidence_bars(probs, predicted_class):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#1a1d27')
    ax.set_facecolor('#1a1d27')

    colors = ['#4ade80' if c == predicted_class else '#2d3142' for c in CLASSES]
    bars = ax.barh(CLASSES, probs * 100, color=colors,
                   edgecolor='#0f1117', height=0.55)

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{prob * 100:.1f}%', va='center', ha='left',
                color='#e8eaf0', fontsize=9, fontfamily='monospace')

    ax.set_xlim(0, 115)
    ax.set_xlabel('Confidence (%)', color='#6b7280', fontsize=9)
    ax.tick_params(colors='#9ca3af', labelsize=9)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.xaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3142')

    plt.tight_layout()
    return fig

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<p class="main-header" style="font-size:1.4rem;">♻️ GarbageAI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Smart Classification System</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-title">Select Model</p>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Select Model",
        options=["Transfer Learning (92%)", "CNN from Scratch (81%)"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<p class="section-title">Classes</p>', unsafe_allow_html=True)
    for cls in CLASSES:
        st.markdown(f'<span class="class-badge">{CLASS_ICONS[cls]} {cls}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Model Info</p>', unsafe_allow_html=True)
    if "Transfer" in model_choice:
        st.markdown("""
        <div class="metric-card">
            <div style="color:#4ade80;font-family:monospace;font-size:0.8rem;">MobileNetV2</div>
            <div style="color:#9ca3af;font-size:0.8rem;margin-top:4px;">Backbone: ImageNet weights</div>
            <div style="color:#9ca3af;font-size:0.8rem;">Accuracy: 92.00%</div>
            <div style="color:#9ca3af;font-size:0.8rem;">Training: ~40 min</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div style="color:#4ade80;font-family:monospace;font-size:0.8rem;">Custom CNN</div>
            <div style="color:#9ca3af;font-size:0.8rem;margin-top:4px;">4-block architecture</div>
            <div style="color:#9ca3af;font-size:0.8rem;">Accuracy: 80.89%</div>
            <div style="color:#9ca3af;font-size:0.8rem;">Training: ~69 min</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT — TABS
# ============================================================

st.markdown('<p class="main-header">Smart Garbage Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a garbage image — get instant AI-powered classification</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍  Classify Image", "📊  Model Comparison", "🗂️  Class Samples"])

# ============================================================
# TAB 1 — CLASSIFY IMAGE
# ============================================================

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<p class="section-title">Upload Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop an image here or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

        if uploaded_file:
            # Load the selected model
            with st.spinner("Loading model..."):
                if "Transfer" in model_choice:
                    model = load_tl_model()
                    mtype = 'tl'
                else:
                    model = load_cnn_model()
                    mtype = 'cnn'

            if model is None:
                st.error("⚠️ Model file not found. Make sure the .keras files are in the same folder as app.py")
            else:
                with st.spinner("Classifying..."):
                    start = time.time()
                    pred_class, confidence, probs = predict(model, image, mtype)
                    elapsed = time.time() - start

                # ── Result card ────────────────────────────
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:3rem;">{CLASS_ICONS[pred_class]}</div>
                    <div class="pred-class">{pred_class}</div>
                    <div class="pred-conf">Confidence: {confidence * 100:.2f}%</div>
                    <div style="color:#4b5563;font-size:0.75rem;margin-top:0.5rem;font-family:monospace;">
                        inference: {elapsed*1000:.0f}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Recycling tip ──────────────────────────
                st.info(f"♻️ **Recycling Tip:** {CLASS_TIPS[pred_class]}")

                # ── Confidence bars ────────────────────────
                st.markdown('<p class="section-title">Confidence per Class</p>', unsafe_allow_html=True)
                fig = plot_confidence_bars(probs, pred_class)
                st.pyplot(fig)
                plt.close()

        else:
            st.markdown("""
            <div style="background:#1a1d27;border:1px dashed #2d3142;border-radius:12px;
                        padding:3rem;text-align:center;color:#4b5563;">
                <div style="font-size:2.5rem;">📂</div>
                <div style="font-family:monospace;font-size:0.85rem;margin-top:0.5rem;">
                    Upload an image to begin
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2 — MODEL COMPARISON TABLE
# ============================================================

with tab2:
    st.markdown('<p class="section-title">Model Comparison</p>', unsafe_allow_html=True)

    st.markdown("""
    <table class="comparison-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>CNN from Scratch</th>
                <th>Transfer Learning (MobileNetV2)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Test Accuracy</td>
                <td>80.89%</td>
                <td class="winner">92.00% ✓</td>
            </tr>
            <tr>
                <td>Precision</td>
                <td>0.8125</td>
                <td class="winner">0.9202 ✓</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>0.8089</td>
                <td class="winner">0.9200 ✓</td>
            </tr>
            <tr>
                <td>F1-Score</td>
                <td>0.8084</td>
                <td class="winner">0.9194 ✓</td>
            </tr>
            <tr>
                <td>Final Train Accuracy</td>
                <td>77.36%</td>
                <td class="winner">88.03% ✓</td>
            </tr>
            <tr>
                <td>Final Val Accuracy</td>
                <td>80.44%</td>
                <td class="winner">90.89% ✓</td>
            </tr>
            <tr>
                <td>Training Time</td>
                <td class="winner">~69 min ✓</td>
                <td>~40 min</td>
            </tr>
            <tr>
                <td>Trainable Parameters</td>
                <td class="winner">489,477 ✓</td>
                <td>~2.3M (head only, Phase 1)</td>
            </tr>
            <tr>
                <td>Architecture</td>
                <td>4-block custom CNN</td>
                <td>MobileNetV2 + custom head</td>
            </tr>
            <tr>
                <td>Backbone</td>
                <td>None (from scratch)</td>
                <td>ImageNet pre-trained</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Visual accuracy comparison chart ──────────────────
    st.markdown('<p class="section-title">Visual Comparison</p>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#1a1d27')

    metrics     = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    cnn_scores  = [0.8089, 0.8125, 0.8089, 0.8084]
    tl_scores   = [0.9200, 0.9202, 0.9200, 0.9194]

    x = np.arange(len(metrics))
    width = 0.35

    for ax in axes:
        ax.set_facecolor('#1a1d27')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_edgecolor('#2d3142')
        ax.tick_params(colors='#9ca3af')
        ax.yaxis.label.set_color('#6b7280')

    # Bar chart
    bars1 = axes[0].bar(x - width/2, cnn_scores,  width, label='CNN Scratch',         color='#3b82f6', alpha=0.85)
    bars2 = axes[0].bar(x + width/2, tl_scores,   width, label='Transfer Learning',   color='#4ade80', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, color='#9ca3af', fontsize=9)
    axes[0].set_ylim(0.7, 1.0)
    axes[0].set_title('Metrics Comparison', color='#e8eaf0', fontsize=11, fontweight='bold', pad=10)
    axes[0].legend(facecolor='#1a1d27', edgecolor='#2d3142',
                   labelcolor='#9ca3af', fontsize=8)
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f'{bar.get_height():.2f}', ha='center', va='bottom',
                     color='#9ca3af', fontsize=7)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f'{bar.get_height():.2f}', ha='center', va='bottom',
                     color='#9ca3af', fontsize=7)

    # Training time comparison
    models_names = ['CNN\nScratch', 'Transfer\nLearning']
    times        = [69.0, 40.3]
    bar_colors   = ['#3b82f6', '#4ade80']
    tb = axes[1].bar(models_names, times, color=bar_colors, alpha=0.85, width=0.4)
    axes[1].set_title('Training Time (minutes)', color='#e8eaf0', fontsize=11, fontweight='bold', pad=10)
    axes[1].set_ylabel('Minutes', color='#6b7280', fontsize=9)
    for bar, val in zip(tb, times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val} min', ha='center', va='bottom',
                     color='#e8eaf0', fontsize=9, fontfamily='monospace')
    axes[1].tick_params(colors='#9ca3af')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================
# TAB 3 — CLASS SAMPLES
# ============================================================

with tab3:
    st.markdown('<p class="section-title">About Each Class</p>', unsafe_allow_html=True)

    cols = st.columns(5)
    for i, cls in enumerate(CLASSES):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div style="font-size:2.5rem;">{CLASS_ICONS[cls]}</div>
                <div style="font-family:monospace;color:#4ade80;
                            font-size:0.85rem;margin:8px 0 4px;">{cls.upper()}</div>
                <div style="color:#6b7280;font-size:0.75rem;line-height:1.4;">
                    {CLASS_TIPS[cls]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">How to Get Best Results</p>', unsafe_allow_html=True)

    tips_col1, tips_col2 = st.columns(2)
    with tips_col1:
        st.markdown("""
        <div class="metric-card">
            <div style="color:#4ade80;font-family:monospace;font-size:0.8rem;margin-bottom:8px;">
                ✓ GOOD IMAGES
            </div>
            <ul style="color:#9ca3af;font-size:0.85rem;line-height:1.8;padding-left:1.2rem;">
                <li>Single object, centered in frame</li>
                <li>Good lighting, no shadows</li>
                <li>Plain or simple background</li>
                <li>Clear, in-focus image</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with tips_col2:
        st.markdown("""
        <div class="metric-card">
            <div style="color:#ef4444;font-family:monospace;font-size:0.8rem;margin-bottom:8px;">
                ✗ AVOID
            </div>
            <ul style="color:#9ca3af;font-size:0.85rem;line-height:1.8;padding-left:1.2rem;">
                <li>Multiple objects in one image</li>
                <li>Very dark or blurry photos</li>
                <li>Extremely small objects</li>
                <li>Heavily cropped images</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#2d3142;font-family:monospace;font-size:0.75rem;padding:1rem;">
    Smart Garbage Classification · Computer Vision & Image Processing Final Project<br>
    Models: CNN from Scratch · Transfer Learning (MobileNetV2) · Built with Streamlit
</div>
""", unsafe_allow_html=True)
