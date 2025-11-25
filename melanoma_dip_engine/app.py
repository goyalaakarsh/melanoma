import os
import sys
import numpy as np
import pandas as pd
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO

# --- PATH SETUP ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'melanoma_dip_engine'))

# Attempt imports
try:
    import image_processing as ip
    import feature_extraction as fe
    import utils
    import config
except ImportError as e:
    st.error(f"⚠️ DIP Engine modules not found: {e}. Ensure the directory structure is correct.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Melanoma AI | Dark Mode",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM DARK THEME CSS ---
st.markdown("""
<style>
    /* Main Background adjustments */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Gradient Title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* Dark Mode Cards (Glassmorphism) */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Result Banners */
    .risk-banner {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
        letter-spacing: 1px;
    }
    
    .risk-high {
        background: linear-gradient(145deg, #3f0d0d 0%, #7f1d1d 100%);
        border: 1px solid #ef4444;
        color: #fca5a5;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.2);
    }
    
    .risk-low {
        background: linear-gradient(145deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        color: #6ee7b7;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
    }
    
    .risk-moderate {
        background: linear-gradient(145deg, #78350f 0%, #92400e 100%);
        border: 1px solid #fbbf24;
        color: #fde68a;
        box-shadow: 0 0 20px rgba(251, 191, 36, 0.2);
    }

    /* Text Colors */
    h1, h2, h3, p, li {
        color: #e2e8f0 !important;
    }
    
    /* Remove default streamlit margins */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Load YOLO classification model"""
    model_path = os.path.join(os.path.dirname(__file__), 'yolo-models', 'melanoma_classifier_opt.pt')
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

# --- HELPER FUNCTIONS ---
def fig_to_image(fig):
    """Convert matplotlib figure to image array for Streamlit"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0e1117', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Close to free memory
    return np.array(img)

def visualize_grabcut_for_streamlit(image, initial_mask, refined_mask, metrics):
    """Wrapper for visualize_grabcut_comparison that returns image array"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('GrabCut Segmentation Refinement', fontsize=16, fontweight='bold')
    fig.patch.set_facecolor('#0e1117')
    
    axes[0, 0].imshow(image)
    axes[0, 0].imshow(initial_mask, cmap='jet', alpha=0.4)
    axes[0, 0].set_title('Initial Segmentation', fontweight='bold', color='white')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(refined_mask, cmap='jet', alpha=0.4)
    axes[0, 1].set_title('GrabCut Refined', fontweight='bold', color='white')
    axes[0, 1].axis('off')
    
    diff = cv2.absdiff(initial_mask, refined_mask)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Refinement Changes', fontweight='bold', color='white')
    axes[0, 2].axis('off')
    
    overlay_init = image.copy()
    overlay_init[initial_mask > 0] = [255, 0, 0]
    blended_init = cv2.addWeighted(image, 0.6, overlay_init, 0.4, 0)
    axes[1, 0].imshow(blended_init)
    axes[1, 0].set_title('Initial Overlay', fontweight='bold', color='white')
    axes[1, 0].axis('off')
    
    overlay_ref = image.copy()
    overlay_ref[refined_mask > 0] = [0, 255, 0]
    blended_ref = cv2.addWeighted(image, 0.6, overlay_ref, 0.4, 0)
    axes[1, 1].imshow(blended_ref)
    axes[1, 1].set_title('Refined Overlay', fontweight='bold', color='white')
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')
    metrics_text = f'''
    GRABCUT REFINEMENT METRICS
    ===================================
    
    Applied: {metrics.get("grabcut_applied", False)}
    Iterations: {metrics.get("grabcut_iterations", 0)}
    
    Area Changes:
       Initial: {metrics.get("initial_area", 0):,} pixels
       Refined: {metrics.get("refined_area", 0):,} pixels
       Change: {metrics.get("area_change_percent", 0):.2f}%
    '''
    
    axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                   fontsize=11, verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   color='white')
    
    plt.tight_layout()
    return fig_to_image(fig)

def visualize_color_correction_for_streamlit(original, corrected):
    """Wrapper for visualize_color_correction_comparison that returns image array"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Color Constancy Correction', fontsize=16, fontweight='bold')
    fig.patch.set_facecolor('#0e1117')
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontweight='bold', color='white')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(corrected)
    axes[0, 1].set_title('Color-Corrected Image', fontweight='bold', color='white')
    axes[0, 1].axis('off')
    
    diff = cv2.absdiff(original, corrected)
    axes[0, 2].imshow(diff)
    axes[0, 2].set_title('Absolute Difference', fontweight='bold', color='white')
    axes[0, 2].axis('off')
    
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist_orig = cv2.calcHist([original], [i], None, [256], [0, 256])
        axes[1, 0].plot(hist_orig, color=color, alpha=0.7, label=f'{color.upper()} channel')
    axes[1, 0].set_title('Original Histograms', fontweight='bold', color='white')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].tick_params(colors='white')
    axes[1, 0].spines['bottom'].set_color('white')
    axes[1, 0].spines['top'].set_color('white')
    axes[1, 0].spines['left'].set_color('white')
    axes[1, 0].spines['right'].set_color('white')
    
    for i, color in enumerate(colors):
        hist_corr = cv2.calcHist([corrected], [i], None, [256], [0, 256])
        axes[1, 1].plot(hist_corr, color=color, alpha=0.7, label=f'{color.upper()} channel')
    axes[1, 1].set_title('Corrected Histograms', fontweight='bold', color='white')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].tick_params(colors='white')
    axes[1, 1].spines['bottom'].set_color('white')
    axes[1, 1].spines['top'].set_color('white')
    axes[1, 1].spines['left'].set_color('white')
    axes[1, 1].spines['right'].set_color('white')
    
    mean_orig = [np.mean(original[:, :, i]) for i in range(3)]
    mean_corr = [np.mean(corrected[:, :, i]) for i in range(3)]
    
    x = np.arange(3)
    width = 0.35
    axes[1, 2].bar(x - width/2, mean_orig, width, label='Original', color=['red', 'green', 'blue'], alpha=0.7)
    axes[1, 2].bar(x + width/2, mean_corr, width, label='Corrected', color=['darkred', 'darkgreen', 'darkblue'], alpha=0.7)
    axes[1, 2].set_title('Channel Mean Values', fontweight='bold', color='white')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(['Red', 'Green', 'Blue'])
    axes[1, 2].legend()
    axes[1, 2].tick_params(colors='white')
    axes[1, 2].spines['bottom'].set_color('white')
    axes[1, 2].spines['top'].set_color('white')
    axes[1, 2].spines['left'].set_color('white')
    axes[1, 2].spines['right'].set_color('white')
    
    plt.tight_layout()
    return fig_to_image(fig)

def get_risk_level(score):
    """Get risk level based on score"""
    if score < 0.3:
        return ("LOW", "#10b981")
    elif score < 0.7:
        return ("MODERATE", "#fbbf24")
    else:
        return ("HIGH", "#ef4444")

# --- CHART FUNCTIONS (Dark Mode Optimized) ---
def create_gauge_chart(value, title):
    """Semi-circle gauge for dark mode"""
    # Dynamic Color
    if value < 0.3: color = "#34d399" # Green
    elif value < 0.7: color = "#fbbf24" # Orange
    else: color = "#f87171" # Red
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 14, 'color': "#94a3b8"}},
        number = {'suffix': "%", 'font': {'size': 24, 'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'visible': False},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.1)",
            'steps': [{'range': [0, 100], 'color': "rgba(0,0,0,0)"}],
        }
    ))
    fig.update_layout(
        height=150, 
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_radar_chart(features):
    """Radar chart for ABCT metrics"""
    categories = ['Asymmetry', 'Border', 'Color', 'Texture']
    
    # Use normalized scores from feature extraction
    values = [
        min(1.0, features.get('asymmetry_score', 0)),
        min(1.0, features.get('border_irregularity_score', 0)),
        min(1.0, features.get('color_variation_score', 0)),
        min(1.0, features.get('texture_contrast_score', 0))
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(79, 172, 254, 0.3)',
        line=dict(color='#00f2fe', width=2),
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showline=False, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='gray')),
            angularaxis=dict(tickfont=dict(size=12, color='#e2e8f0'), gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        showlegend=False
    )
    return fig

# --- ANALYSIS ENGINE ---
def analyze_image(image_arr, model):
    """
    Complete analysis pipeline following notebook flow:
    load_and_preprocess → remove_hair → segment_lesion → refine_segmentation_grabcut → extract_all_features
    """
    results = {}
    
    try:
        # Step 1: YOLO Classification
        if model is not None:
            # Save temporary image for YOLO
            temp_path = "temp_dark.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR))
            
            try:
                yolo_results = model(temp_path, verbose=False)[0]
                probs = yolo_results.probs.data.cpu().numpy()
                
                # Find class indices
                class_names = yolo_results.names
                if 'melanoma' in class_names.values():
                    mel_idx = list(class_names.values()).index('melanoma')
                elif len(probs) == 2:
                    mel_idx = 0  # Assume first class is melanoma
                else:
                    mel_idx = 0
                
                results['p_mel'] = float(probs[mel_idx])
                results['p_benign'] = 1.0 - results['p_mel']
                results['is_melanoma'] = results['p_mel'] > 0.35
            except Exception as e:
                st.warning(f"YOLO classification failed: {e}")
                results['p_mel'] = 0.0
                results['p_benign'] = 1.0
                results['is_melanoma'] = False
        else:
            results['p_mel'] = 0.0
            results['p_benign'] = 1.0
            results['is_melanoma'] = False
        
        # Step 2: Preprocessing (following notebook)
        rgb_image, hsv_image, lab_image = preprocess_from_array(image_arr)
        results['rgb_preprocessed'] = rgb_image
        results['hsv_image'] = hsv_image
        
        # Step 3: Hair Removal
        hair_free_image, hair_metrics = ip.remove_hair(rgb_image)
        results['hair_free'] = hair_free_image
        results['hair_metrics'] = hair_metrics
        
        # Step 4: Initial Segmentation
        initial_mask, initial_contour, seg_metrics = ip.segment_lesion(hair_free_image)
        results['initial_mask'] = initial_mask
        results['initial_contour'] = initial_contour
        results['seg_metrics'] = seg_metrics
        
        # Step 5: GrabCut Refinement
        binary_mask, main_contour, grabcut_metrics = ip.refine_segmentation_grabcut(
            hair_free_image, initial_mask, initial_contour
        )
        results['mask'] = binary_mask
        results['contour'] = main_contour
        results['grabcut_metrics'] = grabcut_metrics
        
        # Step 6: Feature Extraction (Complete)
        if main_contour is not None and np.sum(binary_mask) > 0:
            features = fe.extract_all_features(
                original_image=hair_free_image,
                hsv_image=hsv_image,
                mask=binary_mask,
                contour=main_contour
            )
            results['features'] = features
            results['features_extracted'] = True
        else:
            results['features'] = {}
            results['features_extracted'] = False
            st.warning("⚠️ Lesion segmentation failed - feature extraction skipped")
        
        results['success'] = True
        
    except Exception as e:
        st.error(f"❌ Error in analysis pipeline: {e}")
        import traceback
        st.code(traceback.format_exc())
        results['success'] = False
        results['error'] = str(e)
    
    return results

def preprocess_from_array(image_arr):
    """Preprocess image array directly without file I/O"""
    # Convert to BGR if needed (OpenCV format)
    if len(image_arr.shape) == 3:
        bgr_image = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
    else:
        bgr_image = image_arr
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # Resize to standard size
    rgb_resized = cv2.resize(rgb_image, config.IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
    
    # Apply CLAHE contrast enhancement if enabled
    if config.CONTRAST_ENHANCEMENT:
        lab_temp = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_SIZE)
        lab_temp[:, :, 0] = clahe.apply(lab_temp[:, :, 0])
        rgb_resized = cv2.cvtColor(lab_temp, cv2.COLOR_LAB2RGB)
    
    # Convert to HSV and LAB color spaces
    hsv_image = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2HSV)
    lab_image = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2LAB)
    
    return rgb_resized, hsv_image, lab_image

# --- MAIN APP ---
def main():
    # Header
    st.markdown('<h1 class="main-title">Melanoma Risk Prediction for Triage & Decision Support</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'bmp'])
        
        st.write("---")
        st.markdown("### 📊 Models")
        st.success("• YOLO11 Classification")
        st.success("• ABCT Feature Engine")
        st.success("• Advanced DIP (GrabCut, FFT, BWV)")
    
    # Main Logic
    if not uploaded_file:
        st.markdown("""
        <div style="text-align: center; margin-top: 50px; opacity: 0.7;">
            <h3>Waiting for Upload...</h3>
            <p>Upload a dermoscopic image to begin differential diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Load resources
    try:
        image = Image.open(uploaded_file)
        img_arr = np.array(image)
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return
    
    # Run Analysis (only once per image)
    if 'curr_file' not in st.session_state or st.session_state.curr_file != uploaded_file.name:
        with st.spinner("🔄 Processing Lesion Structure... This may take a moment."):
            results = analyze_image(img_arr, model)
            st.session_state.results = results
            st.session_state.curr_file = uploaded_file.name
            st.session_state.original_image = img_arr
    
    res = st.session_state.results
    img_arr = st.session_state.original_image
    
    if not res.get('success', False):
        st.error("❌ Analysis failed. Please try again or check the image format.")
        if 'error' in res:
            st.code(res['error'])
        return
    
    # ==========================================
    # 1. TOP SECTION: DIAGNOSIS BANNER
    # ==========================================
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
        
    with col2:
        st.write("") # Spacer
        # Calculate risk based on features only, not YOLO classification
        overall_risk = 0.0
        if res.get('features_extracted', False):
            features = res['features']
            asym_score = features.get('asymmetry_score', 0)
            border_score = features.get('border_irregularity_score', 0)
            color_score = features.get('color_variation_score', 0)
            texture_score = features.get('texture_contrast_score', 0)
            overall_risk = (asym_score + border_score + color_score + texture_score) / 4.0
            risk_level, risk_color = get_risk_level(overall_risk)
            
            if risk_level == "HIGH":
                st.markdown("""
                <div class="risk-banner risk-high">
                    <h1>⚠️ HIGH RISK DETECTED</h1>
                </div>
                """, unsafe_allow_html=True)
            elif risk_level == "MODERATE":
                st.markdown("""
                <div class="risk-banner risk-moderate">
                    <h1>⚠️ MODERATE RISK DETECTED</h1>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="risk-banner risk-low">
                    <h1>✅ LOW RISK</h1>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-banner risk-low">
                <h1>⏳ PROCESSING...</h1>
                <p>Feature extraction in progress</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.write("") # Spacer
        # Show overall risk from features in gauge, not YOLO confidence
        st.plotly_chart(create_gauge_chart(overall_risk, "Risk Score"), use_container_width=True)

    # ==========================================
    # 2. TABS FOR DETAILED VIEW
    # ==========================================
    # st.write("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔬 Clinical Features (ABCD+T)", 
        "🔬 Advanced DIP Features",
        "👁️ Visual Pipeline", 
        "📥 Detailed Metrics"
    ])
    
    # --- TAB 1: OVERVIEW ---
    with tab1:
        if res.get('features_extracted', False):
            features = res['features']
            
            # Key Metrics Summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                asym_score = features.get('asymmetry_score', 0)
                risk_level, risk_color = get_risk_level(asym_score)
                st.metric("Asymmetry", f"{asym_score:.3f}", delta=None)
                st.markdown(f"<small style='color:{risk_color}'>Risk: {risk_level}</small>", unsafe_allow_html=True)
            
            with col2:
                border_score = features.get('border_irregularity_score', 0)
                risk_level, risk_color = get_risk_level(border_score)
                st.metric("Border Irregularity", f"{border_score:.3f}", delta=None)
                st.markdown(f"<small style='color:{risk_color}'>Risk: {risk_level}</small>", unsafe_allow_html=True)
            
            with col3:
                color_score = features.get('color_variation_score', 0)
                color_count = features.get('color_variation', 1)
                risk_level, risk_color = get_risk_level(color_score)
                st.metric("Color Variation", f"{color_count} colors", delta=None)
                st.markdown(f"<small style='color:{risk_color}'>Risk: {risk_level}</small>", unsafe_allow_html=True)
            
            with col4:
                texture_score = features.get('texture_contrast_score', 0)
                risk_level, risk_color = get_risk_level(texture_score)
                st.metric("Texture Contrast", f"{texture_score:.3f}", delta=None)
                st.markdown(f"<small style='color:{risk_color}'>Risk: {risk_level}</small>", unsafe_allow_html=True)
            
            # Overall Risk Calculation
            overall_risk = (asym_score + border_score + color_score + texture_score) / 4.0
            risk_level, risk_color = get_risk_level(overall_risk)
            
    
            # Pipeline Images
            st.write("---")
            st.markdown("### 🔄 Processing Pipeline")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("##### 1. Original Image")
                st.image(img_arr, use_container_width=True)
            
            with col2:
                st.markdown("##### 2. Color Corrected")
                if 'rgb_preprocessed' in res:
                    st.image(res['rgb_preprocessed'], use_container_width=True)
            
            with col3:
                st.markdown("##### 3. Hair Removal")
                if 'hair_free' in res:
                    st.image(res['hair_free'], use_container_width=True)
            
            with col4:
                st.markdown("##### 4. Segmentation")
                if 'mask' in res and res['mask'] is not None:
                    overlay = utils.create_overlay_image(res['hair_free'], res['mask'])
                    st.image(overlay, use_container_width=True)
        else:
            st.warning("⚠️ Feature extraction incomplete. Check segmentation results.")
    
    # --- TAB 2: CLINICAL FEATURES (ABCD+T) ---
    with tab2:
        if res.get('features_extracted', False):
            features = res['features']
            
            # Radar Chart
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.markdown("### 🎯 Risk Profile Radar")
                st.plotly_chart(create_radar_chart(features), use_container_width=True)
            
            with col2:
                st.markdown("### 📏 Feature Metrics")
                
                # Feature Table
                feature_data = {
                    'Feature': ['Asymmetry', 'Border Irregularity', 'Color Variation', 'Texture Contrast'],
                    'Raw Value': [
                        f"{features.get('asymmetry', 0):.3f}",
                        f"{features.get('border_irregularity', 0):.3f}",
                        f"{int(features.get('color_variation', 1))} colors",
                        f"{features.get('glcm_contrast', 0):.1f}"
                    ],
                    'Normalized Score': [
                        f"{features.get('asymmetry_score', 0):.3f}",
                        f"{features.get('border_irregularity_score', 0):.3f}",
                        f"{features.get('color_variation_score', 0):.3f}",
                        f"{features.get('texture_contrast_score', 0):.3f}"
                    ],
                    'Risk Level': [
                        get_risk_level(features.get('asymmetry_score', 0))[0],
                        get_risk_level(features.get('border_irregularity_score', 0))[0],
                        get_risk_level(features.get('color_variation_score', 0))[0],
                        get_risk_level(features.get('texture_contrast_score', 0))[0]
                    ]
                }
                df_features = pd.DataFrame(feature_data)
                st.dataframe(df_features, use_container_width=True, hide_index=True)
            
            # Individual Feature Visualizations
            st.write("---")
            st.markdown("### 🔍 Detailed Feature Visualizations")
            
            # Asymmetry
            if 'mask' in res and res['mask'] is not None:
                try:
                    asym_viz = utils.visualize_asymmetry(res['mask'])
                    st.markdown("#### (A) Asymmetry Analysis")
                    st.image(asym_viz, use_container_width=True)
                    st.caption(f"Asymmetry Score: {features.get('asymmetry', 0):.3f}")
                except Exception as e:
                    st.warning(f"Asymmetry visualization failed: {e}")
            
            # Border
            if 'contour' in res and res['contour'] is not None:
                try:
                    border_viz = utils.visualize_border(res['hair_free'], res['contour'])
                    st.markdown("#### (B) Border Irregularity")
                    st.image(border_viz, use_container_width=True)
                    st.caption(f"Border Irregularity: {features.get('border_irregularity', 0):.3f}")
                except Exception as e:
                    st.warning(f"Border visualization failed: {e}")
            
            # Color
            if 'mask' in res and res['mask'] is not None:
                try:
                    color_count = int(features.get('color_variation', 1))
                    color_viz = utils.visualize_color_clusters(res['hair_free'], res['mask'], color_count)
                    st.markdown("#### (C) Color Variation")
                    st.image(color_viz, use_container_width=True)
                    st.caption(f"Distinct Colors: {color_count}")
                except Exception as e:
                    st.warning(f"Color visualization failed: {e}")
            
            # Texture
            if 'mask' in res and res['mask'] is not None:
                try:
                    texture_viz = utils.visualize_texture(res['hair_free'], res['mask'])
                    st.markdown("#### (T) Texture Analysis")
                    st.image(texture_viz, use_container_width=True)
                    st.caption(f"GLCM Contrast: {features.get('glcm_contrast', 0):.1f}")
                except Exception as e:
                    st.warning(f"Texture visualization failed: {e}")
            
            # Diameter
            if 'mask' in res and res['mask'] is not None:
                try:
                    diameter_viz = utils.visualize_diameter(res['hair_free'], res['mask'], features)
                    st.markdown("#### (D) Diameter Measurement")
                    st.image(diameter_viz, use_container_width=True)
                    st.caption(f"Largest Diameter: {features.get('largest_diameter_mm', 0):.2f} mm")
                except Exception as e:
                    st.warning(f"Diameter visualization failed: {e}")
        else:
            st.warning("⚠️ Feature extraction incomplete. Check segmentation results.")
    
    # --- TAB 3: ADVANCED DIP FEATURES ---
    with tab3:
        if res.get('features_extracted', False):
            features = res['features']
            
            # Advanced Features Overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔬 FFT Frequency Analysis")
                fft_high = features.get('fft_high_frequency_energy', 0)
                fft_ratio = features.get('fft_high_low_ratio', 0)
                st.metric("High-Frequency Energy", f"{fft_high:.4f}")
                st.metric("High/Low Ratio", f"{fft_ratio:.4f}")
                if fft_high > 0.5:
                    st.warning("⚠️ High texture complexity detected")
                else:
                    st.success("✅ Smooth texture pattern")
            
            with col2:
                st.markdown("### 🔵 Blue-White Veil Detection")
                bwv_present = features.get('blue_white_veil_present', 0)
                bwv_coverage = features.get('blue_white_veil_coverage_percentage', 0)
                bwv_confidence = features.get('blue_white_veil_confidence', 0)
                
                if bwv_present:
                    st.error("🚨 BLUE-WHITE VEIL DETECTED")
                    st.metric("Coverage", f"{bwv_coverage:.2f}%")
                    st.metric("Confidence", f"{bwv_confidence:.3f}")
                    st.caption("Blue-white veil is a melanoma indicator")
                else:
                    st.success("✅ No blue-white veil detected")
                    st.metric("Coverage", f"{bwv_coverage:.2f}%")
            
            # Advanced Visualizations
            st.write("---")
            st.markdown("### 📊 Advanced Feature Visualizations")
            
            # Advanced Features Dashboard
            if 'mask' in res and res['mask'] is not None:
                try:
                    advanced_viz = utils.visualize_advanced_features(res['hair_free'], res['mask'], features)
                    st.image(advanced_viz, use_container_width=True)
                except Exception as e:
                    st.warning(f"Advanced features visualization failed: {e}")
            
            # GrabCut Comparison
            if 'initial_mask' in res and 'mask' in res:
                try:
                    st.markdown("#### 🔧 GrabCut Refinement Comparison")
                    grabcut_viz = visualize_grabcut_for_streamlit(
                        res['hair_free'], 
                        res['initial_mask'], 
                        res['mask'], 
                        res.get('grabcut_metrics', {})
                    )
                    st.image(grabcut_viz, use_container_width=True)
                except Exception as e:
                    st.warning(f"GrabCut visualization failed: {e}")
            
            # Color Constancy Comparison
            if 'rgb_preprocessed' in res:
                try:
                    st.markdown("#### 🎨 Color Constancy Correction")
                    # Resize original to match preprocessed size
                    original_resized = cv2.resize(img_arr, config.IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
                    color_corr_viz = visualize_color_correction_for_streamlit(
                        original_resized, 
                        res['rgb_preprocessed']
                    )
                    st.image(color_corr_viz, use_container_width=True)
                except Exception as e:
                    st.warning(f"Color correction visualization failed: {e}")
        else:
            st.warning("⚠️ Feature extraction incomplete. Check segmentation results.")
    
    # --- TAB 4: VISUAL PIPELINE ---
    with tab4:
        st.markdown("### 🔄 Complete Processing Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 1. Original Image")
            st.image(img_arr, use_container_width=True)
            
            if 'rgb_preprocessed' in res:
                st.markdown("##### 2. Preprocessed (Color Corrected)")
                st.image(res['rgb_preprocessed'], use_container_width=True)
        
        with col2:
            if 'hair_free' in res:
                st.markdown("##### 3. Hair Removal")
                st.image(res['hair_free'], use_container_width=True)
            
            if 'initial_mask' in res:
                st.markdown("##### 4. Initial Segmentation")
                st.image(res['initial_mask'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'mask' in res and res['mask'] is not None:
                st.markdown("##### 5. GrabCut Refined Mask")
                st.image(res['mask'], use_container_width=True)
        
        with col2:
            if 'mask' in res and res['mask'] is not None and 'hair_free' in res:
                st.markdown("##### 6. Final Overlay")
                overlay = utils.create_overlay_image(res['hair_free'], res['mask'])
                st.image(overlay, use_container_width=True)
        
        # Quality Metrics
        st.write("---")
        st.markdown("### 📊 Segmentation Quality Metrics")
        
        if 'seg_metrics' in res:
            seg_metrics = res['seg_metrics']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                confidence = seg_metrics.get('confidence_score', 0)
                st.metric("Confidence", f"{confidence:.3f}")
            
            with col2:
                area_pct = seg_metrics.get('area_percentage', 0)
                st.metric("Lesion Coverage", f"{area_pct:.2f}%")
            
            with col3:
                num_contours = seg_metrics.get('num_contours_found', 0)
                st.metric("Contours Found", num_contours)
            
            with col4:
                if 'grabcut_metrics' in res and res['grabcut_metrics'].get('grabcut_applied', False):
                    st.success("✅ GrabCut Applied")
                else:
                    st.info("ℹ️ GrabCut Skipped")
    
    # --- TAB 5: DETAILED METRICS ---
    with tab5:
        st.markdown("### 📥 Complete Feature Export")
        
        if res.get('features_extracted', False):
            features = res['features']
            
            # Prepare comprehensive CSV
            csv_data = {
                'Filename': uploaded_file.name,
                'YOLO_Prediction': 'Melanoma' if res.get('is_melanoma', False) else 'Benign',
                'YOLO_Confidence': res.get('p_mel', 0.0),
            }
            
            # Add all features
            csv_data.update(features)
            
            # Add segmentation metrics
            if 'seg_metrics' in res:
                csv_data['segmentation_confidence'] = res['seg_metrics'].get('confidence_score', 0)
                csv_data['lesion_area_percentage'] = res['seg_metrics'].get('area_percentage', 0)
            
            # Add GrabCut metrics
            if 'grabcut_metrics' in res:
                csv_data['grabcut_applied'] = res['grabcut_metrics'].get('grabcut_applied', False)
                csv_data['grabcut_area_change'] = res['grabcut_metrics'].get('area_change_percent', 0)
            
            df_export = pd.DataFrame([csv_data])
            
            st.markdown("#### 📊 Feature Summary Table")
            st.dataframe(df_export, use_container_width=True, height=400)
            
            # Download button
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Complete Report (CSV)", 
                csv, 
                f"melanoma_report_{uploaded_file.name}.csv", 
                "text/csv",
                key='download-csv'
            )
            
            # Raw features display
            with st.expander("🔍 View Raw Feature Dictionary"):
                st.json(features)
        else:
            st.warning("⚠️ No features available for export.")

if __name__ == "__main__":
    main()
