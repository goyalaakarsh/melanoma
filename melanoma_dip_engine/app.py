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

# --- PATH SETUP ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'melanoma_dip_engine'))

# Attempt imports
try:
    import image_processing as ip
    import feature_extraction as fe
except ImportError:
    st.error("⚠️ DIP Engine modules not found. Ensure the directory structure is correct.")
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
    model_path = r'melanoma_pipeline_v2\models\melanoma_classifier_opt.pt'
    try:
        return YOLO(model_path)
    except Exception:
        return None

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
    
    # Normalize inputs to 0-1 for chart
    values = [
        min(1.0, features['A_risk']),
        min(1.0, features['B_risk']),
        min(1.0, features['C_risk']),
        min(1.0, features['T_risk'])
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
    results = {}
    
    # 1. Classification
    temp_path = "temp_dark.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR))
    
    if model:
        pred = model(temp_path, verbose=False)[0]
        # Identify "Melanoma" index dynamically
        names = pred.names
        mel_idx = next((k for k, v in names.items() if 'melanoma' in v.lower()), 1)
        
        probs = pred.probs.data.cpu().numpy()
        results['p_mel'] = float(probs[mel_idx])
        results['p_benign'] = 1.0 - results['p_mel']
        
        # Threshold Logic
        results['is_melanoma'] = results['p_mel'] > 0.35
    else:
        results['p_mel'] = 0.0
        results['is_melanoma'] = False

    # 2. DIP Pipeline
    clean_img, _ = ip.remove_hair(image_arr)
    mask, contour, _ = ip.segment_lesion(clean_img)
    
    results['img_clean'] = clean_img
    results['mask'] = mask
    
    # Feature Extraction
    if contour is not None and len(contour) > 0:
        hsv = cv2.cvtColor(clean_img, cv2.COLOR_RGB2HSV)
        feats = fe.extract_all_features(clean_img, hsv, mask, contour)
        
        # Calculate Risk Levels
        results['feats'] = {
            'A_val': feats.get('asymmetry_score', 0),
            'A_risk': min(1.0, feats.get('asymmetry_score', 0) * 10),
            
            'B_val': feats.get('border_irregularity', 0),
            'B_risk': min(1.0, (feats.get('border_irregularity', 1.0) - 1.0) / 5.0),
            
            'C_val': feats.get('color_variation', 1),
            'C_risk': min(1.0, (feats.get('color_variation', 1) - 1) / 4.0),
            
            'T_val': feats.get('glcm_contrast', 0),
            'T_risk': min(1.0, feats.get('glcm_contrast', 0) / 200.0)
        }
    else:
        results['feats'] = None
        
    return results

# --- MAIN APP ---
def main():
    # Header
    st.markdown('<h1 class="main-title">Melanoma DIP Engine</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("System optimized for Dark Mode.")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        st.write("---")
        st.markdown("### 📊 Models")
        st.success("• YOLOv11 Classification")
        st.success("• ABCT Feature Engine")
    
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
    image = Image.open(uploaded_file)
    img_arr = np.array(image)
    model = load_model()
    
    # Run Analysis (only once per image)
    if 'curr_file' not in st.session_state or st.session_state.curr_file != uploaded_file.name:
        with st.spinner("Processing Lesion Structure..."):
            results = analyze_image(img_arr, model)
            st.session_state.results = results
            st.session_state.curr_file = uploaded_file.name
    
    res = st.session_state.results
    
    # ==========================================
    # 1. TOP SECTION: DIAGNOSIS BANNER
    # ==========================================
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
        
    with col2:
        st.write("") # Spacer
        if res['is_melanoma']:
            st.markdown("""
            <div class="risk-banner risk-high">
                <h1>⚠️ HIGH RISK DETECTED</h1>
                <p>AI Confidence > Threshold (0.35)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-banner risk-low">
                <h1>✅ LOW RISK (BENIGN)</h1>
                <p>Features appear stable</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.write("") # Spacer
        st.plotly_chart(create_gauge_chart(res['p_mel'], "Malignancy Probability"), use_container_width=True)

    # ==========================================
    # 2. TABS FOR DETAILED VIEW
    # ==========================================
    st.write("---")
    tab1, tab2, tab3 = st.tabs(["🔬 Clinical Features (ABCT)", "👁️ Visual Pipeline", "📊 Raw Data"])
    
    # --- TAB 1: ABCT ANALYSIS ---
    with tab1:
        if res['feats']:
            c1, c2 = st.columns([1, 1.5])
            
            with c1:
                st.markdown("### 🎯 Risk Profile Radar")
                st.plotly_chart(create_radar_chart(res['feats']), use_container_width=True)
            
            with c2:
                st.markdown("### 📏 Feature Metrics")
                f = res['feats']
                
                # HTML Cards for Metrics (Dark Theme)
                st.markdown(f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div class="metric-card">
                        <h4 style="color:#4facfe; margin:0">A - Asymmetry</h4>
                        <h2 style="margin:5px 0">{f['A_val']:.3f}</h2>
                        <small style="opacity:0.7">Score (0-1.0)</small>
                    </div>
                    <div class="metric-card">
                        <h4 style="color:#4facfe; margin:0">B - Border</h4>
                        <h2 style="margin:5px 0">{f['B_val']:.3f}</h2>
                        <small style="opacity:0.7">Irregularity Score</small>
                    </div>
                    <div class="metric-card">
                        <h4 style="color:#4facfe; margin:0">C - Color</h4>
                        <h2 style="margin:5px 0">{f['C_val']}</h2>
                        <small style="opacity:0.7">Distinct Colors</small>
                    </div>
                    <div class="metric-card">
                        <h4 style="color:#4facfe; margin:0">T - Texture</h4>
                        <h2 style="margin:5px 0">{f['T_val']:.1f}</h2>
                        <small style="opacity:0.7">Contrast Score</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Feature extraction skipped (Lesion boundary undefined).")

    # --- TAB 2: VISUAL PIPELINE ---
    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 1. Original")
            st.image(image, use_container_width=True)
        with c2:
            st.markdown("##### 2. Hair Removal")
            st.image(res['img_clean'], use_container_width=True)
        with c3:
            st.markdown("##### 3. AI Segmentation")
            # Overlay Mask
            overlay = img_arr.copy()
            if res['mask'] is not None:
                overlay[res['mask'] > 0] = [200, 0, 0] # Red Tint
                final = cv2.addWeighted(img_arr, 0.7, overlay, 0.3, 0)
                st.image(final, use_container_width=True)
            else:
                st.error("Segmentation Failed")

    # --- TAB 3: DOWNLOADS ---
    with tab3:
        st.markdown("### 📥 Export Data")
        
        # Prepare CSV
        csv_data = {
            'Filename': uploaded_file.name,
            'Prediction': 'Melanoma' if res['is_melanoma'] else 'Benign',
            'Confidence': res['p_mel'],
        }
        if res['feats']:
            csv_data.update(res['feats'])
            
        df_export = pd.DataFrame([csv_data])
        st.dataframe(df_export, use_container_width=True)
        
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report (CSV)", csv, "report.csv", "text/csv")

if __name__ == "__main__":
    main()