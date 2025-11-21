"""
🔬 Melanoma Detection & Analysis System
=======================================
Advanced AI-Powered Skin Lesion Analysis Platform
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

# Add melanoma_dip_engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'melanoma_dip_engine'))

import src.image_processing as ip
import src.feature_extraction as fe

# Page configuration
st.set_page_config(
    page_title="Melanoma Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        margin: 0.5rem 0;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Risk level badges */
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        padding: 0.75rem 2rem;
        border-radius: 30px;
        color: white;
        font-weight: 700;
        display: inline-block;
        font-size: 1.3rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        padding: 0.75rem 2rem;
        border-radius: 30px;
        color: white;
        font-weight: 700;
        display: inline-block;
        font-size: 1.3rem;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 0.75rem 2rem;
        border-radius: 30px;
        color: white;
        font-weight: 700;
        display: inline-block;
        font-size: 1.3rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px dashed #667eea;
    }
    
    /* Stremlit button customization */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: transform 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    .feature-card h4 {
        color: #667eea;
        margin-top: 0;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        margin: 0.5rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #333;
    }
    
    .feature-card strong {
        color: #667eea;
        font-weight: 600;
    }
    
    .feature-card i {
        color: #6c757d;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Model paths
MODEL_CLS_PATH = r'melanoma_pipeline_v2\models\melanoma_classifier_opt.pt'

@st.cache_resource
def load_model():
    """Load the classification model"""
    try:
        model = YOLO(MODEL_CLS_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_gauge_chart(value, title, max_value=1.0):
    """Create a beautiful gauge chart for risk metrics"""
    # Determine bar color based on value
    if value < 0.33:
        bar_color = "#10b981"  # Green
    elif value < 0.66:
        bar_color = "#f59e0b"  # Amber
    else:
        bar_color = "#ef4444"  # Red
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18, 'color': '#667eea', 'family': 'Arial'}},
        number = {'font': {'size': 32, 'color': bar_color, 'family': 'Arial'}, 'valueformat': '.3f'},
        gauge = {
            'axis': {
                'range': [None, max_value], 
                'tickwidth': 2, 
                'tickcolor': "#667eea",
                'tickfont': {'size': 12}
            },
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#667eea",
            'steps': [
                {'range': [0, max_value*0.33], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [max_value*0.33, max_value*0.66], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [max_value*0.66, max_value], 'color': 'rgba(239, 68, 68, 0.15)'}
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 5},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Arial"}
    )
    
    return fig

def create_risk_radar_chart(asymmetry, border, color, texture):
    """Create radar chart for ABCT analysis"""
    categories = ['Asymmetry', 'Border', 'Color', 'Texture']
    values = [asymmetry, border, color, texture]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Risk Factors',
        line=dict(color='#667eea', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showline=False,
                showgrid=True,
                gridcolor='rgba(102, 126, 234, 0.2)'
            ),
            angularaxis=dict(
                showline=True,
                linecolor='rgba(102, 126, 234, 0.3)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14, color='#333')
    )
    
    return fig

def analyze_image(image_array, cls_model):
    """Perform complete melanoma analysis pipeline"""
    results = {}
    
    # Save temporary image for YOLO
    temp_path = "temp_upload.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    
    # ============================================================================
    # STEP 1: Classification
    # ============================================================================
    with st.spinner("🔍 Analyzing image with AI classifier..."):
        cls_results = cls_model(temp_path, verbose=False)[0]
        
        probs = cls_results.probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        top1_name = cls_results.names[top1_idx]
        
        all_probs = probs.data.cpu().numpy()
        
        melanoma_idx = 0 if cls_results.names[0].upper() == 'MELANOMA' else 1
        not_melanoma_idx = 1 - melanoma_idx
        
        melanoma_conf = all_probs[melanoma_idx]
        not_melanoma_conf = all_probs[not_melanoma_idx]
        
        is_melanoma = top1_name.upper() == 'MELANOMA'
        
        results['predicted_class'] = top1_name.upper()
        results['classification_confidence'] = top1_conf
        results['melanoma_probability'] = melanoma_conf
        results['not_melanoma_probability'] = not_melanoma_conf
        results['is_melanoma'] = is_melanoma
        results['decision_margin'] = abs(melanoma_conf - not_melanoma_conf)
    
    # ============================================================================
    # STEP 2: Detailed Analysis (if melanoma detected)
    # ============================================================================
    if is_melanoma:
        with st.spinner("🔬 Performing detailed lesion analysis..."):
            try:
                # Load and preprocess
                rgb_image = image_array
                hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
                
                # Hair removal
                hair_free_image, quality_metrics = ip.remove_hair(rgb_image)
                
                # Segmentation
                binary_mask, main_contour, seg_metrics = ip.segment_lesion(hair_free_image)
                
                # Feature extraction (ABCT)
                features = fe.extract_all_features(
                    original_image=hair_free_image,
                    hsv_image=hsv_image,
                    mask=binary_mask,
                    contour=main_contour
                )
                
                # Extract features
                asymmetry_score = features.get('asymmetry_score', 0)
                border_irregularity = features.get('border_irregularity', 0)
                color_variation = features.get('color_variation', 0)
                texture_contrast = features.get('glcm_contrast', 0)
                texture_homogeneity = features.get('glcm_homogeneity', 0)
                texture_energy = features.get('glcm_energy', 0)
                texture_correlation = features.get('glcm_correlation', 0)
                
                # Calculate risk scores (normalized 0-1)
                asymmetry_risk = min(1.0, asymmetry_score * 10)
                border_risk = min(1.0, (border_irregularity - 1.0) / 5.0)
                color_risk = min(1.0, (color_variation - 1) / 4.0)
                texture_risk = min(1.0, texture_contrast / 200.0)
                
                # Overall risk
                overall_risk = (asymmetry_risk + border_risk + color_risk + texture_risk) / 4.0
                
                # Determine risk level
                if overall_risk < 0.3:
                    risk_level = "LOW"
                elif overall_risk < 0.7:
                    risk_level = "MODERATE"
                else:
                    risk_level = "HIGH"
                
                # Calculate segmentation quality properly
                if binary_mask is not None and main_contour is not None:
                    seg_quality = min(1.0, len(main_contour) / 1000.0)  # Quality based on contour complexity
                else:
                    seg_quality = seg_metrics.get('confidence', 0.5) if seg_metrics else 0.5
                
                # Store all results
                results.update({
                    'analysis_performed': True,
                    'hair_free_image': hair_free_image,
                    'binary_mask': binary_mask,
                    'asymmetry_score': asymmetry_score,
                    'asymmetry_risk': asymmetry_risk,
                    'border_irregularity': border_irregularity,
                    'border_risk': border_risk,
                    'color_variation': color_variation,
                    'color_risk': color_risk,
                    'texture_contrast': texture_contrast,
                    'texture_homogeneity': texture_homogeneity,
                    'texture_energy': texture_energy,
                    'texture_correlation': texture_correlation,
                    'texture_risk': texture_risk,
                    'overall_risk_score': overall_risk,
                    'risk_level': risk_level,
                    'segmentation_quality': seg_quality,
                    'lesion_area_pixels': np.sum(binary_mask > 0) if binary_mask is not None else 0,
                })
                
            except Exception as e:
                results.update({
                    'analysis_performed': False,
                    'analysis_error': str(e)
                })
    else:
        results['analysis_performed'] = False
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return results

def main():
    # Header
    st.markdown('<h1 class="main-title">🔬 Melanoma Detection & Analysis System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-Powered Skin Lesion Classification & Risk Assessment</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/microscope.png", width=80)
        st.markdown("### 📋 About This System")
        st.markdown("""
        This advanced diagnostic tool uses:
        - **Deep Learning Classification** for melanoma detection
        - **ABCT Analysis** for comprehensive risk assessment
        - **Computer Vision** for lesion segmentation
        - **Feature Extraction** for detailed metrics
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Components")
        st.markdown("""
        **A** - Asymmetry Analysis  
        **B** - Border Irregularity  
        **C** - Color Variation  
        **T** - Texture Patterns
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.info("This tool is for research purposes only. Always consult a healthcare professional for medical diagnosis.")
    
    # Load model
    cls_model = load_model()
    
    if cls_model is None:
        st.error("❌ Failed to load the classification model. Please check the model path.")
        return
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Skin Lesion Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG, JPEG)",
        type=['jpg', 'png', 'jpeg'],
        help="Upload a clear image of the skin lesion for analysis"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width='stretch')
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
                # Convert to array
                image_array = np.array(image)
                if image_array.shape[-1] == 4:  # RGBA
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                
                # Perform analysis
                with st.spinner("Processing... Please wait"):
                    results = analyze_image(image_array, cls_model)
                    st.session_state.results = results
                    st.session_state.analysis_complete = True
                    st.session_state.original_image = image_array
        
        # Display results
        if st.session_state.analysis_complete and st.session_state.results:
            results = st.session_state.results
            
            st.markdown("---")
            st.markdown('<h2 class="section-header">📊 Classification Results</h2>', unsafe_allow_html=True)
            
            # Classification metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p>Predicted Class</p>
                    <h3>{results['predicted_class']}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <p>Confidence</p>
                    <h3>{results['classification_confidence']*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <p>Decision Margin</p>
                    <h3>{results['decision_margin']*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability distribution
            st.markdown("### 📈 Class Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': ['Melanoma', 'Not Melanoma'],
                'Probability': [results['melanoma_probability'], results['not_melanoma_probability']]
            })
            
            fig = px.bar(
                prob_df,
                x='Class',
                y='Probability',
                color='Class',
                color_discrete_map={'Melanoma': '#ef4444', 'Not Melanoma': '#10b981'},
                text='Probability'
            )
            fig.update_traces(
                texttemplate='%{text:.1%}', 
                textposition='outside',
                marker=dict(line=dict(color='#fff', width=2))
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                yaxis_title="Probability",
                xaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=14),
                yaxis=dict(
                    gridcolor='rgba(128,128,128,0.2)',
                    range=[0, 1.1]
                )
            )
            st.plotly_chart(fig, use_container_width=True, key="prob_dist_chart")
            
            # Detailed Analysis (if melanoma detected)
            if results['is_melanoma'] and results.get('analysis_performed', False):
                st.markdown("---")
                st.markdown('<h2 class="section-header">🔬 Detailed Lesion Analysis</h2>', unsafe_allow_html=True)
                
                # Risk assessment
                risk_level = results['risk_level']
                risk_score = results['overall_risk_score']
                
                risk_class = f"risk-{risk_level.lower()}"
                st.markdown(f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <h3>Overall Risk Assessment</h3>
                    <div class="{risk_class}" style="margin: 1rem auto; width: fit-content;">
                        {risk_level} RISK - Score: {risk_score:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # ABCT Metrics - Gauge Charts
                st.markdown("### 📊 ABCT Risk Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_asym = create_gauge_chart(
                        results['asymmetry_risk'],
                        "A - Asymmetry Risk"
                    )
                    st.plotly_chart(fig_asym, use_container_width=True, key="asym_gauge")
                    
                    fig_color = create_gauge_chart(
                        results['color_risk'],
                        "C - Color Risk"
                    )
                    st.plotly_chart(fig_color, use_container_width=True, key="color_gauge")
                
                with col2:
                    fig_border = create_gauge_chart(
                        results['border_risk'],
                        "B - Border Risk"
                    )
                    st.plotly_chart(fig_border, use_container_width=True, key="border_gauge")
                    
                    fig_texture = create_gauge_chart(
                        results['texture_risk'],
                        "T - Texture Risk"
                    )
                    st.plotly_chart(fig_texture, use_container_width=True, key="texture_gauge")
                
                # Radar Chart
                st.markdown("### 🎯 Comprehensive Risk Radar")
                fig_radar = create_risk_radar_chart(
                    results['asymmetry_risk'],
                    results['border_risk'],
                    results['color_risk'],
                    results['texture_risk']
                )
                st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")
                
                # Detailed metrics
                st.markdown("### 📋 Detailed Feature Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>🔄 Asymmetry Analysis</h4>
                        <p><strong>Score:</strong> {results['asymmetry_score']:.4f}</p>
                        <p><strong>Risk Level:</strong> {results['asymmetry_risk']:.3f}</p>
                        <p><i>Measures left-right and top-bottom symmetry</i></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>🎨 Color Variation</h4>
                        <p><strong>Distinct Colors:</strong> {results['color_variation']}</p>
                        <p><strong>Risk Level:</strong> {results['color_risk']:.3f}</p>
                        <p><i>Number of distinct color regions detected</i></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>📐 Border Irregularity</h4>
                        <p><strong>Score:</strong> {results['border_irregularity']:.4f}</p>
                        <p><strong>Risk Level:</strong> {results['border_risk']:.3f}</p>
                        <p><i>Measures edge smoothness and regularity</i></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>🧩 Texture Properties</h4>
                        <p><strong>Contrast:</strong> {results['texture_contrast']:.3f}</p>
                        <p><strong>Homogeneity:</strong> {results['texture_homogeneity']:.3f}</p>
                        <p><strong>Energy:</strong> {results['texture_energy']:.3f}</p>
                        <p><strong>Correlation:</strong> {results['texture_correlation']:.3f}</p>
                        <p><i>GLCM-based texture analysis</i></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Image Processing Results
                st.markdown("### 🖼️ Image Processing Pipeline")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Original Image**")
                    st.image(st.session_state.original_image, width='stretch')
                
                with col2:
                    st.markdown("**Hair Removed**")
                    st.image(results['hair_free_image'], width='stretch')
                
                with col3:
                    st.markdown("**Segmentation Mask**")
                    # Create colored mask overlay
                    mask_colored = np.zeros_like(st.session_state.original_image)
                    mask_colored[results['binary_mask'] > 0] = [255, 100, 100]
                    overlay = cv2.addWeighted(st.session_state.original_image, 0.6, mask_colored, 0.4, 0)
                    st.image(overlay, width='stretch')
                
                # Additional metrics
                st.markdown("### 📏 Additional Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Lesion Area", f"{results['lesion_area_pixels']:,} pixels")
                
                with col2:
                    st.metric("Segmentation Quality", f"{results['segmentation_quality']:.2f}")
                
                with col3:
                    lesion_percent = (results['lesion_area_pixels'] / (st.session_state.original_image.shape[0] * st.session_state.original_image.shape[1]) * 100)
                    st.metric("Image Coverage", f"{lesion_percent:.1f}%")
                
            elif results['is_melanoma'] and not results.get('analysis_performed', False):
                st.error(f"⚠️ Melanoma detected but detailed analysis failed: {results.get('analysis_error', 'Unknown error')}")
            
            else:
                st.success("✅ No melanoma detected. The lesion appears benign based on AI classification.")
                st.info("💡 While detailed ABCT analysis is not performed for non-melanoma cases, we recommend regular skin checks and consultation with a dermatologist for any concerning lesions.")
            
            # Download report
            st.markdown("---")
            st.markdown("### 📥 Download Analysis Report")
            
            # Create downloadable report
            report_data = {
                'Image Name': uploaded_file.name,
                'Predicted Class': results['predicted_class'],
                'Classification Confidence': f"{results['classification_confidence']*100:.2f}%",
                'Melanoma Probability': f"{results['melanoma_probability']*100:.2f}%",
            }
            
            if results.get('analysis_performed', False):
                report_data.update({
                    'Risk Level': results['risk_level'],
                    'Overall Risk Score': f"{results['overall_risk_score']:.4f}",
                    'Asymmetry Score': f"{results['asymmetry_score']:.4f}",
                    'Border Irregularity': f"{results['border_irregularity']:.4f}",
                    'Color Variation': results['color_variation'],
                    'Texture Contrast': f"{results['texture_contrast']:.4f}",
                })
            
            report_df = pd.DataFrame([report_data])
            csv = report_df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download CSV Report",
                data=csv,
                file_name=f"melanoma_analysis_{uploaded_file.name}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
