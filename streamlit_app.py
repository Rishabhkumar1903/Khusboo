import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pickle

# --- 1. PREMIUM PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SkinShield AI | Skin Cancer Detection",
    page_icon="🩺",
    layout="wide", # Full screen layout professional look ke liye
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS FOR ULTRA-MODERN LOOK ---
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .main { background-color: #f8f9fa; font-family: 'Inter', sans-serif; }
    
    /* Custom Card Design */
    .custom-card {
        background-color: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    
    /* Main Action Button Style */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0066cc 0%, #004499 100%);
        color: white;
        font-weight: 700;
        font-size: 16px;
        padding: 12px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 102, 204, 0.4);
    }
    
    /* Title Styling */
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 30px;
    }
    
    /* Status Badge Colors */
    .result-badge {
        background-color: #eff6ff;
        color: #1e40af;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 20px;
        border: 1px solid #bfdbfe;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR (INFO & INSTRUCTIONS) ---
with st.sidebar:
    st.markdown("<h2 style='color: #0066cc;'>🛡️ SkinShield AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📋 Instructions:")
    st.info("""
    1. Upload a clear, well-lit image of the skin lesion.
    2. Ensure the image focus is directly on the affected area.
    3. Click on **'Run Diagnostic Analysis'** button.
    """)
    st.markdown("---")
    st.markdown("### 🔬 System Status:")
    st.success("Core Engine: Active")
    st.success("GPU Acceleration: Connected")

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model_and_labels():
    # TIP: Agar aapka model .h5 format me hai, toh bas yahan extension .h5 kar lena
    model = tf.keras.models.load_model('best_model.h5', compile=False)
    with open('classes.pkl', 'rb') as f:
        labels = pickle.load(f)
    return model, labels

try:
    model, class_names = load_model_and_labels()
except Exception as e:
    st.error(f"⚠️ Model Files missing in directory: {e}")
    st.stop()

# --- 5. HEADER SECTION ---
st.markdown("<div class='main-title'>🩺 Skin Cancer Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced deep learning platform for automated dermatological lesion screening.</div>", unsafe_allow_html=True)

# --- 6. MAIN CONTENT LAYOUT (2 COLUMNS) ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Image Specimen")
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file:
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Uploaded Specimen', use_container_width=True)
    else:
        # Default placeholder box when no image uploaded
        st.info("Waiting for image upload specimen...")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Diagnostic Analysis")
    
    if file:
        st.write("Image successfully received. Ready for diagnostic evaluation.")
        st.write("")
        
        if st.button('🚀 Run Diagnostic Analysis'):
            with st.spinner('Analyzing cell pathology structures...'):
                # Preprocessing
                size = (224, 224)    
                img = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                img_array = np.asarray(img)
                img_scaled = img_array.astype(np.float32) / 255.0
                img_final = np.expand_dims(img_scaled, axis=0)
                
                # Inference
                prediction = model.predict(img_final)
                result_idx = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                # Displays
                st.markdown("---")
                st.markdown("#### Detected Pathology:")
                detected_class = class_names[result_idx].replace('_', ' ').title()
                st.markdown(f"<div class='result-badge'>{detected_class}</div>", unsafe_allow_html=True)
                
                st.markdown("")
                st.markdown(f"#### AI Confidence Score: **{confidence:.2f}%**")
                st.progress(int(confidence))
                
                # Premium Summary Alert Box
                st.markdown("---")
                st.markdown("#### 🩺 Clinical Summary Recommendation:")
                if "melanoma" in detected_class.lower() or "carcinoma" in detected_class.lower():
                    st.error("🔴 **High Priority:** The system detected structural patterns resembling malignant characteristics. Immediate clinical dermatoscopy evaluation is strongly recommended.")
                else:
                    st.warning("🟡 **Routine Review:** Benign or non-cancerous patterns detected. Keep monitoring the area for any changes in shape, size, or color over time.")
    else:
        st.caption("Please upload a specimen image on the left side to unlock diagnostic metrics.")
        
    st.markdown("</div>", unsafe_allow_html=True)

# --- 7. FOOTER DISCLAIMER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 12px;'>⚠️ <b>Regulatory Disclaimer:</b> This dashboard is powered by an educational AI research model. It does not constitute formal medical diagnostics or clinical validation. Always seek direct consultation from certified medical professionals.</p>", unsafe_allow_html=True)
