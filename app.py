import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
from PIL import Image
import io
import time
import os

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="Fracture Detection AI", layout="wide")

st.markdown(
    """
    <style>
    /* Medical X-Ray Background with Light Overlay for perfect readability */
    .stApp {
        background: linear-gradient(rgba(240, 244, 248, 0.85), rgba(240, 244, 248, 0.95)),
                    url("https://www.shutterstock.com/image-photo/xray-image-ankle-fracture-blue-260nw-2253312433.jpg") no-repeat center center fixed;
        background-size: cover;
    }

    /* Remove top padding */
    .stMainBlockContainer {
        padding-top: 0 !important;
    }

    /* Deep Blue Header */
    .custom-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 3rem 2rem;
        text-align: center;
        border-bottom-left-radius: 20px;
        border-bottom-right-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }

    .custom-header h1 {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        padding-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .custom-header p {
        color: #e0f2fe;
        font-size: 1.2rem;
        margin: 0;
    }

    /* Target the uploaded X-ray image to match the result card styling */
    [data-testid="stImage"] img {
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255,255,255,0.5);
    }

    /* Results Card */
    .result-card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 2.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255,255,255,0.5);
        text-align: center;
        color: #1e293b;
        height: 100%;
        backdrop-filter: blur(10px);
    }

    .result-card h3 {
        color: #64748b;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1.5rem;
    }

    .highlight-fracture {
        color: #dc2626;
        font-size: 2rem;
        font-weight: 900;
        margin: 10px 0;
    }

    .highlight-normal {
        color: #16a34a;
        font-size: 2rem;
        font-weight: 900;
        margin: 10px 0;
    }

    .confidence-box {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 2rem 0;
        border: 1px solid #e2e8f0;
    }

    .confidence-box p {
        margin: 0;
        color: #334155;
    }

    .confidence-value {
        font-size: 1.8rem !important;
        font-weight: bold;
        color: #0f172a !important;
    }

    /* Section Titles */
    .section-title {
        text-align: center;
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .section-subtitle {
        text-align: center;
        color: #475569;
        margin-bottom: 2rem;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        color: #475569;
        padding: 2rem;
        margin-top: 4rem;
        border-top: 1px solid rgba(0,0,0,0.1);
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- HEADER ---
st.markdown("<div class='custom-header'><h1>Fracture Detection AI</h1><p>Advanced Deep Learning Analysis for Radiographic Imaging</p></div>", unsafe_allow_html=True)

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_keras_model():
    model_path = "model/recognition_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please check the path.")
        st.stop()
    
    # Explicitly tell TensorFlow where to find the missing preprocessing function
    custom_objects_dict = {
        'preprocess_input': tf.keras.applications.vgg16.preprocess_input
    }
    
    try:
        # Load the model and inject the missing function
        return load_model(
            model_path, 
            custom_objects=custom_objects_dict, 
            compile=False, 
            safe_mode=False
        )
    except TypeError:
        # Fallback for slightly older versions of TensorFlow
        return load_model(
            model_path, 
            custom_objects=custom_objects_dict, 
            compile=False
        )

model = load_keras_model()
data_label = ['fractured', 'not fractured']
image_height = 256
image_width = 256

# --- MAIN APP LOGIC ---
st.markdown("<div class='section-title'>Initialize Scan Analysis</div>", unsafe_allow_html=True)

up_col1, up_col2, up_col3 = st.columns([1, 2, 1])
with up_col2:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        st.markdown("<br>", unsafe_allow_html=True)

        # FIX 1: Tighter grid layout [1, 3, 3, 1] to push the image and results box closer together
        col1, col2, col3, col4 = st.columns([1, 3, 3, 1])

        with col2:
            with st.spinner("Processing neural network layers..."):
                time.sleep(1)

            image = Image.open(io.BytesIO(uploaded_file.read()))
            if image.mode != "RGB":
                image = image.convert("RGB")

            img_arr = np.array(image.resize((image_width, image_height)))
            img_arr = img_arr.reshape((1, image_height, image_width, 3))

            predict = model.predict(img_arr)
            predicted_class = np.argmax(predict)
            confidence = np.max(predict) * 100

            st.image(image, use_container_width=True) # Removed the unaligned caption

        with col3:
            result_class = "highlight-fracture" if predicted_class == 0 else "highlight-normal"
            display_text = "FRACTURE DETECTED" if predicted_class == 0 else "NO FRACTURE DETECTED"

            st.markdown(f"<div class='result-card'><h3>Analysis Complete</h3><p style='color: #475569; font-size: 1.1rem; margin-bottom: 5px;'>Diagnostic Status</p><p class='{result_class}'>{display_text}</p><div class='confidence-box'><p>AI Confidence Level</p><p class='confidence-value'>{confidence:.1f}%</p></div><p style='font-size: 0.85rem; color: #64748b; font-style: italic; margin-top: 1rem;'>This result is generated by an automated deep learning model.<br>Always consult a qualified radiologist for official medical diagnoses.</p></div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")

# --- EXAMPLE IMAGES ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Reference Database</div>", unsafe_allow_html=True)
st.markdown("<div class='section-subtitle'>Sample classifications handled by the model</div>", unsafe_allow_html=True)

# FIX 2: Custom HTML Flexbox to perfectly align and crop the reference images to the exact same size
st.markdown('''
<div style="display: flex; gap: 30px; justify-content: center; max-width: 900px; margin: 0 auto;">
    <div style="flex: 1; text-align: center;">
        <img src="https://www.shutterstock.com/image-photo/blue-tone-radiograph-on-dark-600nw-2267523647.jpg" style="width: 100%; height: 280px; object-fit: cover; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
        <p style="color: #475569; font-weight: bold; margin-top: 15px; font-size: 1.1rem;">Positive: Fracture Present</p>
    </div>
    <div style="flex: 1; text-align: center;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpqwF54Bgvy8KwtTxT7W3mIEjH6MxqqMrYhg&s" style="width: 100%; height: 280px; object-fit: cover; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
        <p style="color: #475569; font-weight: bold; margin-top: 15px; font-size: 1.1rem;">Negative: Normal Bone Structure</p>
    </div>
</div>
''', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<div class='custom-footer'>Trained on a specialized dataset of X-ray images for automated fracture detection.</div>", unsafe_allow_html=True)