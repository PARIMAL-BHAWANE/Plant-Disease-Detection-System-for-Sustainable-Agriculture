import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# 🌙 Enable Dark Mode
st.markdown(
    """
    <style>
        body { background-color: #0e1117; color: white; }
        .stApp { background-color: #0e1117; }
        .stButton > button { background-color: #1E90FF; color: white; border-radius: 10px; }
        .stTextInput, .stFileUploader, .stSelectbox, .stTextArea {
            background-color: #262730;
            color: white;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
def download_model():
    url = "https://drive.google.com/uc?id=1UnvkEgnUKv2arImj6E72dkuevqp-NR86"
    output = "trained_plant_disease_model.keras"

    # If model already exists, no need to download
    if os.path.exists(output) and os.path.getsize(output) > 0:
        st.success("✅ Model file already exists.")
        return

    try:
        st.info("📥 Downloading model... Please wait.")
        gdown.download(url, output, quiet=False)

        # Validate the downloaded file
        if not os.path.exists(output) or os.path.getsize(output) == 0:
            st.error("🚨 Model download failed! Please check the link and permissions.")
            return
        st.success("✅ Model downloaded successfully!")
    
    except Exception as e:
        st.error(f"🚨 Error downloading model: {e}")

    # Preprocess the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)

    return np.argmax(predictions)  # Return the predicted class index


# 🌍 Initialize session state for navigation
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "HOME"

# 🏠 Home and Disease Recognition Buttons
st.markdown("<h1 style='text-align: center;'>🌱 Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("🏠 HOME"):
        st.session_state.app_mode = "HOME"
with col2:
    if st.button("🔍 DISEASE RECOGNITION"):
        st.session_state.app_mode = "DISEASE RECOGNITION"

# 🌿 Load and Display Main Image
img = Image.open("Diseases.png")
st.image(img, use_container_width=True)

# 🏠 Home Page
if st.session_state.app_mode == "HOME":
    st.markdown("<h2 style='text-align: center;'>Welcome to the Plant Disease Detection System!</h2>", unsafe_allow_html=True)

# 🔍 Disease Recognition Page
elif st.session_state.app_mode == "DISEASE RECOGNITION":
    st.header("🔍 Disease Recognition")
    test_image = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)

        if st.button("🚀 Predict"):
            st.snow()
            st.write("🧠 Model Prediction")
            result_index = model_prediction(test_image)

            # 🍏 Class Labels
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            # 🎯 Display Prediction Result
            st.success(f"✅ Model predicts: **{class_name[result_index]}**")

# 🌍 Add extra spacing to push the footer down
st.markdown("<br>", unsafe_allow_html=True)

# 📌 Centered GitHub Profile Section (Now at the Bottom Without Overlap)
st.markdown(
    """
    <div style="text-align: center;">
        <h3>📌 Connect with Me</h3>
        <img src="https://avatars.githubusercontent.com/PARIMAL-BHAWANE" width="100" style="border-radius: 50%;">
        <br><br>
        <a href="https://github.com/PARIMAL-BHAWANE" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-%23181717.svg?style=for-the-badge&logo=github&logoColor=white">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
