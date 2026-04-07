import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

st.title("Medical Image Classifier")

# ✅ Google Drive model download
FILE_ID = "1ZJZrG3NGE5sWN34vR2kdSdDYJAuIz8R"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ✅ Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    st.write("Prediction:", prediction)
    st.write("Predicted Class:", predicted_class)