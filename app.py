import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("depression_detection_model.h5")
    return model

model = load_model()

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Title
st.title("EEG Depression Detection")
st.write("Upload an EEG image to detect if depression is **Positive** or **Negative**.")

# File uploader
uploaded_file = st.file_uploader("Choose an EEG Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded EEG Image", use_column_width=True)

    # Preprocess image
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(image_array)[0][0]
    label = "Positive" if prediction >= 0.5 else "Negative"

    st.markdown(f"### 🔍 Prediction: **{label}**")
