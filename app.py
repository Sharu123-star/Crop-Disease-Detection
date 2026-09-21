import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# Class names
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight"
]

# Display-friendly names
display_names = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___healthy": "Healthy",
    "Tomato___Late_blight": "Late Blight"
}

# Disease descriptions
disease_descriptions = {
    "Tomato___Bacterial_spot":
        "Bacterial Spot can appear as small dark spots on tomato leaves and may cause yellowing around affected areas.",

    "Tomato___Early_blight":
        "Early Blight commonly produces dark spots with concentric ring patterns on older leaves.",

    "Tomato___healthy":
        "The leaf is classified as healthy by the model, with no disease class detected.",

    "Tomato___Late_blight":
        "Late Blight can cause dark, irregular lesions on tomato leaves and may spread rapidly under favorable conditions."
}

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crop_disease_mobilenetv2.keras")


model = load_model()

# Title
st.title("🌱 Crop Disease Detection")

st.write(
    "Upload a tomato leaf image to detect its disease."
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    # Preprocess image
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Find predicted class
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]

    # Calculate confidence
    confidence = predictions[0][predicted_index] * 100

    # -----------------------------
    # Prediction result
    # -----------------------------

    st.subheader("Prediction")

    st.success(
        f"Detected: {display_names[predicted_class]}"
    )

    st.write(
        f"Confidence: **{confidence:.2f}%**"
    )

    # -----------------------------
    # Confidence level
    # -----------------------------

    if confidence >= 80:
        confidence_level = "High Confidence"
    elif confidence >= 60:
        confidence_level = "Moderate Confidence"
    else:
        confidence_level = "Low Confidence"

    st.info(
        f"Confidence Level: **{confidence_level}**"
    )

    # -----------------------------
    # Disease information
    # -----------------------------

    st.subheader("About the Prediction")

    st.info(
        disease_descriptions[predicted_class]
    )

    # -----------------------------
    # Class probabilities
    # -----------------------------

    st.subheader("Class Probabilities")

    for i, class_name in enumerate(class_names):

        probability = predictions[0][i] * 100

        st.write(
            f"{display_names[class_name]}: "
            f"**{probability:.2f}%**"
        )

        st.progress(
            float(predictions[0][i])
        )

    # -----------------------------
    # Disclaimer
    # -----------------------------

    st.markdown("---")

    st.caption(
        "⚠️ This AI model is intended for educational and "
        "demonstration purposes. It should not replace "
        "professional agricultural advice."
    )