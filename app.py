import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="centered"
)

class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight"
]

display_names = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___healthy": "Healthy",
    "Tomato___Late_blight": "Late Blight"
}

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


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "crop_disease_mobilenetv2.keras"
    )


model = load_model()

st.title("🌱 Crop Disease Detection")
st.write("Upload a tomato leaf image to detect its disease.")

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    # Resize image to the same size used during training
    image_resized = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image_resized)

    # IMPORTANT:
    # Match the preprocessing used during model training
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = (
        predictions[0][predicted_index] * 100
    )

    st.subheader("Prediction")

    st.success(
        f"Detected: {display_names[predicted_class]}"
    )

    st.write(
        f"Confidence: **{confidence:.2f}%**"
    )

    if confidence >= 80:
        confidence_level = "High Confidence"

    elif confidence >= 60:
        confidence_level = "Moderate Confidence"

    else:
        confidence_level = "Low Confidence"

    st.info(
        f"Confidence Level: **{confidence_level}**"
    )

    st.subheader("About the Prediction")

    st.info(
        disease_descriptions[predicted_class]
    )

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

    st.markdown("---")

    st.caption(
        "⚠️ This AI model is intended for educational and "
        "demonstration purposes. It should not replace "
        "professional agricultural advice."
    )