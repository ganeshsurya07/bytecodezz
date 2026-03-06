import streamlit as st
from transformers import pipeline
from PIL import Image
import json

# -----------------------------
# Load Hugging Face Model
# -----------------------------

@st.cache_resource
def load_model():
    classifier = pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )
    return classifier

classifier = load_model()

# -----------------------------
# Load Disease Info
# -----------------------------

with open("disease_info.json", "r", encoding="utf-8") as f:
    disease_data = json.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌱 AgroDetect AI – Plant Disease Detection")

st.write("Upload a plant leaf image to detect disease.")

language = st.selectbox(
    "Select Language",
    ["English", "Hindi", "Telugu"]
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Disease"):

        result = classifier(image)

        label = result[0]["label"]
        confidence = round(result[0]["score"] * 100, 2)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence}%")

        # If disease exists in JSON show info
        if label in disease_data:

            description = disease_data[label][language]["description"]
            treatment = disease_data[label][language]["treatment"]

            st.subheader("Disease Description")
            st.write(description)

            st.subheader("Recommended Treatment")
            st.write(treatment)

        else:
            st.warning("Disease information not available.")