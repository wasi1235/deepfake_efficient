import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the pre-trained model
@st.cache_resource
def load_keras_model():
    model = load_model("models/efficientnetB2.keras")  # Update the path to your model
    return model

model = load_keras_model()

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((32, 32))  # Resize to the model's expected input size (32x32)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image to the range [0, 1]
    return img_array

# Streamlit app UI
st.title("Deepfake Image Detection")
st.write("Upload an image to check if it's fake or real.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display the uploaded image
        img = Image.open(uploaded_file)

        # Convert image to RGB if it has an alpha channel (RGBA)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image and make a prediction
        img_array = preprocess_image(img)
        with st.spinner('Processing the image...'):
            prediction = model.predict(img_array)

        # Assuming the model returns a value between 0 and 1 (0 = real, 1 = fake)
        if prediction >= 0.5:
            st.write("### This image is **fake**.")
        else:
            st.write("### This image is **real**.")
    
    except Exception as e:
        st.error(f"Error processing the image: {e}")
