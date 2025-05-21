import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

# Load model from folder
model = tf.keras.models.load_model('weather_classifier')

# Class names (ensure this order matches training)
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

st.title("🌤️ Weather Image Classifier")
st.markdown("Upload an image, and the model will predict the weather condition.")

# Upload
uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")
