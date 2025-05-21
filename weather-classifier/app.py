import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load class names
class_names = ['Cloudy', 'Rainy', 'Shine', 'Sunrise']

# Load model
model = tf.keras.models.load_model('weather_classifier.keras')

# Preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title(" Weather Image Classifier")
st.write("Upload an image of weather and get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Classifying...")
    input_image = preprocess_image(image)
    predictions = model.predict(input_image)
    predicted_class = class_names[np.argmax(predictions)]

    st.success(f" redicted Weather: `{predicted_class}`")
