import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import requests
import os

# 🌿 Set up Streamlit UI
st.title("🌾 Smart Crop Disease Detection System")
st.write("Upload a crop leaf image to detect the disease and get weather-based natural remedy suggestions 🌦️")

# ✅ Google Drive model link (replace FILE_ID)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1B1PJWr56pg9JORiYnbX0CcNFOflChSZS"
MODEL_PATH = "model_after_group6.keras"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading AI model... please wait ⏳")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    st.success("✅ Model downloaded successfully!")

model = tf.keras.models.load_model(MODEL_PATH)
st.success("🤖 Model loaded successfully!")


# ✅ File uploader
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

# ✅ City for weather
city = st.text_input("Enter your city for weather-based advice 🌤️", "Pune")

# 🌿 Class names (You can expand this list)
class_names = [
    "Apple___Black_rot", "Apple___Scab", "Apple___healthy",
    "Corn___Gray_leaf_spot", "Corn___Common_rust", "Corn___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Leaf_Mold", "Tomato___Late_blight",
    "Tomato___healthy"
]

# ✅ Predict button
if st.button("🔍 Predict Disease"):
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224,224))
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
        pred = model.predict(img_array)
        predicted_class = np.argmax(pred)
        disease = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"
        st.success(f"🌿 Predicted Disease: **{disease}**")

        # 🌦️ Fetch weather info
        api_key = "YOUR_OPENWEATHER_API_KEY"
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            temp = data['main']['temp']
            hum = data['main']['humidity']
            condition = data['weather'][0]['description']
            st.info(f"🌡️ Temperature: {temp}°C | 💧 Humidity: {hum}% | ☁️ Condition: {condition}")
        else:
            st.warning("⚠️ Unable to fetch weather info. Check your API key or city name.")

        # 🌱 Natural remedy suggestions (simple)
        if "Late_blight" in disease:
            st.write("🪴 Suggestion: Spray neem oil twice a week and remove infected leaves.")
        elif "Leaf_Mold" in disease:
            st.write("🪴 Suggestion: Improve ventilation and apply baking soda + water mix.")
        else:
            st.write("✅ Your crop looks healthy or mild issue detected.")
    else:
        st.warning("⚠️ Please upload an image first!")
