import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load model
try:
    model = YOLO("model.pt")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# App title
st.title("🚗 Vehicle Detector")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Run detection
        results = model.predict(img)
        
        # Display results
        res_plotted = results[0].plot()[:, :, ::-1]  # Convert BGR to RGB
        st.image(res_plotted, caption="Detection Results", use_column_width=True)
        
        # Show detection details
        st.write("### Detection Summary")
        for box in results[0].boxes:
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)
            st.write(f"- {class_name} (Confidence: {confidence:.2f})")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")