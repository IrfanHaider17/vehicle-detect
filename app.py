import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO("model.pt")  # Ensure this matches your .pt filename

# App title
st.title("🚗 Vehicle Detector")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
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