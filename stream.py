import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(
    page_title="Steel Surface Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# ------------------------
# Model Path
# ------------------------
MODEL_PATH = r"runs\Defect_Detection\weights\best.pt"

# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found!\n\nExpected location:\n{MODEL_PATH}")
        st.stop()

    return YOLO(MODEL_PATH)

model = load_model()

# ------------------------
# Title
# ------------------------
st.title("🔍 Steel Surface Defect Detection")

st.write(
    "Detect six types of steel surface defects using a custom-trained YOLOv8 model."
)

# ------------------------
# Sidebar
# ------------------------
st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Confidence",
    0.10,
    1.00,
    0.30,
    0.05
)

# ------------------------
# Upload Image
# ------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    results = model.predict(
        image_np,
        conf=confidence
    )

    annotated = results[0].plot()

    with col2:
        st.subheader("Detection")
        st.image(annotated, use_container_width=True)

    st.subheader("Detected Defects")

    boxes = results[0].boxes

    if len(boxes) == 0:
        st.warning("No defects detected.")
    else:

        names = model.names

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            st.success(
                f"{names[cls]}   |   Confidence: {conf:.2f}"
            )

    annotated_rgb = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    success, buffer = cv2.imencode(".png", annotated_rgb)

    if success:
        st.download_button(
            "Download Result",
            buffer.tobytes(),
            file_name="prediction.png",
            mime="image/png"
        )