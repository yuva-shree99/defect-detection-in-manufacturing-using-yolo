import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="YOLOv8 Defect Detection",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 YOLOv8 Defect Detection System")
st.write("Upload an image to detect defects.")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return YOLO(
        "best.pt"
    )




model = load_model()

# ---------------- Upload Image ----------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # Original Image
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        temp_path = temp.name

    # Prediction
    with st.spinner("Running Detection..."):
        results = model.predict(
            source=temp_path,
            conf=0.25,
            save=False
        )

    result = results[0]

    # Annotated Image
    annotated_image = result.plot()

    with col2:
        st.subheader("Detection Result")
        st.image(annotated_image, use_container_width=True)

    st.markdown("---")

    if len(result.boxes) == 0:
        st.warning("No defects detected.")
    else:
        st.success(f"Detected {len(result.boxes)} defect(s).")

        st.subheader("Detection Details")

        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            st.write(
                f"**{i+1}. {model.names[cls]}** — Confidence: **{confidence:.2%}**"
            )