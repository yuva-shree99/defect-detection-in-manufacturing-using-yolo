import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="YOLOv8 Defect Detection",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 YOLOv8 Defect Detection System")
st.write("Upload an image or choose a sample image to detect defects.")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- Choose Input ----------------
option = st.radio(
    "Choose an option",
    ["📤 Upload Your Image", "🖼️ Use Sample Image"]
)

image = None

# ---------------- Upload ----------------
if option == "📤 Upload Your Image":

    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# ---------------- Sample Images ----------------
else:

    sample_folder = "sample_images"

    if os.path.exists(sample_folder):

        defect_classes = sorted([
            folder for folder in os.listdir(sample_folder)
            if os.path.isdir(os.path.join(sample_folder, folder))
        ])

        if defect_classes:

            selected_class = st.selectbox(
                "Select Defect Type",
                defect_classes
            )

            class_folder = os.path.join(
                sample_folder,
                selected_class
            )

            sample_images = sorted([
                img for img in os.listdir(class_folder)
                if img.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            if sample_images:

                selected_image = st.selectbox(
                    "Select Sample Image",
                    sample_images
                )

                image_path = os.path.join(
                    class_folder,
                    selected_image
                )

                image = Image.open(image_path)

            else:
                st.warning("No images found in this folder.")

        else:
            st.warning("No defect folders found.")

    else:
        st.error("sample_images folder not found.")

# ---------------- Prediction ----------------
if image is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        temp_path = temp.name

    with st.spinner("Running Detection..."):

        results = model.predict(
            source=temp_path,
            conf=0.05,
            save=False
        )

    result = results[0]

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