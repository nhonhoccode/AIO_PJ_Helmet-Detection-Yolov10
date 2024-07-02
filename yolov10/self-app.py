from ultralytics import YOLOv10
import numpy as np
from PIL import Image
import streamlit as st


def detect_image(image):
    MODEL_PATH = "yolov10n.pt"
    model = YOLOv10(MODEL_PATH)
    results = model(image)
    processed_image = results[0].plot()
    return processed_image


def main():
    st.title("Object Detection By YOLOv10")
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file is not None:
        # st.image(file, caption="Uploaded Image")

        image = Image.open(file)
        image = np.array(image)
        processed_image = detect_image(image)

        col1, col2 = st.columns(2)

        with col1:
            st.header("Original Image")
            st.image(image, caption="Uploaded Image")

        with col2:
            st.header("Processed Image")
            st.image(processed_image, caption="Processed Image")


if __name__ == "__main__":
    main()
