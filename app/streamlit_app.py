"""Streamlit demo UI for the medical image classifier.

Run locally with:
    streamlit run app/streamlit_app.py

The demo loads the model directly (in-process) so it works without the API
running. It lets a user upload an image, view the predicted class with a
probability bar chart, and optionally render a Grad-CAM explanation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from medical_classifier.config import load_config  # noqa: E402
from medical_classifier.constants import DISCLAIMER  # noqa: E402
from medical_classifier.inference.predictor import Predictor  # noqa: E402


@st.cache_resource
def get_predictor() -> Predictor | None:
    cfg = load_config()
    try:
        return Predictor(cfg.model_file)
    except FileNotFoundError:
        return None


def main() -> None:
    st.set_page_config(page_title="Medical Image Disease Classifier", page_icon="🧠")
    st.title("Medical Image Disease Classifier")
    st.caption(
        "Production-style demo inspired by an Alzheimer's PET classification workflow."
    )
    st.warning(DISCLAIMER)

    predictor = get_predictor()
    if predictor is None:
        st.error(
            "No trained model found. Run `make train` (or "
            "`python -m medical_classifier.training.train`) to create a checkpoint."
        )
        return

    uploaded = st.file_uploader(
        "Upload a medical image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )
    show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)

    if uploaded is None:
        st.info("Upload an image to get a prediction. Try the files in `samples/`.")
        return

    image = Image.open(uploaded)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input", use_container_width=True)

    result = predictor.predict_pil(image)
    with col2:
        st.metric("Predicted class", result.predicted_class)
        st.metric("Confidence", f"{result.confidence:.1%}")

    st.subheader("Class probabilities")
    st.bar_chart(result.class_probabilities)

    if show_gradcam:
        from medical_classifier.inference.gradcam import GradCAM

        st.subheader("Grad-CAM explanation")
        st.caption("Highlighted regions most influenced the prediction.")
        with GradCAM(predictor) as cam:
            png = cam.overlay_png(image)
        st.image(png, caption="Grad-CAM overlay", use_container_width=True)


if __name__ == "__main__":
    main()
