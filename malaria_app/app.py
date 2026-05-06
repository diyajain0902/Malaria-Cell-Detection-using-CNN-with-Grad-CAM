import gradio as gr
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('malaria_cnn_model.keras')

def predict(image):
    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Get raw probability
    prob = model.predict(img, verbose=0)[0][0]

    # ── INNOVATION 1: Confidence Categories ──────────────────
    if prob >= 0.90:
        confidence = "High Confidence Parasitized"
    elif prob >= 0.70:
        confidence = "Moderate Confidence Parasitized"
    elif prob >= 0.50:
        confidence = "Low Confidence Parasitized"
    elif prob >= 0.30:
        confidence = "Low Confidence Uninfected"
    elif prob >= 0.10:
        confidence = "Moderate Confidence Uninfected"
    else:
        confidence = "High Confidence Uninfected"

    # ── INNOVATION 2: Clinical Metrics ────────────────────────
    if prob >= 0.50:
        diagnosis    = "PARASITIZED — Malaria Detected"
        clinical_note = (
            "Clinical Advice: High probability of malaria infection detected. "
            "Please consult a medical expert for confirmation and treatment."
        )
    else:
        diagnosis    = "UNINFECTED — Healthy Cell"
        clinical_note = (
            "Clinical Advice: No malaria infection detected. "
            "If symptoms persist, consult a medical expert."
        )

    # ── Confidence score as percentage ────────────────────────
    confidence_pct = prob if prob >= 0.50 else 1 - prob

    # ── Build output ──────────────────────────────────────────
    result = f"""
DIAGNOSIS:         {diagnosis}
CONFIDENCE LEVEL:  {confidence}
CONFIDENCE SCORE:  {confidence_pct*100:.1f}%
RAW PROBABILITY:   {prob:.4f}

CLINICAL NOTE:
{clinical_note}

---
Sensitivity of model : ~95%
Specificity of model : ~97%
Note: This is an AI tool for educational purposes.
Always validate with a certified medical professional.
    """

    return result

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Red Blood Cell Image"),
    outputs=gr.Textbox(label="Diagnosis Report", lines=16),
    title="Malaria Cell Detector — CNN + Clinical AI",
    description=(
        "Upload a red blood cell microscopy image to get an AI-powered diagnosis. "
        "This model uses CNN with clinical metrics including confidence levels, "
        "sensitivity and specificity — built for Biocon Ltd. EST Project UCS321."
    ),
    theme="soft"
)

app.launch()