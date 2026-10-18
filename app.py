import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="CropDoctor", page_icon="🌿")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cropdoctor_finetuned.keras")
    class_names = json.load(open("class_names.json"))
    return model, class_names

model, class_names = load_model()

st.title("🌿 CropDoctor")
st.write("Upload a clear photo of a single leaf (tomato, potato or bell pepper) to detect the disease.")

file = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Your photo", use_container_width=True)

    x = np.array(img.resize((224, 224)), dtype="float32")
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x)[0]
    top3 = np.argsort(probs)[::-1][:3]

    best = class_names[top3[0]].replace("_", " ").strip()
    confidence = probs[top3[0]] * 100

    st.subheader(f"Result: {best}")
    st.write(f"Confidence: {confidence:.1f}%")

    if confidence < 60:
        st.warning("The model is not sure. Try a clearer, closer photo of one leaf.")

    st.write("Other possibilities:")
    for i in top3[1:]:
        st.write(f"- {class_names[i].replace('_', ' ').strip()} ({probs[i]*100:.1f}%)")