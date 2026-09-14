from pathlib import Path

import streamlit as st

from inference import Translator

ARTIFACT_DIR = Path(__file__).parent / "deployment_artifacts"

st.set_page_config(page_title="English -> Amharic Translator", page_icon="🌍")
st.title("English → Amharic Neural Machine Translation")
st.caption("Attention-based Seq2Seq + LSTM")

@st.cache_resource
def load_translator():
    return Translator(artifact_dir=ARTIFACT_DIR)

translator = load_translator()

text = st.text_area("Enter an English sentence:", "I am going to the university.")

if st.button("Translate", type="primary") and text.strip():
    with st.spinner("Translating..."):
        translation = translator.translate(text)
    st.subheader("Amharic translation")
    st.write(translation if translation else "(model produced an empty output — try a shorter/simpler sentence)")

with st.expander("About"):
    st.write(
        "This app uses an Attention-based Seq2Seq LSTM model trained from scratch on a "
        "parallel English-Amharic corpus. See the accompanying notebook for data preprocessing, "
        "training, evaluation (BLEU/chrF), and error analysis."
    )
