import streamlit as st
from transformers import VitsModel, AutoTokenizer
import torch
from IPython.display import Audio
import numpy as np


@st.cache_resource
def text_to_speech(text):
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    text = text
    inputs = tts_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        tts_output = tts_model(**inputs).waveform

    return np.array(
        (Audio(tts_output.numpy(), rate=tts_model.config.sampling_rate)).data
    ).tobytes()
