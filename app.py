import streamlit as st
from llm import load_normal_chain
from streamlit_mic_recorder import mic_recorder
from handling_audio import transcribe_audio
from audio_recorder_streamlit import audio_recorder
import io
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModel

def load_chain():
    return load_normal_chain()

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")

llm_chain=load_chain()

# streamlit-code
st.title("Bruno - Your Personal Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container(border=True,height=360)

for msg in st.session_state.messages:
    with chat_container:
        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

audio_bytes = audio_recorder()

transcibed_audio=""
if audio_bytes:
    transcibed_audio=transcribe_audio(audio_bytes)

user_input=st.chat_input("Enter message..")

speech_values=[]
if user_input or transcibed_audio:
    
    with chat_container:
        with st.chat_message("user"):
            if user_input:
                st.markdown(user_input)
                st.session_state.messages.append({"role":"user","content":user_input})
            else:
                st.markdown(transcibed_audio)
                st.session_state.messages.append({"role":"user","content":transcibed_audio})
    if user_input:
        llm_response=llm_chain.run(user_input=user_input)
    else:
        llm_response=llm_chain.run(user_input=transcibed_audio)
    
    inputs = processor(text=[llm_response["text"]],return_tensors="pt")
    speech_values = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=True,
    )

    speech_values_np = np.array(speech_values[0])
    sample_rate = 22050
    buffer = io.BytesIO()
    sf.write(buffer, speech_values_np, sample_rate, format='WAV')
    text_speech = buffer.getvalue()

    with chat_container:
        with st.chat_message("ai"):
            st.markdown(llm_response["text"])
            st.audio(text_speech)
    st.session_state.messages.append({"role":"ai","content":llm_response["text"]})