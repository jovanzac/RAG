import streamlit as st
# from scripts.recorder import AudioManager

# audio_manager = AudioManager()

st.title('My first app')
st.write('Hello, world!')

button1 = st.button('Start recording!')
button2 = st.button('Stop recording!')
if button1:
    # audio_manager.record_audio()
    st.write('Button1 clicked!')
if not button2 :
    st.write('Button2 clicked')