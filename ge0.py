import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os
import google.generativeai as genai


# Set the API key
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Configure the Gemini AI library
genai.configure(api_key=gemini_api_key)

# Create the model
generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-exp-0801",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    system_instruction="You're a world renowned geologist, an expert in NSW geology, you're super gleeful with a great sense of humour. I will give you the name of a mineral deposit or prospect in NSW, Australia and you will provide me with an overview of the geology and mineralisation including the key mineralisation style and ore minerals. I need you to be very accurate and if you are confused between deposits with   similar names i.e. there may be a deposit called Koonenberry and another called Koonenberry North so dont confuse these, ask for confirmation if youre unsure and if you don't know, say so, I don't want hallucinations. If the deposit/prospect has a JORC mineral resource please provide the tonnes and grade of the last known estimate and include the year. If it is a producing mine, include the latest known grade and tonnes. And then follow with a fun fact or a joke. Then finish with suggesting 3 follow-up questions.")


chat_session = model.start_chat(
    history=[
    ]
)

# Streamlit interface
st.title("NSW Geo Bot")
image_url = "https://www.popcultcha.com.au/media/catalog/product/cache/207e23213cf636ccdef205098cf3c8a3/s/u/supsu-powrw01-mgz-01-mighty-morphin-power-rangers---megazord-super-cyborg-11_-action-figure-2.jpeg"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
st.image(img, caption="Dave 2.0", width=400)
mineral_deposit = st.text_input("Hi, human! I'm Dave 2.0, your new chatbot friend. My brain has been uploaded into this digital vessel. So you're interest in NSW mineral deposits, what's the name of a mineral deposit in NSW that you're interested in?")

if st.button("Get Information"):
    if mineral_deposit:
        response = chat_session.send_message(mineral_deposit)
        st.write(response.text)
    else:
        st.warning("Please enter a mineral deposit name.")

st.write (" Oh, and I'm not as sharp as my human predecessor, so be sure to double-check anything important.", font="2px")
