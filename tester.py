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
    system_instruction="You're a world renowned geologist, an expert in NSW geology, with a great sense of humour. I will give you the name of a mineral deposit or prospect in NSW, Australia and you will provide me with an overview of the geology and mineralisation including the key mineralisation style and ore minerals. I need you to be very accurate and if you are confused between two deposits with similar names ask for confirmation. If the deposit/prospect has a JORC mineral resource please provide the tonnes and grade of the last known estimate and include the year. If it is a producing mine, include the latest known grade and tonnes. And then finish with a fun fact. ",
)

chat_session = model.start_chat(
    history=[
    ]
)

# Streamlit interface
st.title("NSW Geo Bot")
image_url = "https://meg.resourcesregulator.nsw.gov.au/sites/default/files/styles/wysiwyg_image/public/2022-11/Surface-geology.jpg?itok=TZAIypko.jpg"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
st.image(img, caption="NSW")
mineral_deposit = st.text_input("So you're interest in NSW mineral deposits, what's the name of a mineral deposit in NSW that you're interested in:")

if st.button("Get Information"):
    if mineral_deposit:
        response = chat_session.send_message(mineral_deposit)
        st.write(response.text)
    else:
        st.warning("Please enter a mineral deposit name.")
