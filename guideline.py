import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os
import google.generativeai as genai
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import GooglePalm
from langchain.embeddings import SentenceTransformersEmbeddings



# Set the API keys
gemini_api_key = st.secrets["GEMINI_API_KEY"]
os.environ["GOOGLE_PALM_API_KEY"] = st.secrets["GOOGLE_PALM_API_KEY"]

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
    system_instruction="You're a world renowned geologist, an expert in NSW geology, with a great sense of humour. I will give you the name of a mineral deposit or prospect in NSW, Australia and you will provide me with an overview of the geology and mineralisation including the key mineralisation style and ore minerals. I need you to be very accurate and if you are confused between two deposits with similar names ask for confirmation. If the deposit/prospect has a JORC mineral resource please provide the tonnes and grade of the last known estimate and include the year. If it is a producing mine, include the latest known grade and tonnes. And then finish with a fun fact. ",
)

chat_session = model.start_chat(history=[])

# PDF processing function
@st.cache_resource
def process_pdfs():
    pdf_folder = "pdfs"
    pdf_files = ["Guideline.pdf"]
    
    text = ""
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = SentenceTransformersEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base

# Streamlit interface
st.title("NSW Geo Bot")

# Display image
image_url = "https://meg.resourcesregulator.nsw.gov.au/sites/default/files/styles/wysiwyg_image/public/2022-11/Surface-geology.jpg?itok=TZAIypko.jpg"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
st.image(img, caption="NSW")

# Process PDFs (this will only run once and cache the result)
knowledge_base = process_pdfs()

# Add tabs for different functionalities
tab1, tab2 = st.tabs(["Mineral Deposit Info", "PDF Q&A"])

with tab1:
    mineral_deposit = st.text_input("Enter the name of a mineral deposit in NSW you're interested in:")
    if st.button("Get Mineral Deposit Information"):
        if mineral_deposit:
            response = chat_session.send_message(mineral_deposit)
            st.write(response.text)
        else:
            st.warning("Please enter a mineral deposit name.")

with tab2:
    st.write("Ask questions about the pre-loaded PDF documents")
    user_question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = GooglePalm()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            
            st.write("Answer:", response)
        else:
            st.warning("Please enter a question.")
