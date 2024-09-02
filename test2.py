import streamlit as st
import requests
import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os

# Set up API key using Streamlit secrets
gemini_api_key = st.secrets["GEMINI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Function to download PDF from GitHub
def download_pdf_from_github(github_pdf_url):
    response = requests.get(github_pdf_url)
    return io.BytesIO(response.content)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to create vector store
def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(chunks, embeddings)
    return vector_store

# Set up Streamlit app
st.title("PDF Question Answering System (Gemini)")

# GitHub PDF URL input
github_pdf_url = "https://github.com/chapmangj/Test_ge0/blob/main/pdfs/Guideline.pdf"

if github_pdf_url:
    # Download and process PDF
    pdf_file = download_pdf_from_github(github_pdf_url)
    pdf_text = extract_text_from_pdf(pdf_file)
    vector_store = create_vector_store(pdf_text)
    
    # Set up Gemini language model
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro")
    
    # Create retrieval QA chain
    gemini_qa = RetrievalQA.from_chain_type(llm=gemini_llm, chain_type="stuff", retriever=vector_store.as_retriever())
    
    # User question input
    user_question = st.text_input("Ask a question about the PDF content:")
    
    if user_question:
        # Get answer from Gemini model
        gemini_answer = gemini_qa.run(user_question)
        
        # Display answer
        st.subheader("Gemini Answer:")
        st.write(gemini_answer)

else:
    st.write("Please enter a GitHub URL for your PDF file to get started.")
