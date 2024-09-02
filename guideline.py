import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import os

# Set up environment variables
os.environ["GOOGLE_PALM_API_KEY"] = st.secrets["GOOGLE_PALM_API_KEY"]

# Load PDF
@st.cache_resource
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

# Create vector store
@st.cache_resource
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore

# Set up retrieval QA chain
def setup_qa_chain(vectorstore):
    llm = GooglePalm()  # This uses the Gemini API
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qa_chain

# Streamlit app
def main():
    st.title("PDF Question Answering App")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF and create vector store
        texts = load_pdf("temp.pdf")
        vectorstore = create_vector_store(texts)

        # Set up QA chain
        qa_chain = setup_qa_chain(vectorstore)

        # Question input
        question = st.text_input("Ask a question about the PDF:")

        if question:
            # Get answer
            answer = qa_chain.run(question)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()

