import requests
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from io import BytesIO
import openai

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Define the Streamlit app title
st.title("ELT LM GPT")

# Define the question-answering prompt template
template = """
You are an assistant for question-answering tasks in detail. Answer the question based on the context below. If you can't answer the question reply "I don't know". Keep the answer lengthy and informative.

Context: {context}

Question: {question}
"""

def get_pdf_text(pdf_url):
    """Fetch and read the content of a PDF file from a given URL."""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_content = BytesIO(response.content)
        pdf_reader = PdfReader(pdf_content)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch PDF content. Error: {e}")
        return None

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create a vector store from text chunks using FAISS and OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def format_docs(docs):
    """Format documents by concatenating their content."""
    return "\n\n".join(doc.page_content for doc in docs)

# GitHub repository URL and PDF file path
github_repo_url = "https://raw.githubusercontent.com/GeniusMundi10/LMgpt/main"
pdf_file_path = "LM_PCR_All_Amendements.pdf"
pdf_file = f"{github_repo_url}/{pdf_file_path}"

# Streamlit input and button
question = st.text_input("Enter Question")

if st.button("Get Answer"):
    text = get_pdf_text(pdf_file)
    if text:
        chunks = get_text_chunks(text)
        vector_store = get_vectorstore(chunks)
        retriever = vector_store.as_retriever()
        
        # Retrieve relevant context
        relevant_docs = retriever.get_relevant_documents(question)
        context = format_docs(relevant_docs)
        
        # Prepare the full prompt
        prompt = template.format(context=context, question=question)
        
        # Stream the response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            stream=True
        )

        # Display the streamed response
        st.header("Answer:")
        complete_answer = ""
        for chunk in response:
            chunk_message = chunk['choices'][0]['delta']
            if 'content' in chunk_message:
                complete_answer += chunk_message['content']
                st.write(chunk_message['content'])
