import requests
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from PyPDF2 import PdfReader
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma,FAISS,DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import streamlit as st
from io import BytesIO
#load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#claudeai_key = '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'
model=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo",streaming=True,max_tokens=4096)


st.set_page_config(page_title="ELT LM GPT", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f0f0f5;
    }
    .chat-bubble {
        margin: 10px 0;
        padding: 10px;
        border-radius: 10px;
        max-width: 80%;
        word-wrap: break-word;
    }
    .chat-bubble.user {
        background-color: #d1e7dd;
        align-self: flex-end;
    }
    .chat-bubble.assistant {
        background-color: #ffe8a1;
        align-self: flex-start;
    }
    .container {
        display: flex;
        flex-direction: column;
    }
    .centered {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='centered'>ELT LM GPT</h1>", unsafe_allow_html=True)


template = """
You are an assistant for question-answering tasks in detail. Answer the question based on the context below. If you can't answer the question reply "I dont Know". Keep the answer lengthy and informative.

Context:{context}

Question:{question}
"""


prompt = ChatPromptTemplate.from_template(template)


def get_pdf_text(pdf_url):
	response = requests.get(pdf_url)
	if response.status_code == 200:
		pdf_content = BytesIO(response.content)
		pdf_reader = PdfReader(pdf_content)
		text = ""
		for page in pdf_reader.pages:
			text += page.extract_text()
		return text
	else:
		raise Exception(f"Failed to fetch PDF content. Status code:{response.status_code}")



def get_text_chunks(text):
	text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=200,length_function=len)
	chunks = text_splitter.split_text(text)
	return chunks

def get_vectorstore(text_chunks):
	embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
	vector_store = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
	#vector_store = DocArrayInMemorySearch.from_texts(text_chunks,embeddings)
	return vector_store
def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)
#home_directory = os.path.expanduser("~")
#pdf_file = os.path.join(home_directory,"Downloads","LM_PCR_All_Amendements.pdf")
github_repo_url = "https://raw.githubusercontent.com/GeniusMundi10/LMgpt/main"

# PDF file path within the repository
pdf_file_path = "LM_PCR_All_Amendements.pdf"

# Combine GitHub URL and PDF file path to get the raw content URL
pdf_file= f"{github_repo_url}/{pdf_file_path}"



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []	


st.markdown("<h3 class='centered'>Ask a Question</h3>", unsafe_allow_html=True)
question = st.text_input("", key="input")

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
        prompt_text = template.format(context=context, question=question)

        # Set up the runnable pipeline for streaming response
        setup = RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
        chain = setup | prompt | model | parser

        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})

        response_placeholder = st.empty()  # Placeholder for streaming response
        complete_answer = ""

        # Invoke the chain with streaming
        for chunk in chain.stream(question):
            if chunk:
                complete_answer += chunk
                response_placeholder.markdown(complete_answer)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": complete_answer})

# Display chat history in a better layout
st.markdown("<h3 class='centered'>Chat History</h3>", unsafe_allow_html=True)
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"<div class='chat-bubble user'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble assistant'><b>Assistant:</b> {message['content']}</div>", unsafe_allow_html=True)
