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
from langchain_pinecone import PineconeVectorStore
from io import BytesIO
#load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#claudeai_key = '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'
model=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo",streaming=True)


parser = StrOutputParser()

chain = model | parser

st.title("ELT LM GPT")

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
#print(pdf_file)

#print(text)

#try:
#	chain.invoke({
#		"context": text,
#		"question": "What are the  Legal Metrology amendments. "
#		})
#except Exception as e:
#	print(e)	







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
        prompt_text = template.format(context=context, question=question)

        # Set up the runnable pipeline for streaming response
        setup = RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
        chain = setup | prompt | model | parser

        st.header("Answer:")
        response_placeholder = st.empty()  # Placeholder for streaming response
        complete_answer = ""

        # Invoke the chain with streaming
        for chunk in chain.stream(question):
            if chunk:
                complete_answer += chunk
                response_placeholder.markdown(complete_answer)
