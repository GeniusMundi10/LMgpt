
import os
#from dotenv() import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from PyPDF2 import PdfReader
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma,FAISS,DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import streamlit as st
#load_dotenv

OPENAI_API_KEY = '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'
claudeai_key = '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'
model=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo")
#print(model)
#answer=model.invoke("Who will win the  elections in India")
#print(answer)

parser = StrOutputParser()

chain = model | parser
#answer=chain.invoke("What cricket team won the icc cricket world cup 2023 last year?")
#print(answer)
st.title("ELT LM GPT")

template = """
Answer the question based on the context below. If you can't answer the question reply "I dont Know".

Context:{context}

Question:{question}
"""

prompt = ChatPromptTemplate.from_template(template)
prompt.format(context = "Jatin has three girlfriends.",question="How many  girlfriends does jatin have?")
#print(a)
chain = prompt | model | parser
ans=chain.invoke({"context":"LangChain is a framework designed to simplify the process of building applications that leverage large language models (LLMs).  Imagine building applications like chatbots or virtual assistants without needing to be an expert in complex LLM interactions. LangChain provides a toolbox of components and functionalities like data retrieval and reasoning, allowing developers to chain these elements together to create intelligent applications. This not only streamlines development but also empowers those less familiar with LLMs to experiment and prototype LLM-powered applications.","question":"Can you elaborate about this paragraph in detail"})
#print(ans)



def get_pdf_text(pdf):
    text = ""
    
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


home_directory = os.path.expanduser("~")
pdf_file = os.path.join(home_directory,"Downloads","LM_PCR_All_Amendements.pdf")
#print(pdf_file)
text = get_pdf_text(pdf_file)
#print(text)

#try:
#	chain.invoke({
#		"context": text,
#		"question": "What are the  Legal Metrology amendments. "
#		})
#except Exception as e:
#	print(e)	





def get_text_chunks(text):
	text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=200,length_function=len)
	chunks = text_splitter.split_text(text)
	return chunks

def get_vectorstore(text_chunks):
	embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
	#vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
	vector_store = DocArrayInMemorySearch.from_texts(text_chunks,embeddings)
	return vector_store

question = st.text_input("Enter Question")

if st.button("Get Answer"):
	text = get_pdf_text(pdf_file)
	chunks=get_text_chunks(text)
#print (len(chunks))
	vector_store=get_vectorstore(chunks)

	retriever=vector_store.as_retriever()

	query = "Summarize the key points from the PDF"
#a=retriever.invoke(query)
#print(a)
	docs=vector_store.similarity_search(query)
#a=chain.invoke({
#	"context":docs,
#	"question" : query
#	})
#print(a)
	setup = RunnableParallel(context = retriever, question=RunnablePassthrough())
#a=setup.invoke("WHat are the amendments number?")
#print(a)

	chain = setup | prompt | model | parser
	answer=chain.invoke(question)
	st.header("Answer:")
	st.write(answer)

