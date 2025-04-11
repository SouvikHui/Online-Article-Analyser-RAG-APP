import os
import shutil
import numpy as np
import streamlit as st
import pickle
from typing import List
from groq import Groq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_nomic import NomicEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

vectorstore_folder = "faiss_store_rag.pkl"
NOMIC_API_KEY = os.environ['NOMIC_API_KEY']

st.set_page_config(page_title="Article Analyzer", page_icon="📰")
st.title("Article Analyzer")
st.sidebar.title("Article URLs")


# Initiate Session_state
if 'is_processed' not in st.session_state:
    st.session_state.is_processed = False
if 'chat_hist' not in st.session_state:
    st.session_state.chat_hist = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'clear_input' not in st.session_state:
    st.session_state.clear_input = False

# Reset Button
if st.sidebar.button('Reset'):
    if os.path.exists(vectorstore_folder):
        shutil.rmtree(vectorstore_folder)
    st.session_state.clear()
    st.rerun()

# Clear Chat Button
if st.sidebar.button('Clear Chat'):
    st.session_state.chat_hist = []
    st.success('✅ Chat HIstory Cleared ✅')

# URLs collection
def input_urls(): 
    num_urls = st.sidebar.number_input("How many articles?", min_value=1, max_value=10, value=2)
    urls = [st.sidebar.text_input(f"Article {i+1}") for i in range(num_urls)]
    return urls

# Process URLs
def process_url(urls):
    st.info('Processing URLs')
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    for d,u in zip(data,urls):
        d.metadata['source'] = u
    st.info('Splitting Data...')
    text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.'],
            chunk_size=1000,
            chunk_overlap=200
        )
    st.info("✂️ Splitting texts...")
    docs = text_splitter.split_documents(data)
    return docs

# Document Embedding
def embedding_doc(docs:List[Document]):
    embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1.5",nomic_api_key=NOMIC_API_KEY) ## Nomic embedding
    st.info("📐 Embedding & Storing...")
    if os.path.exists(vectorstore_folder):
        shutil.rmtree(vectorstore_folder)
    vectorstore_nomic = FAISS.from_documents(docs,embeddings_model)
    vectorstore_nomic.save_local(vectorstore_folder)
    vec_store = FAISS.load_local(vectorstore_folder,embeddings=embeddings_model,allow_dangerous_deserialization=True)
    st.session_state.retriever = vec_store.as_retriever()
    st.session_state.is_processed = True
    st.success('✅ Articles processed and Vectorstor DB loaded!')

# User's query on the provided article
def format_docs(docs:List[Document]):
    return '\n\n'.join(f"[Source:{doc.metadata.get('source')}]\n{doc.page_content}" for doc in docs)

def ask_question(ret,llm):
     
    RAG_SYSTEM_PROMPT = """
        You are a helpful assistant for question-answering based on provided article contents.

        Your response must follow this format:

            Relevant:
            <Shortened quoted text from the article...>

            Source:
            <URL of the quoted article>

            Explanation:
            <Explanation and additional insights based on the quote>

        Instructions for you:
        - Use only the content between the triple backticks below as context for your answer.
            Context:
                ```
                {context}
                ```
        - If the user asks for an explanation or comprehension, give a clear and simple one that a beginner can understand. Make it rich with
            information present in the context or in the entire article. Make sure to cover the full context or the article, 
            focusing on the 20% of key points that explain 80% of the topic, without missing any important parts. 
        - If no relevant content is found, reply:
            "It is not provided in the article, but I can assist you using my knowledge if you want. Would you like that?" Then in the next run, 
            if the user agrees or responds positively, access the previous chat_hist and provide an answer from your own knowledge:
            - If a known source is available, mention it.
            - Otherwise, say: "This is from my knowledge only. Kindly verify it on the internet."

        """
    RAG_HUMAN_PROMPT = "{input}"

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_HUMAN_PROMPT)
    ])

    # Show Chat History first
    with st.expander("🕒 Chat History", expanded=True):
        for speaker, msg in st.session_state.chat_hist:
            st.markdown(f"**{speaker}:**\n\n{msg}")
    # Ask input at the bottom
    if st.session_state.get("clear_input"):
        st.session_state["user_input"] = ""
        st.session_state.clear_input = False
    with st.container():
        query = st.text_area("Ask a Question:", key="user_input", placeholder="Type your question here")
        send = st.button("Send", use_container_width=True)
    
    if send and query:
        st.session_state.chat_hist.append(("You", query))
        rag_chain = (
        {
            "context": ret | RunnableLambda(format_docs), # Retrieve data from vectorstore -> format documents into a string
            "input": RunnablePassthrough() # Propogate the 'input' variable to the next step
        } 
        | RAG_PROMPT # Context and Input variables
        | llm # Using LLM with formatted prompt
        | StrOutputParser() # Parse through LLM response to get only the string response
    )
        response=rag_chain.invoke(query)

        st.session_state.chat_hist.append(("Assistant", response))
        st.session_state.clear_input = True
        st.rerun()

urls = input_urls()

if st.sidebar.button("Process URLs"):
    if os.path.exists(vectorstore_folder):
        shutil.rmtree(vectorstore_folder)
    data = process_url(urls)
    embedding_doc(data)

# if st.sidebar.button('Process PDF'):
#     pass

if not st.session_state.retriever and os.path.exists(vectorstore_folder):
    embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1.5", nomic_api_key=NOMIC_API_KEY)
    vec_store = FAISS.load_local(vectorstore_folder, embeddings=embeddings_model, allow_dangerous_deserialization=True)
    st.session_state.retriever = vec_store.as_retriever()
    st.session_state.is_processed = True

llm_groq = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9,max_retries=2)
if st.session_state.is_processed and st.session_state.retriever:
    ask_question(st.session_state.retriever,llm_groq)