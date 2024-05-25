from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.callbacks import get_openai_callback
import streamlit as st


# Function for API configuration at sidebar
def sidebar_api_key_configuration():
    st.sidebar.subheader("API Keys")
    api_key = st.sidebar.text_input("Enter your API Key 🗝️", type="password",
                                    help='Get API Key from: https://platform.openai.com/api-keys')
    if api_key == '':
        st.sidebar.warning('Enter the API Key 🗝️')
        st.session_state.prompt_activation = False
    elif api_key.startswith('sk-') and (len(api_key) == 51):
        st.sidebar.success('Lets Proceed!', icon='️👉')
        st.session_state.prompt_activation = True
    else:
        st.sidebar.warning('Please enter the correct API Key 🗝️!', icon='⚠️')
        st.session_state.prompt_activation = False
    return api_key


# Read PDF data
def read_pdf_data(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split data into chunks
def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text)
    return text_chunks


# Create vectorstore
def create_vectorstore(openai_api_key, pdf_docs):
    raw_text = read_pdf_data(pdf_docs)  # Get PDF text
    text_chunks = split_data(raw_text)  # Get the text chunks
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Get response from llm of user asked question
def get_llm_response(llm, prompt, question):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(st.session_state.vector_store.as_retriever(), document_chain)
    with get_openai_callback() as cb:
        response = retrieval_chain.invoke({'input': question})
        st.session_state.total_token += cb.total_tokens
        st.session_state.total_cost += cb.total_cost
        st.session_state.successful_requests += cb.successful_requests
        return response

