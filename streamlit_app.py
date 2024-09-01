import streamlit as st

import json
import requests
import time

from newspaper import Article

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# from langchain.chains import RetrievalQA

# from langchain.document_loaders import UnstructuredHTMLLoader

from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter

st.title("Simple app to demonstrate Retrieval Augmented Generation, also known as RAG")
st.markdown("Enter a URL that you want the app to use as context, and the question you want answered")

with st.form("user_inputs_for_rag_form", clear_on_submit=False):
	st.session_state['API_KEY'] = st.text_input("Insert your Hugging Face Access Token here")
	st.session_state['url'] = st.text_input("Insert the URL you would like to provide as context")
	st.session_state['question'] = st.text_input("Your question here")
	submit = st.form_submit_button("Submit")


def query(API_URL, headers, payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

if submit:
    article = Article(st.session_state['url'])
    article.download()
    article.parse()
    st.session_state['article_title'] = article.title
    st.session_state['article_text'] = article.text


if 'article_text' in st.session_state and st.session_state['article_text'] is not None:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    st.session_state['docs'] = text_splitter.create_documents([st.session_state['article_text']])
    st.write(f"Content from the context URL retrieved, {len(st.session_state['docs'])} document chunks created")
    # st.write(st.session_state)

if 'docs'in st.session_state and st.session_state['docs'] is not None:
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs 
    )
    db = FAISS.from_documents(st.session_state['docs'], embeddings)
    searchDocs = db.similarity_search(st.session_state['question'])
    st.session_state['most_relevant_chunk'] = searchDocs[0].page_content
    # for eachresult in searchDocs:
    	# print(f"{eachresult.page_content} *** with metadata [{eachresult.metadata}]")
    # st.write(f"The most relevant document chunk identified is {searchDocs[0].page_content}")

if 'most_relevant_chunk' in st.session_state and st.session_state['most_relevant_chunk'] is not None:
    INFERENCE_API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
    message = [
        {"role": "system", "content": "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users. Respond back in a single sentence, not more."}, 
        {"role": "user", "content": f"Here is the context: {st.session_state['most_relevant_chunk']}. Now answer the following question from the user: {st.session_state['question']}"},
    ]
    headers = {"Authorization": f"Bearer {st.session_state['API_KEY']}"}
    output = query(INFERENCE_API_URL, headers, {"inputs": str(message),"wait_for_model": True})
    st.divider()
    st.write(output[0])