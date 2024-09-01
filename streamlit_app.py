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

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("Simple app to demonstrate Retrieval Augmented Generation, also known as RAG")
st.markdown("Enter a URL that you want the app to use as context, and the question you want answered")

with st.form("user_inputs_for_rag_form", clear_on_submit=False):
	st.session_state['API_KEY'] = st.text_input("Insert your Groq API KEY here")
	st.session_state['url'] = st.text_input("Insert the URL you would like to provide as context")
	st.session_state['question'] = st.text_input("Your question here")
	submit = st.form_submit_button("Submit")


# def query(API_URL, headers, payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()

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
    st.write(f"Document chunk most relevant to the question retrieved via similarity search")


if 'most_relevant_chunk' in st.session_state and st.session_state['most_relevant_chunk'] is not None:
	simple_prompt = PromptTemplate(
	    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
	    You are an AI assistant that answers questions for users using the given context. Answer questions using a direct style.\n 
	    Do not share more information that the requested by the users. Respond back in a single sentence, not more.\n
	    <|eot_id|><|start_header_id|>user<|end_header_id|>
	    Here is the context: {context_material}
	    Here is the user question: {user_question}
	    <|eot_id|>
	    <|start_header_id|>assistant<|end_header_id|>
	    """,
	    input_variables=["context_material", "user_question"],
	)
	llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=st.session_state['API_KEY']
	)
	response_chain = simple_prompt | llm | StrOutputParser()
	response = response_chain.invoke({
		"context_material": st.session_state['most_relevant_chunk'],
		"user_question": st.session_state['question'],
	})
	st.write(f"LLM invoked with context material and user question")
	st.divider()
	st.markdown("LLM has responded back with:")
	st.header(response)