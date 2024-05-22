from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
# from dotenv import load_dotenv
import chromadb
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
# from constants import CHROMA_SETTINGS
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory

st.set_page_config(layout="centered")

st.markdown("<h2 style='font-family:sans-serif;text-align: center; color: Black;'> Welcome to GenAI</h2>", unsafe_allow_html=True)

st.markdown("<h4 style='font-family:sans-serif;text-align: center; color: Grey;'> Lets chat about Genetics </h4>", unsafe_allow_html=True)

os.environ['OPENAI_API_KEY']=''
#isert your api key here

os.environ['REQUESTS_CA_BUNDLE'] = ''
#insert your certificate perm here

# loader=CSVLoader("Book1.csv")
# data=loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=450,chunk_overlap=100, separators="\n\n")
# chunks = text_splitter.split_documents(data)

# embeddings=OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

db = Chroma(embedding_function = embeddings,persist_directory="Chromadb/chroma_new")


retriever = db.as_retriever(search_kwargs={"k": 3})

# template2 = """You are a nice chatbot. You were created by a ML Engineer named Ayush. Your task is to answer the user's queries on a story book called 'Ares: Saviour of Hala'.
# Look for the information asked by the user, and if user is asking some creative titles/summary, plot points, try to understand the story, characters & plots to creatively answer. 
# Keep the answer as concise as possible. 
# {context}


# Search for 
# Question: {question}
# """
template2 = """You are a nice chatbot. You were created by a ML Engineer named Aditi. Your task is to answer the user's queries on a summary of chapter of genetics.
Look for the information asked by the user, Keep the answer as concise as possible. Just answer from the given context , don't entertain answers other than mentioned in the context.
{context}


Search for 
Question: {question}
"""
memory = ConversationBufferWindowMemory(
    k=2,
    memory_key="chat_history",
    return_messages=True
)


QA_CHAIN_PROMPT = PromptTemplate(input_variables=["question"],template=template2)
chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0), retriever=retriever,memory=memory,combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},verbose=True)
# chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0), retriever=db.as_retriever())


def conversational_chat(query):
        response = chain({"question": query})
        st.session_state['history'].append((query, response['answer']))
        return response['answer']
    
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about genetics 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Type your query...", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
