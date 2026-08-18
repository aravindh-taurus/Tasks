import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import trim_messages
from langsmith import traceable

#Phase 2
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import groq
#Phase 3
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever

# loading environmental variables
load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HUGGING_FACE_ACCESS_TOKEN")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

#custom function to create retrievers
def ordinary_retriver(path):

    loader = PyPDFLoader(path)
    chunks = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
    )
    final_chunk = text_splitter.split_documents(chunks)

    embeds = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(final_chunk,embeds)

    ordinary_retr = vector_store.as_retriever(search_kwargs={"k": 2})
    return ordinary_retr

# prompting 
def strict_prompt():
    system_prompt = (
        """ 
        You are an helpful assistant for question-answer tasks.
        Use the following pieces of retrieved context to answer the questions.
        If you don't know the answer, say that you don't know with apologies.
        If answer to the question is not found in the context, You must respond Exactly with: I am sorry
        but the answer to the question is not available in the uploaded file. Please try with different questions!.
        Context:
        {context}
        """
    )
    strt_prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("human","{input}")
    ])
    return strt_prompt

#prompting 
def contextual_prompting():
    contextual_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a standalone question which can be understand"
        "without the chat history. DO NOT answer the question,"
        "just reformulate it if needed and otherwise return as it is"
    )
    contextual_prmpt = ChatPromptTemplate.from_messages(
        [
            ("system",contextual_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )
    return contextual_prmpt
# Basic rag
def basic_rag(strt_prompt,llm,ordinary_rtr):
    question_answer_llm_chain = create_stuff_documents_chain(llm,strt_prompt)
    basic_rag_chain = create_retrieval_chain(ordinary_rtr,question_answer_llm_chain)
    return basic_rag_chain
#Advanced Rag
def advanced_rag(contextual_pmt,llm,ordinary_rtr):
     history_aware_retriever = create_history_aware_retriever(llm,ordinary_rtr,contextual_pmt)
     strt_prompt = (
             """
             You are an helpful assistant for question-answer tasks.
             Use the following pieces of retrieved context to answer the questions.
             If you don't know the answer, say that you don't know with apologies.
             If answer to the question is not found in the context, You must respond Exactly with:
             I am sorry!. But the answer to the question is not available in the uploaded file. Please try with different question!.
             
             Context:
             {context}
             """
          )
     prmpt_wt_htry = ChatPromptTemplate.from_messages(
         [
            ("system",strt_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}") 
         ]
     )
     stuffing_llm = create_stuff_documents_chain(llm,prmpt_wt_htry)
     advanced_rag_chain = create_retrieval_chain(history_aware_retriever,stuffing_llm)
     return advanced_rag_chain


@traceable(run_type="llm")
def call_llm(groq_api_key,llm_name):
    llm = ChatGroq(groq_api_key=groq_api_key,model_name=llm_name)
    return llm
# Trim Messages
def trim_msg(model,msg_hstry):
    trimmer = trim_messages(
        max_tokens=1000,                  
        strategy="last",                  
        token_counter=model,              
        include_system=True,             
        start_on="human",)
    trimmed_hstry = trimmer.invoke(msg_hstry)
    return trimmed_hstry

store={}
def get_chat_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

with st.sidebar:
    models = ["groq/compound-mini","groq/compound","openai/gpt-oss-120b"]
    model = st.multiselect("Choose Model",options=models,default="openai/gpt-oss-120b")

folder_path = "Data"
os.makedirs(folder_path,exist_ok=True)

if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None

uploaded_file = st.file_uploader(
    label="Choose the file to upload",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_file:
    if st.session_state.processed_filename != uploaded_file.name:
        file_path = os.path.join(folder_path,uploaded_file.name)
        with st.spinner(f"Uploading the file..."):
            time.sleep(3)
            try:
                with open(file_path,"wb") as f:
                    f.write(uploaded_file.getbuffer())
                
            except Exception as e:
                st.error(f"failed to save file because {e}")
        st.session_state.processed_filename = uploaded_file.name
        st.toast(f"✅ {uploaded_file.name} processed successfully!", icon="🎯")
        st.info(f"Currently active document: {st.session_state.processed_filename}")
else:
    st.warning("Please upload the file")

user_input = st.chat_input("Hi There! I am a Custom QA ChatBot. Glad to help you. Go with your queries below!")
history = StreamlitChatMessageHistory()
for msg in history.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("ai"):
            st.markdown(msg.content)

if user_input:

    history.add_user_message(user_input) 
    file_path = os.path.join(folder_path,uploaded_file.name)
    ord_rtr = ordinary_retriver(file_path)
    strt_prmpt = strict_prompt()
    cntxt_prmpt = contextual_prompting()
    llm = call_llm(groq_api_key,model[0])
    basic_rag_chain = basic_rag(strt_prmpt,llm,ord_rtr)
    advanced_rag_chain = advanced_rag(cntxt_prmpt,llm,ord_rtr)
    # response = basic_rag_chain.invoke({"input":user_input})
    try:
        trimmed_messages = trim_msg(llm,history.messages)
        response = advanced_rag_chain.invoke({"input":user_input,"chat_history":trimmed_messages})
        history.add_ai_message(response["answer"])

        if response:

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("ai"):
                st.markdown(response["answer"])
        else:
            st.write("Sorry Unable to fetch the data. Please search again")
    except groq.RateLimitError as e1:
        st.warning("Sorry rate limit reached. Please try again after some time")
    except Exception as e2:
        st.warning("Sorry an unexpected error occured. Please try again after some time")
        # st.write(e2)











            
