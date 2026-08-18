import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
import os
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
app = FastAPI(
    title="History-Aware RAG API",
    description="API endpoint for querying organization policy documents with full chat history memory.",
    version="1.0.0"
)

# 1. Define Request and Response Schemas using Pydantic
class ChatMessageSchema(BaseModel):
    role: str  # Must be "human" or "ai"
    content: str

class RAGRequest(BaseModel):
    input: str
    chat_history: List[ChatMessageSchema] = []

class RAGResponse(BaseModel):
    answer: str

try:
    os.environ["GROQ_API_KEY"] = "your_actual_groq_api_key"
    
    ord_rtr = ordinary_retriver()
    strt_prmpt = strict_prompt()
    cntxt_prmpt = contextual_prompting()
    llm = call_llm(os.environ["GROQ_API_KEY"])
    
    advanced_rag_chain = advanced_rag(cntxt_prmpt, strt_prmpt, llm, ord_rtr)
    
except Exception as e:
    print(f"CRITICAL: Failed to initialize RAG components: {e}")
    advanced_rag_chain = None

def format_chat_history(json_history: List[ChatMessageSchema]):
    langchain_messages = []
    for msg in json_history:
        if msg.role.lower() == "human":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role.lower() == "ai":
            langchain_messages.append(AIMessage(content=msg.content))
    return langchain_messages

@app.post("/api/v1/chat", response_model=RAGResponse)
async def chat_endpoint(payload: RAGRequest):
    if advanced_rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG system pipeline is uninitialized.")
    
    try:

        formatted_history = format_chat_history(payload.chat_history)
        
        response = advanced_rag_chain.invoke({
            "input": payload.input,
            "chat_history": formatted_history
        })
        
        return RAGResponse(answer=response["answer"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")
