from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from config import Config
import os
import uuid

def process_and_store_document(pdf_file_path):
    """Process PDF -> chunks -> embeddings -> vector DB"""
    
    #unique directory name for this session
    session_id = str(uuid.uuid4())[:8]  
    vector_db_path = f"vector_db_{session_id}"
    
    # check the directory exists
    os.makedirs(vector_db_path, exist_ok=True)
    
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Use Cohere embeddings
    embedding_model = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=Config.COHERE_API_KEY,
        user_agent="langchain"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=vector_db_path
    )
    
    
    return len(chunks), vector_db_path