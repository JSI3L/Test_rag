from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from chunking import process_documents
import os
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

vector_store = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory="./db",  # Where to save data locally
)
import hashlib

def get_document_id(doc):
    """Genera un ID único basado en el contenido y fuente del documento."""
    content = f"{doc.metadata.get('source', '')}_{doc.page_content}"
    return hashlib.md5(content.encode()).hexdigest()

docs = process_documents()
ids = [get_document_id(doc) for doc in docs]
vector_store.add_documents(documents=docs, ids=ids)


