import os
from dotenv import load_dotenv  
from langchain_openai import OpenAIEmbeddings
import ../chunking as chunking
from supabase import create_client , client
from langchain_community.vectorstores import SupabaseVectorStore

load_dotenv()  

chunks = chunking.process_documents()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

client = create_client(supabase_url, supabase_key)

vectorstore = SupabaseVectorStore.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    client=client,
    table_name="documents_rag", #Esta es la tabla que hay que crear en Supabase, debe tener una columna "content" para el texto y una columna "embedding" para el vector de embedding
    query_name="match_documents_rag",
)

print(f"Documentos procesados y almacenados en Supabase{vectorstore}")

retrieved_docs = vectorstore.as_retriever()
