import os
from dotenv import load_dotenv  
from supabase import create_client 
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document 
load_dotenv()  


supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)
chunks = [
    Document(page_content="Este es el primer fragmento de texto."),
    Document(page_content="Este es el segundo fragmento de texto."),
    Document(page_content="Este es el tercer fragmento de texto.")
] 

vectorstore = SupabaseVectorStore.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    client=supabase,
    table_name="documents_rag", #Esta es la tabla que hay que crear en Supabase, debe tener una columna "content" para el texto y una columna "embedding" para el vector de embedding
)
print(f"Documentos procesados y almacenados en Supabase{vectorstore}")