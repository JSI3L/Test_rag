from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm

def process_documents() -> List[Document]:  
    all_docs: List[Document] = []
    path = Path("./docs")
    for doc_txt in path.glob("*.txt"):
        try:
            with open(doc_txt, "r", encoding="utf-8") as f:
                content = f.read()
            all_docs.append(Document(
                page_content=content,
                metadata={"source": doc_txt.name, "type": "txt"}
            ))
            print(f"Procesado documento de texto {doc_txt.name} con éxito.")
        except Exception as e:
            print(f"Error al procesar {doc_txt.name}: {e}")
    for doc_pdf in path.glob("*.pdf"):
        try:
            pages = pymupdf4llm.to_markdown(str(doc_pdf), page_chunks=True)
            if not pages:
                print(f"{doc_pdf.name} no tiene texto extraíble")
                continue
            for page in pages:
                text = page.get("text", "").strip()
                if text: 
                    all_docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": doc_pdf.name,
                            "page": page.get("metadata", {}).get("page", 0),
                            "type": "pdf"
                            }))
            print(f"Procesado {doc_pdf.name} con éxito. ({len(pages)} páginas)")
        except Exception as e:
            print(f"Error al procesar {doc_pdf.name}: {e}")
    print(f"Total de documentos procesados: {len(all_docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=32
    )
    chunks = text_splitter.split_documents(all_docs)
    
    print(f"Total de fragmentos creados: {len(chunks)}")          
    return chunks

chunks = process_documents()
print(f"Primer fragmento de texto: {chunks[0].page_content[:200]}...")
print(f"Último fragmento de texto: {chunks[len(chunks)-1].page_content[:200]}...")