from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from embeddings import vector_store
 
app = FastAPI(title="QA Agent API")
 
system_prompt = (
    "you are an question-answering assistant "
    "uses the following retrieved documents to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Always use all the retrieved documents to answer the question. "
    "use three sentences at most to answer the question. "
    "context: {context} "
)
 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieves relevant documents from the vector store based on the query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
 
tools = [retrieve_context]
model = ChatOpenAI(model="gpt-3.5-turbo")

agent = create_agent(model, tools, system_prompt=system_prompt)
 
class QueryRequest(BaseModel):
    query: str
 
class QueryResponse(BaseModel):
    answer: str
 
@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """Returns the final answer from the agent (non-streaming)."""
    try:
        final_message = None
        for event in agent.stream(
            {"messages": [{"role": "user", "content": request.query}]},
            stream_mode="values",
        ):
            final_message = event["messages"][-1]
 
        if final_message is None:
            raise HTTPException(status_code=500, detail="Agent returned no response.")
 
        return QueryResponse(answer=final_message.content)
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
