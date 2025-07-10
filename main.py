from fastapi import FastAPI
from pydantic import BaseModel
from rag_agent import agent
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Health RAG API", description="RAG agent powered by Gemini, LangGraph, and Pinecone")

# Optional CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    result = agent.invoke(
        {"messages": [HumanMessage(content=request.question)]},
    )
    if result:
        return result
    else: 
        return "API Limits reached for Gemini"
