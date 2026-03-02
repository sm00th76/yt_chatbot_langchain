from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import ask_question

app = FastAPI()

class QueryRequest(BaseModel):
    url: str
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def HomePage():
    return {"status": "ok"}

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    try:
        answer = ask_question(request.url, request.question)
        return QueryResponse(
            answer=answer
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))