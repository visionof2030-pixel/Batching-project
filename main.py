from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from batching import generate_questions_pro

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int = 10

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: QuizRequest):
    try:
        return generate_questions_pro(
            topic=req.topic,
            total_questions=req.total_questions,
            language=req.language
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))