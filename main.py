from fastapi import FastAPI
from pydantic import BaseModel
from batching import generate_questions_pro

app = FastAPI()

class QuizRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int = 10

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: QuizRequest):
    return generate_questions_pro(
        topic=req.topic,
        total_questions=req.total_questions,
        language=req.language
    )