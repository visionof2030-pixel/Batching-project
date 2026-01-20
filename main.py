from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from batching import generate_batch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BatchRequest(BaseModel):
    topic: str
    language: str = "ar"
    batch_size: int = 20
    offset: int = 0  # فقط للواجهة (اختياري)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/generate/batch")
def generate_batch_endpoint(req: BatchRequest):
    return {
        "offset": req.offset,
        "batch_size": req.batch_size,
        "data": generate_batch(
            topic=req.topic,
            batch_size=req.batch_size,
            language=req.language
        )
    }