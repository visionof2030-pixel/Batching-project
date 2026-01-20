import os, json, math, itertools, datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

from database import SessionLocal, LicenseKey

MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
MAX_RETRY = 3
MAX_TOTAL = 200

keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12)]
keys = [k for k in keys if k]
key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(MODEL)

def safe_json(text):
    try:
        return json.loads(text[text.find("{"):text.rfind("}")+1])
    except:
        return None

def build_prompt(topic, lang, count):
    return f"""
اكتب الناتج باللغة {"العربية" if lang=="ar" else "الإنجليزية"}.
أنشئ {count} سؤال اختيار من متعدد.
أعد JSON فقط بنفس الصيغة المعروفة.
الموضوع: {topic}
"""

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int
    license_key: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def validate_key(key: str, db):
    lk = db.query(LicenseKey).filter(LicenseKey.key == key).first()
    if not lk or not lk.is_valid():
        raise HTTPException(status_code=403, detail="Invalid or expired license key")
    lk.used_requests += 1
    db.commit()

@app.post("/generate/batch")
def generate(req: GenerateRequest, db=Depends(get_db)):
    validate_key(req.license_key, db)

    total = min(req.total_questions, MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    final_questions = []

    for _ in range(batches):
        needed = min(BATCH_SIZE, total - len(final_questions))
        if needed <= 0:
            break

        for _ in range(MAX_RETRY):
            model = get_model()
            data = safe_json(model.generate_content(
                build_prompt(req.topic, req.language, needed)
            ).text)

            if data and "questions" in data:
                final_questions.extend(data["questions"][:needed])
                break

    if len(final_questions) != total:
        raise HTTPException(500, "Generation incomplete")

    return {"questions": final_questions}