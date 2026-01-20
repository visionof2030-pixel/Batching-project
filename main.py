import os
import json
import math
import uuid
import itertools
import datetime
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import google.generativeai as genai

MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
MAX_RETRY = 3
MAX_TOTAL = 200

DATABASE_URL = "sqlite:///licenses.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class LicenseKey(Base):
    __tablename__ = "license_keys"

    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, index=True)
    expires_at = Column(DateTime)
    max_requests = Column(Integer)
    used_requests = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

    def is_valid(self):
        return (
            self.is_active
            and self.used_requests < self.max_requests
            and datetime.datetime.utcnow() < self.expires_at
        )

Base.metadata.create_all(engine)

keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12)]
keys = [k for k in keys if k]
if not keys:
    raise RuntimeError("No Gemini API keys found")

key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(MODEL)

def safe_json(text: str):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return None

def lang_instruction(lang: str):
    return (
        "Write the final output in clear academic English."
        if lang == "en"
        else "اكتب الناتج النهائي باللغة العربية الفصحى الواضحة."
    )

def build_prompt(topic: str, lang: str, count: int):
    return f"""
{lang_instruction(lang)}

أنشئ {count} سؤال اختيار من متعدد من الموضوع التالي.

قواعد صارمة:
- 4 خيارات فقط لكل سؤال
- شرح موسع للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- لا تكرر الأفكار
- أعد JSON فقط

الصيغة:
{{
  "questions": [
    {{
      "q": "",
      "options": ["", "", "", ""],
      "answer": 0,
      "explanations": ["", "", "", ""]
    }}
  ]
}}

الموضوع:
{topic}
"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int = 10

class CreateLicenseRequest(BaseModel):
    secret: str
    days: int
    max_requests: int

ADMIN_SECRET = os.getenv("ADMIN_SECRET")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_license(license_key: str = Header(...), db=Depends(get_db)):
    lic = db.query(LicenseKey).filter(LicenseKey.key == license_key).first()
    if not lic or not lic.is_valid():
        raise HTTPException(status_code=403, detail="Invalid or expired license")
    lic.used_requests += 1
    db.commit()
    return lic

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/admin/create-license")
def create_license(req: CreateLicenseRequest, db=Depends(get_db)):
    if req.secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    key = "ST-" + uuid.uuid4().hex[:12].upper()
    expires = datetime.datetime.utcnow() + datetime.timedelta(days=req.days)

    lic = LicenseKey(
        key=key,
        expires_at=expires,
        max_requests=req.max_requests,
        used_requests=0,
        is_active=True
    )

    db.add(lic)
    db.commit()

    return {
        "license_key": key,
        "expires_at": expires.isoformat(),
        "max_requests": req.max_requests
    }

@app.post("/generate/batch")
def generate_batch(req: GenerateRequest, lic=Depends(verify_license)):
    total = min(max(req.total_questions, 1), MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    final_questions = []

    for batch_index in range(batches):
        remaining = total - len(final_questions)
        if remaining <= 0:
            break

        needed = min(BATCH_SIZE, remaining)

        for attempt in range(MAX_RETRY):
            model = get_model()
            prompt = build_prompt(req.topic, req.language, needed)

            try:
                response = model.generate_content(prompt)
                data = safe_json(response.text)

                if not data or "questions" not in data:
                    raise ValueError("Invalid JSON")

                final_questions.extend(data["questions"][:needed])
                break

            except Exception as e:
                if attempt == MAX_RETRY - 1:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Batch failed: {str(e)}"
                    )

    if len(final_questions) != total:
        raise HTTPException(
            status_code=500,
            detail=f"Generated {len(final_questions)} out of {total}"
        )

    return {"questions": final_questions}