# main.py
import os
import json
import math
import itertools
import secrets
import datetime
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
MAX_TOTAL = 200

DATABASE_URL = "sqlite:///licenses.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class LicenseKey(Base):
    __tablename__ = "license_keys"
    id = Column(Integer, primary_key=True)
    license_key = Column(String, unique=True, index=True)
    expires_at = Column(DateTime)
    max_requests = Column(Integer)
    used_requests = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    device_id = Column(String, nullable=True)

    def is_valid(self):
        return (
            self.is_active
            and self.used_requests < self.max_requests
            and datetime.datetime.utcnow() < self.expires_at
        )

Base.metadata.create_all(engine)

keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12) if os.getenv(f"GEMINI_KEY_{i}")]
if not keys:
    raise RuntimeError("No Gemini API keys configured")

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

def build_prompt(text: str, count: int, lang: str):
    lang_line = "اكتب باللغة العربية الفصحى." if lang == "ar" else "Write in clear English."
    return f"""
{lang_line}

أنشئ {count} أسئلة اختيار من متعدد من النص التالي.

قواعد:
- 4 خيارات لكل سؤال
- شرح موسع وواضح للإجابة الصحيحة
- شرح مختصر ودقيق لكل خيار خاطئ
- أعد JSON فقط

الصيغة:
{{
 "questions":[
   {{
     "q":"",
     "options":["","","",""],
     "answer":0,
     "explanations":["","","",""]
   }}
 ]
}}

النص:
{text}
"""

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

class GenerateRequest(BaseModel):
    topic: str = ""
    language: str = "ar"
    total_questions: int = 5
    mode: str = "manual"

@app.post("/generate/batch")
def generate(
    req: GenerateRequest,
    license_key: str = Header(...),
    device_id: str = Header(...)
):
    db = SessionLocal()
    lic = db.query(LicenseKey).filter_by(license_key=license_key).first()
    if not lic or not lic.is_valid():
        raise HTTPException(403, "Invalid license")

    if lic.device_id and lic.device_id != device_id:
        raise HTTPException(403, "License already used on another device")
    if not lic.device_id:
        lic.device_id = device_id

    total = min(req.total_questions, MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    questions = []

    try:
        for _ in range(batches):
            need = min(BATCH_SIZE, total - len(questions))
            if need <= 0:
                break
            model = get_model()
            prompt = build_prompt(req.topic, need, req.language)
            res = model.generate_content(prompt)
            data = safe_json(res.text)
            if not data or "questions" not in data:
                raise ValueError("Invalid AI response")
            questions.extend(data["questions"][:need])

        lic.used_requests += 1
        db.commit()
        return {"questions": questions}

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/admin/create")
def admin_create(days: int, max_requests: int, owner: str = "", x_admin_key: str = Header(...)):
    if x_admin_key != os.getenv("ADMIN_SECRET"):
        raise HTTPException(403)
    db = SessionLocal()
    key = "ST-" + secrets.token_hex(6).upper()
    lic = LicenseKey(
        license_key=key,
        expires_at=datetime.datetime.utcnow() + datetime.timedelta(days=days),
        max_requests=max_requests,
    )
    db.add(lic)
    db.commit()
    return {"license_key": key}

@app.get("/admin/licenses")
def admin_list(x_admin_key: str = Header(...)):
    if x_admin_key != os.getenv("ADMIN_SECRET"):
        raise HTTPException(403)
    db = SessionLocal()
    return db.query(LicenseKey).all()