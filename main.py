import os
import json
import math
import secrets
import itertools
import datetime
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import google.generativeai as genai

# ================== CONFIG ==================
MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
MAX_TOTAL = 200
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

# ================== GEMINI KEYS ==================
keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12)]
keys = [k for k in keys if k]
if not keys:
    raise RuntimeError("No Gemini API keys")

key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(MODEL)

# ================== DATABASE ==================
DATABASE_URL = "sqlite:///licenses.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class LicenseKey(Base):
    __tablename__ = "licenses"

    id = Column(Integer, primary_key=True)
    license_key = Column(String, unique=True, index=True)
    expires_at = Column(DateTime)
    max_requests = Column(Integer)
    used_requests = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    device_id = Column(String, nullable=True)
    owner = Column(String, nullable=True)

    def valid(self, device):
        return (
            self.is_active
            and self.used_requests < self.max_requests
            and datetime.datetime.utcnow() < self.expires_at
            and (self.device_id in [None, device])
        )

Base.metadata.create_all(engine)

# ================== FASTAPI ==================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"]
)

# ================== MODELS ==================
class GenerateReq(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int

class CreateKeyReq(BaseModel):
    days: int
    max_requests: int
    owner: str | None = None

class UpdateKeyReq(BaseModel):
    is_active: bool

# ================== HELPERS ==================
def safe_json(text):
    try:
        return json.loads(text[text.find("{"):text.rfind("}")+1])
    except:
        return None

def prompt(topic, lang, count):
    return f"""
أنشئ {count} سؤال اختيار من متعدد.
- 4 خيارات
- شرح موسع للإجابة الصحيحة
- شرح مختصر ودقيق لبقية الخيارات
- أعد JSON فقط

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

الموضوع:
{topic}
"""

# ================== GENERATE ==================
@app.post("/generate/batch")
def generate(
    req: GenerateReq,
    license_key: str = Header(...),
    device_id: str = Header(...)
):
    db = SessionLocal()
    lic = db.query(LicenseKey).filter_by(license_key=license_key).first()
    if not lic or not lic.valid(device_id):
        raise HTTPException(403, "License invalid")

    if lic.device_id is None:
        lic.device_id = device_id

    total = min(req.total_questions, MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    questions = []

    for _ in range(batches):
        need = min(BATCH_SIZE, total - len(questions))
        if need <= 0:
            break
        model = get_model()
        r = model.generate_content(prompt(req.topic, req.language, need))
        data = safe_json(r.text)
        if not data:
            raise HTTPException(500, "Model error")
        questions.extend(data["questions"][:need])

    lic.used_requests += 1
    db.commit()

    return {"questions": questions}

# ================== ADMIN ==================
def check_admin(x_admin_key: str):
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(401, "Unauthorized")

@app.get("/admin/licenses")
def admin_list(x_admin_key: str = Header(...)):
    check_admin(x_admin_key)
    db = SessionLocal()
    return db.query(LicenseKey).all()

@app.post("/admin/create")
def admin_create(req: CreateKeyReq, x_admin_key: str = Header(...)):
    check_admin(x_admin_key)
    db = SessionLocal()
    key = "ST-" + secrets.token_hex(6).upper()
    lic = LicenseKey(
        license_key=key,
        expires_at=datetime.datetime.utcnow() + datetime.timedelta(days=req.days),
        max_requests=req.max_requests,
        owner=req.owner
    )
    db.add(lic)
    db.commit()
    return {"license_key": key}

@app.put("/admin/update/{key}")
def admin_update(key: str, req: UpdateKeyReq, x_admin_key: str = Header(...)):
    check_admin(x_admin_key)
    db = SessionLocal()
    lic = db.query(LicenseKey).filter_by(license_key=key).first()
    lic.is_active = req.is_active
    db.commit()
    return {"ok": True}

@app.post("/admin/reset-device/{key}")
def admin_reset(key: str, x_admin_key: str = Header(...)):
    check_admin(x_admin_key)
    db = SessionLocal()
    lic = db.query(LicenseKey).filter_by(license_key=key).first()
    lic.device_id = None
    db.commit()
    return {"ok": True}

@app.delete("/admin/delete/{key}")
def admin_delete(key: str, x_admin_key: str = Header(...)):
    check_admin(x_admin_key)
    db = SessionLocal()
    db.query(LicenseKey).filter_by(license_key=key).delete()
    db.commit()
    return {"ok": True}