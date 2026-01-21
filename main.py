import os, json, math, itertools, secrets
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pdfplumber
import google.generativeai as genai

MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
MAX_TOTAL = 200

ADMIN_SECRET = os.getenv("ADMIN_SECRET")
if not ADMIN_SECRET:
    raise RuntimeError("ADMIN_SECRET not set")

keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12)]
keys = [k for k in keys if k]
if not keys:
    raise RuntimeError("No Gemini API keys found")

key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(MODEL)

DB_FILE = "licenses.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def now():
    return datetime.utcnow()

def safe_json(text: str):
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        return json.loads(text[s:e])
    except:
        return None

def build_prompt(topic, lang, count):
    return f"""
أنشئ {count} سؤال اختيار من متعدد.

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

الموضوع:
{topic}
"""

def build_prompt_from_text(text, lang, count):
    return f"""
اعتمد فقط على النص التالي دون إضافة أي معلومات خارجية:

{text}

أنشئ {count} سؤال اختيار من متعدد بنفس الصيغة المعروفة.
"""

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_from_image_ai(file):
    image = Image.open(file).convert("RGB")
    model = get_model()
    prompt = "استخرج النص الموجود في الصورة بدقة وأعد النص فقط."
    res = model.generate_content([prompt, image])
    return res.text.strip()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateReq(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int = 10

class CreateLicense(BaseModel):
    days: int = 30
    max_requests: int = 1000
    owner: str = ""

class UpdateLicense(BaseModel):
    days: int | None = None
    max_requests: int | None = None
    is_active: bool | None = None

def validate_license(license_key, device_id):
    db = load_db()
    for l in db:
        if l["license_key"] == license_key:
            if not l["is_active"]:
                raise HTTPException(403, "License disabled")
            if now() > datetime.fromisoformat(l["expires_at"]):
                raise HTTPException(403, "License expired")
            if l["used_requests"] >= l["max_requests"]:
                raise HTTPException(403, "Limit reached")
            if l["bound_device"] is None:
                l["bound_device"] = device_id
            elif l["bound_device"] != device_id:
                raise HTTPException(403, "License used on another