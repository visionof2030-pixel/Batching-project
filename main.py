# =========================
# main.py  (BACKEND كامل)
# =========================
import os, json, math, itertools, secrets, re
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pdfplumber
import google.generativeai as genai

MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
MAX_TOTAL = 200
MAX_TEXT_CHARS = 6000

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

def clean_text(text: str):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u0600-\u06FFa-zA-Z0-9.,؟!()\n ]', '', text)
    return text.strip()

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    text = clean_text(text)
    return text[:MAX_TEXT_CHARS]

def extract_text_from_image_ai(file):
    image = Image.open(file).convert("RGB")
    model = get_model()
    prompt = "استخرج النص الموجود في الصورة بدقة وأعد النص فقط."
    res = model.generate_content([prompt, image])
    text = clean_text(res.text)
    return text[:MAX_TEXT_CHARS]

def build_prompt_from_text(text, count, types):
    return f"""
أنت نظام تعليمي محترف.
أنشئ أسئلة تعليمية دقيقة جدًا بناءً على النص فقط.

أنواع الأسئلة المسموحة:
{", ".join(types)}

قواعد صارمة:
- لا تضف أي معلومة غير موجودة في النص
- الصياغة العربية سليمة وواضحة
- في (أكمل الفراغ):
  - الفراغ قصير جدًا
  - الجواب مأخوذ حرفيًا من النص
  - جواب واحد فقط صحيح

النص:
{text}

عدد الأسئلة: {count}

أعد الناتج بصيغة JSON فقط:
{{
 "questions":[
  {{
   "type":"mcq | tf | fill",
   "q":"",
   "options":["","","",""],
   "answer":0 | true | "النص الناقص",
   "explanations":[""]
  }}
 ]
}}
"""

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateReq(BaseModel):
    topic: str
    total_questions: int = 10
    types: list[str] = ["mcq"]

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
                raise HTTPException(403, "License used on another device")
            l["used_requests"] += 1
            save_db(db)
            return
    raise HTTPException(403, "Invalid license")

@app.post("/generate/batch")
def generate_manual(req: GenerateReq,
                    license_key: str = Header(...),
                    device_id: str = Header(...)):
    validate_license(license_key, device_id)
    model = get_model()
    res = model.generate_content(build_prompt_from_text(req.topic, req.total_questions, req.types))
    data = safe_json(res.text)
    if not data:
        raise HTTPException(500, "Model error")
    return data

@app.post("/generate/from-image")
def generate_from_image(total_questions: int = Form(...),
                        types: str = Form(...),
                        file: UploadFile = File(...),
                        license_key: str = Header(...),
                        device_id: str = Header(...)):
    validate_license(license_key, device_id)
    text = extract_text_from_image_ai(file.file)
    model = get_model()
    res = model.generate_content(build_prompt_from_text(text, total_questions, json.loads(types)))
    data = safe_json(res.text)
    if not data:
        raise HTTPException(500, "Model error")
    return data

@app.post("/generate/from-file")
def generate_from_pdf(total_questions: int = Form(...),
                      types: str = Form(...),
                      file: UploadFile = File(...),
                      license_key: str = Header(...),
                      device_id: str = Header(...)):
    validate_license(license_key, device_id)
    text = extract_text_from_pdf(file.file)
    model = get_model()
    res = model.generate_content(build_prompt_from_text(text, total_questions, json.loads(types)))
    data = safe_json(res.text)
    if not data:
        raise HTTPException(500, "Model error")
    return data