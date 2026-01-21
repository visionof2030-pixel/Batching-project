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

def build_prompt(topic, count):
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

def build_prompt_from_text(text, count):
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
                raise HTTPException(403, "License used on another device")
            l["used_requests"] += 1
            l["last_request_at"] = now().isoformat()
            save_db(db)
            return
    raise HTTPException(403, "Invalid license")

def admin_check(key):
    if key != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/generate/batch")
def generate_manual(
    req: GenerateReq,
    license_key: str = Header(...),
    device_id: str = Header(...)
):
    validate_license(license_key, device_id)

    total = min(req.total_questions, MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    out = []

    for _ in range(batches):
        need = min(BATCH_SIZE, total - len(out))
        model = get_model()
        res = model.generate_content(build_prompt(req.topic, need))
        data = safe_json(res.text)
        if not data:
            raise HTTPException(500, "Model error")
        out.extend(data["questions"][:need])

    return {"questions": out}

@app.post("/generate/from-image")
def generate_from_image(
    total_questions: int = 10,
    file: UploadFile = File(...),
    license_key: str = Header(...),
    device_id: str = Header(...)
):
    validate_license(license_key, device_id)

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "Unsupported image type")

    text = extract_text_from_image_ai(file.file)
    if not text or len(text) < 30:
        raise HTTPException(400, "Text extraction failed")

    total = min(total_questions, MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    out = []

    for _ in range(batches):
        need = min(BATCH_SIZE, total - len(out))
        model = get_model()
        res = model.generate_content(build_prompt_from_text(text, need))
        data = safe_json(res.text)
        if not data:
            raise HTTPException(500, "Model error")
        out.extend(data["questions"][:need])

    return {"questions": out}

@app.post("/generate/from-file")
def generate_from_pdf(
    total_questions: int = 10,
    file: UploadFile = File(...),
    license_key: str = Header(...),
    device_id: str = Header(...)
):
    validate_license(license_key, device_id)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Unsupported file type")

    text = extract_text_from_pdf(file.file)
    if not text or len(text) < 50:
        raise HTTPException(400, "Text extraction failed")

    total = min(total_questions, MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    out = []

    for _ in range(batches):
        need = min(BATCH_SIZE, total - len(out))
        model = get_model()
        res = model.generate_content(build_prompt_from_text(text, need))
        data = safe_json(res.text)
        if not data:
            raise HTTPException(500, "Model error")
        out.extend(data["questions"][:need])

    return {"questions": out}

@app.post("/admin/create")
def admin_create(data: CreateLicense, x_admin_key: str = Header(...)):
    admin_check(x_admin_key)
    db = load_db()
    key = "ST-" + secrets.token_hex(6).upper()
    db.append({
        "license_key": key,
        "expires_at": (now() + timedelta(days=data.days)).isoformat(),
        "max_requests": data.max_requests,
        "used_requests": 0,
        "bound_device": None,
        "is_active": True,
        "owner": data.owner,
        "created_at": now().isoformat(),
        "last_request_at": None
    })
    save_db(db)
    return {"license_key": key}

@app.get("/admin/licenses")
def admin_list(x_admin_key: str = Header(...)):
    admin_check(x_admin_key)
    return load_db()

@app.put("/admin/update/{key}")
def admin_update(key: str, data: UpdateLicense, x_admin_key: str = Header(...)):
    admin_check(x_admin_key)
    db = load_db()
    for l in db:
        if l["license_key"] == key:
            if data.days is not None:
                l["expires_at"] = (now() + timedelta(days=data.days)).isoformat()
            if data.max_requests is not None:
                l["max_requests"] = data.max_requests
            if data.is_active is not None:
                l["is_active"] = data.is_active
            save_db(db)
            return {"status": "updated"}
    raise HTTPException(404, "Not found")

@app.post("/admin/reset-device/{key}")
def admin_reset(key: str, x_admin_key: str = Header(...)):
    admin_check(x_admin_key)
    db = load_db()
    for l in db:
        if l["license_key"] == key:
            l["bound_device"] = None
            save_db(db)
            return {"status": "reset"}
    raise HTTPException(404, "Not found")

@app.delete("/admin/delete/{key}")
def admin_delete(key: str, x_admin_key: str = Header(...)):
    admin_check(x_admin_key)
    db = [l for l in load_db() if l["license_key"] != key]
    save_db(db)
    return {"status": "deleted"}