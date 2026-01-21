import os, json, math, itertools, secrets
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ================== إعدادات ==================
MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
MAX_TOTAL = 200
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

if not ADMIN_SECRET:
    raise RuntimeError("ADMIN_SECRET not set")

# ================== مفاتيح Gemini ==================
keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12)]
keys = [k for k in keys if k]
if not keys:
    raise RuntimeError("No Gemini API keys found")

key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(MODEL)

# ================== قاعدة بيانات بسيطة (JSON) ==================
DB_FILE = "licenses.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ================== أدوات ==================
def safe_json(text: str):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return None

def lang_instruction(lang):
    return "Write in English." if lang == "en" else "اكتب بالعربية الفصحى."

def build_prompt(topic, lang, count):
    return f"""
{lang_instruction(lang)}

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

# ================== FastAPI ==================
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

# ================== التحقق من المفتاح ==================
def validate_license(license_key: str, device_id: str):
    db = load_db()
    for lic in db:
        if lic["license_key"] == license_key:
            if not lic["is_active"]:
                raise HTTPException(403, "License disabled")

            if datetime.utcnow() > datetime.fromisoformat(lic["expires_at"]):
                raise HTTPException(403, "License expired")

            if lic["used_requests"] >= lic["max_requests"]:
                raise HTTPException(403, "License limit reached")

            if lic["bound_device"] is None:
                lic["bound_device"] = device_id
            elif lic["bound_device"] != device_id:
                raise HTTPException(403, "License used on another device")

            lic["used_requests"] += 1
            save_db(db)
            return

    raise HTTPException(403, "Invalid license")

# ================== Endpoint المستخدم ==================
@app.post("/generate/batch")
def generate(
    req: GenerateReq,
    license_key: str = Header(...),
    device_id: str = Header(...)
):
    validate_license(license_key, device_id)

    total = min(req.total_questions, MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    final_questions = []

    for _ in range(batches):
        needed = min(BATCH_SIZE, total - len(final_questions))
        model = get_model()
        prompt = build_prompt(req.topic, req.language, needed)
        res = model.generate_content(prompt)
        data = safe_json(res.text)
        if not data:
            raise HTTPException(500, "Model error")
        final_questions.extend(data["questions"][:needed])

    return {"questions": final_questions}

# ================== Admin Panel ==================

class CreateLicense(BaseModel):
    days: int = 30
    max_requests: int = 1000

def admin_check(key: str):
    if key != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")

@app.post("/admin/create")
def create_license(data: CreateLicense, x_admin_key: str = Header(...)):
    admin_check(x_admin_key)

    db = load_db()
    license_key = "ST-" + secrets.token_hex(6).upper()

    db.append({
        "license_key": license_key,
        "expires_at": (datetime.utcnow() + timedelta(days=data.days)).isoformat(),
        "max_requests": data.max_requests,
        "used_requests": 0,
        "bound_device": None,
        "is_active": True,
        "created_at": datetime.utcnow().isoformat()
    })

    save_db(db)
    return {"license_key": license_key}

@app.get("/admin/licenses")
def list_licenses(x_admin_key: str = Header(...)):
    admin_check(x_admin_key)
    return load_db()

@app.post("/admin/disable/{license_key}")
def disable_license(license_key: str, x_admin_key: str = Header(...)):
    admin_check(x_admin_key)
    db = load_db()
    for lic in db:
        if lic["license_key"] == license_key:
            lic["is_active"] = False
            save_db(db)
            return {"status": "disabled"}
    raise HTTPException(404, "Not found")

@app.post("/admin/reset-device/{license_key}")
def reset_device(license_key: str, x_admin_key: str = Header(...)):
    admin_check(x_admin_key)
    db = load_db()
    for lic in db:
        if lic["license_key"] == license_key:
            lic["bound_device"] = None
            save_db(db)
            return {"status": "reset"}
    raise HTTPException(404, "Not found")