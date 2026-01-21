limport os, json, math, itertools, secrets
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ================== إعدادات عامة ==================
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

# ================== قاعدة البيانات (JSON) ==================
DB_FILE = "licenses.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ================== أدوات مساعدة ==================
def now():
    return datetime.utcnow()

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

# ================== Models ==================
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

# ================== License Logic ==================
def validate_license(license_key: str, device_id: str):
    db = load_db()
    for lic in db:
        if lic["license_key"] == license_key:
            if not lic["is_active"]:
                raise HTTPException(403, "License disabled")

            if now() > datetime.fromisoformat(lic["expires_at"]):
                raise HTTPException(403, "License expired")

            if lic["used_requests"] >= lic["max_requests"]:
                raise HTTPException(403, "License limit reached")

            if lic["bound_device"] is None:
                lic["bound_device"] = device_id
            elif lic["bound_device"] != device_id:
                raise HTTPException(403, "License used on another device")

            lic["used_requests"] += 1
            lic["last_request_at"] = now().isoformat()
            save_db(db)
            return

    raise HTTPException(403, "Invalid license")

def admin_check(key: str):
    if key != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")

# ================== User Endpoint ==================
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

# ================== Admin API ==================

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

@app.put("/admin/update/{license_key}")
def admin_update(license_key: str, data: UpdateLicense, x_admin_key: str = Header(...)):
    admin_check(x_admin_key)
    db = load_db()
    for lic in db:
        if lic["license_key"] == license_key:
            if data.days is not None:
                lic["expires_at"] = (now() + timedelta(days=data.days)).isoformat()
            if data.max_requests is not None:
                lic["max_requests"] = data.max_requests
            if data.is_active is not None:
                lic["is_active"] = data.is_active
            save_db(db)
            return {"status": "updated"}
    raise HTTPException(404, "Not found")

@app.post("/admin/reset-device/{license_key}")
def admin_reset_device(license_key: str, x_admin_key: str = Header(...)):
    admin_check(x_admin_key)
    db = load_db()
    for lic in db:
        if lic["license_key"] == license_key:
            lic["bound_device"] = None
            save_db(db)
            return {"status": "device reset"}
    raise HTTPException(404, "Not found")

@app.delete("/admin/delete/{license_key}")
def admin_delete(license_key: str, x_admin_key: str = Header(...)):
    admin_check(x_admin_key)
    db = load_db()
    db = [l for l in db if l["license_key"] != license_key]
    save_db(db)
    return {"status": "deleted"}