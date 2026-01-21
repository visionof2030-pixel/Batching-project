import os, json, math, itertools, secrets
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pdfplumber
import google.generativeai as genai

MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
MAX_TOTAL = 200
MAX_TEXT_CHARS = 10000  # زيادة لقراءة PDF أفضل

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

class QuestionTypes(BaseModel):
    multiple_choice: bool = True
    true_false: bool = False
    fill_blank: bool = False

class GenerateReq(BaseModel):
    topic: str
    total_questions: int = 10
    question_types: QuestionTypes

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

def build_prompt(topic: str, count: int, q_types: QuestionTypes):
    type_instructions = []
    
    if q_types.multiple_choice:
        type_instructions.append("اختيار من متعدد مع 4 خيارات وشروحات مفصلة")
    if q_types.true_false:
        type_instructions.append("صح/خطأ مع شرح مفصل")
    if q_types.fill_blank:
        type_instructions.append("أكمل الفراغ مع شرح مفصل")

    types_str = "، ".join(type_instructions)
    
    return f"""
أنت نظام توليد أسئلة تعليمية ذكي.
أنشئ {count} سؤالاً بأنواع: {types_str}

الموضوع: {topic}

المتطلبات:
1. دقة عالية في المحتوى
2. شروحات تعليمية مفصلة
3. خيارات متقاربة وذات معنى
4. إجابات محددة وواضحة

صيغة JSON المطلوبة:
{{
  "questions": [
    {{
      "type": "multiple_choice",
      "q": "السؤال",
      "options": ["خيار1", "خيار2", "خيار3", "خيار4"],
      "answer": 0,
      "correct_explanation": "شرح مفصّل للإجابة الصحيحة",
      "wrong_explanations": ["شرح1", "شرح2", "شرح3", "شرح4"]
    }}
  ]
}}
"""

def build_prompt_from_text(text: str, count: int, q_types: QuestionTypes):
    type_instructions = []
    
    if q_types.multiple_choice:
        type_instructions.append("اختيار من متعدد")
    if q_types.true_false:
        type_instructions.append("صح/خطأ")
    if q_types.fill_blank:
        type_instructions.append("أكمل الفراغ")

    types_str = "، ".join(type_instructions)
    
    return f"""
أنت نظام توليد أسئلة تعليمية من النصوص.
النص المصدر:
{text[:5000]}

أنشئ {count} سؤالاً بأنواع: {types_str}

تعليمات مهمة:
1. جميع الأسئلة منبثقة من النص فقط
2. تنوع في أنواع الأسئلة حسب الطلب
3. شروحات مفصلة وتعليمية
4. خيارات منطقية وذات صلة بالنص
5. إجابات دقيقة من النص

صيغة JSON النهائية:
{{
  "questions": [
    {{
      "type": "multiple_choice",
      "q": "سؤال من النص",
      "options": ["خيار1", "خيار2", "خيار3", "خيار4"],
      "answer": 0,
      "correct_explanation": "شرح مفصل",
      "wrong_explanations": ["شرح1", "شرح2", "شرح3", "شرح4"]
    }}
  ]
}}
"""

def extract_text_from_pdf(file):
    """استخراج نص من PDF بدقة عالية"""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                # استخراج النص مع الحفاظ على التنسيق
                page_text = page.extract_text()
                if page_text:
                    text += f"--- صفحة {i+1} ---\n{page_text}\n\n"
                
                # محاولة استخراج الجداول إذا وجدت
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        table_text = "\n".join(["\t".join([str(cell) if cell else "" for cell in row]) for row in table])
                        text += f"[جدول]\n{table_text}\n\n"
        
        # تنظيف النص
        text = clean_extracted_text(text)
        
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "\n...[النص مقصوص بسبب الطول]"
            
        return text
        
    except Exception as e:
        print(f"خطأ في استخراج PDF: {e}")
        raise HTTPException(500, f"خطأ في قراءة ملف PDF: {str(e)}")

def clean_extracted_text(text):
    """تنظيف النص المستخرج"""
    import re
    
    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text)
    
    # إزالة الأسطر الفارغة المتعددة
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # إصلاح الكلمات المقطوعة
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    
    return text.strip()

def extract_text_from_image_ai(file):
    image = Image.open(file).convert("RGB")
    model = get_model()
    prompt = "استخرج النص الموجود في الصورة بدقة وأعد النص فقط."
    res = model.generate_content([prompt, image])
    text = res.text.strip()
    
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    
    return text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    if not (req.question_types.multiple_choice or 
            req.question_types.true_false or 
            req.question_types.fill_blank):
        raise HTTPException(400, "يجب تحديد نوع سؤال واحد على الأقل")

    total = min(req.total_questions, MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    out = []

    for _ in range(batches):
        need = min(BATCH_SIZE, total - len(out))
        model = get_model()
        prompt = build_prompt(req.topic, need, req.question_types)
        
        try:
            res = model.generate_content(prompt)
            data = safe_json(res.text)
            
            if not data or "questions" not in data:
                try:
                    cleaned = res.text
                    cleaned = cleaned.replace("'", '"')
                    cleaned = cleaned.replace("True", "true").replace("False", "false")
                    start = cleaned.find('{')
                    end = cleaned.rfind('}') + 1
                    if start != -1 and end != -1:
                        data = json.loads(cleaned[start:end])
                    else:
                        raise HTTPException(500, "تعذر تحويل الإجابة إلى JSON")
                except Exception as e:
                    print(f"JSON Parse Error: {e}")
                    raise HTTPException(500, "خطأ في معالجة الإجابة من النموذج")
            
            # إضافة الحقول الافتراضية
            for q in data.get("questions", []):
                q_type = q.get("type")
                
                if q_type == "multiple_choice":
                    if "correct_explanation" not in q:
                        q["correct_explanation"] = "تفسير مفصل للإجابة الصحيحة."
                    if "wrong_explanations" not in q:
                        q["wrong_explanations"] = ["تفسير غير متوفر"] * len(q.get("options", []))
                
                elif q_type == "true_false":
                    if "correct_explanation" not in q:
                        q["correct_explanation"] = f"العبارة {'صحيحة' if q.get('answer') else 'خاطئة'}."
                    if "correction" not in q:
                        q["correction"] = ""
                
                elif q_type == "fill_blank":
                    if "correct_explanation" not in q:
                        q["correct_explanation"] = "تفسير مفصل للإجابة الصحيحة."
                    if "alternatives" not in q:
                        q["alternatives"] = []
            
            out.extend(data["questions"][:need])
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(500, f"خطأ في توليد الأسئلة: {str(e)}")

    return {"questions": out[:total]}

@app.post("/generate/from-image")
def generate_from_image(
    total_questions: int = 10,
    multiple_choice: bool = True,
    true_false: bool = False,
    fill_blank: bool = False,
    file: UploadFile = File(...),
    license_key: str = Header(...),
    device_id: str = Header(...)
):
    validate_license(license_key, device_id)

    if not (multiple_choice or true_false or fill_blank):
        raise HTTPException(400, "يجب تحديد نوع سؤال واحد على الأقل")

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "نوع الصورة غير مدعوم")

    text = extract_text_from_image_ai(file.file)
    if not text or len(text) < 30:
        raise HTTPException(400, "فشل استخراج النص من الصورة")

    total = min(total_questions, MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    out = []

    question_types = QuestionTypes(
        multiple_choice=multiple_choice,
        true_false=true_false,
        fill_blank=fill_blank
    )

    for _ in range(batches):
        need = min(BATCH_SIZE, total - len(out))
        model = get_model()
        prompt = build_prompt_from_text(text, need, question_types)
        res = model.generate_content(prompt)
        data = safe_json(res.text)
        
        if not data or "questions" not in data:
            try:
                cleaned = res.text.replace("'", '"')
                cleaned = cleaned.replace("True", "true").replace("False", "false")
                data = json.loads(cleaned)
            except:
                raise HTTPException(500, "خطأ في توليد الأسئلة")
        
        # إضافة الحقول الافتراضية
        for q in data.get("questions", []):
            q_type = q.get("type")
            
            if q_type == "multiple_choice":
                if "correct_explanation" not in q:
                    q["correct_explanation"] = "تفسير مفصل للإجابة الصحيحة."
                if "wrong_explanations" not in q:
                    q["wrong_explanations"] = ["تفسير غير متوفر"] * len(q.get("options", []))
            
            elif q_type == "true_false":
                if "correct_explanation" not in q:
                    q["correct_explanation"] = f"العبارة {'صحيحة' if q.get('answer') else 'خاطئة'}."
                if "correction" not in q:
                    q["correction"] = ""
            
            elif q_type == "fill_blank":
                if "correct_explanation" not in q:
                    q["correct_explanation"] = "تفسير مفصل للإجابة الصحيحة."
                if "alternatives" not in q:
                    q["alternatives"] = []
        
        out.extend(data["questions"][:need])

    return {"questions": out[:total]}

@app.post("/generate/from-file")
def generate_from_pdf(
    total_questions: int = 10,
    multiple_choice: bool = True,
    true_false: bool = False,
    fill_blank: bool = False,
    file: UploadFile = File(...),
    license_key: str = Header(...),
    device_id: str = Header(...)
):
    validate_license(license_key, device_id)

    if not (multiple_choice or true_false or fill_blank):
        raise HTTPException(400, "يجب تحديد نوع سؤال واحد على الأقل")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "نوع الملف غير مدعوم - يجب أن يكون PDF")

    try:
        text = extract_text_from_pdf(file.file)
        if not text or len(text) < 50:
            raise HTTPException(400, "فشل استخراج النص من الملف أو الملف فارغ")
        
        # حفظ النص المستخرج مؤقتاً للتصحيح
        print(f"تم استخراج {len(text)} حرفاً من PDF")
        
    except Exception as e:
        raise HTTPException(500, f"خطأ في معالجة ملف PDF: {str(e)}")

    total = min(total_questions, MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    out = []

    question_types = QuestionTypes(
        multiple_choice=multiple_choice,
        true_false=true_false,
        fill_blank=fill_blank
    )

    for _ in range(batches):
        need = min(BATCH_SIZE, total - len(out))
        model = get_model()
        prompt = build_prompt_from_text(text, need, question_types)
        
        try:
            res = model.generate_content(prompt)
            data = safe_json(res.text)
            
            if not data or "questions" not in data:
                try:
                    cleaned = res.text.replace("'", '"')
                    cleaned = cleaned.replace("True", "true").replace("False", "false")
                    data = json.loads(cleaned)
                except Exception as e:
                    print(f"JSON Error: {e}")
                    raise HTTPException(500, "خطأ في معالجة إجابة النموذج")
            
            # إضافة الحقول الافتراضية
            for q in data.get("questions", []):
                q_type = q.get("type")
                
                if q_type == "multiple_choice":
                    if "correct_explanation" not in q:
                        q["correct_explanation"] = "تفسير مفصل للإجابة الصحيحة."
                    if "wrong_explanations" not in q:
                        q["wrong_explanations"] = ["تفسير غير متوفر"] * len(q.get("options", []))
                
                elif q_type == "true_false":
                    if "correct_explanation" not in q:
                        q["correct_explanation"] = f"العبارة {'صحيحة' if q.get('answer') else 'خاطئة'}."
                    if "correction" not in q:
                        q["correction"] = ""
                
                elif q_type == "fill_blank":
                    if "correct_explanation" not in q:
                        q["correct_explanation"] = "تفسير مفصل للإجابة الصحيحة."
                    if "alternatives" not in q:
                        q["alternatives"] = []
            
            out.extend(data["questions"][:need])
            
        except Exception as e:
            raise HTTPException(500, f"خطأ في توليد الأسئلة من PDF: {str(e)}")

    return {"questions": out[:total]}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)