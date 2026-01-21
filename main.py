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

class QuestionTypes(BaseModel):
    multiple_choice: bool = True
    true_false: bool = False
    fill_blank: bool = False

class LanguageConfig(BaseModel):
    lang: str = "ar"  # ar, en

class GenerateReq(BaseModel):
    topic: str
    total_questions: int = 10
    question_types: QuestionTypes
    language: LanguageConfig = LanguageConfig()

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

def build_prompt(topic: str, count: int, q_types: QuestionTypes, lang: str = "ar"):
    if lang == "en":
        return build_prompt_en(topic, count, q_types)
    else:
        return build_prompt_ar(topic, count, q_types)

def build_prompt_ar(topic: str, count: int, q_types: QuestionTypes):
    type_instructions = []
    
    if q_types.multiple_choice:
        type_instructions.append("""
اختيار من متعدد:
- سؤال مع 4 خيارات
- إجابة واحدة صحيحة فقط
- قدم شرحاً مفصلاً للإجابة الصحيحة (لماذا هذه الإجابة صحيحة؟)
- قدم شرحاً مفصلاً لكل خيار خاطئ (لماذا هذا الخيار غير صحيح؟)
- تأكد من أن الخيارات متقاربة ومربكة بعض الشيء
- استخدم نبرة تعليمية في الشروحات
""")
    
    if q_types.true_false:
        type_instructions.append("""
صح/خطأ:
- عبارة تكون إما صحيحة أو خاطئة
- قدم شرحاً مفصلاً للإجابة (لماذا العبارة صحيحة أو خاطئة؟)
- إذا كانت العبارة خاطئة، صححها واشرح السبب
- قدم أمثلة توضيحية إذا لزم الأمر
""")
    
    if q_types.fill_blank:
        type_instructions.append("""
أكمل الفراغ:
- جملة ناقصة بكلمة أو عبارة
- استخدم _____ أو (...) للإشارة إلى الفراغ
- قدم الإجابة الصحيحة
- قدم شرحاً مفصلاً لماذا هذه الإجابة صحيحة
- اذكر بدائل مقبولة إذا وجدت
- قدم أمثلة على استخدام الكلمة في سياقات مختلفة
""")

    types_str = "\n".join(type_instructions)
    
    return f"""
أنت نظام توليد أسئلة تعليمية ذكي.
مهمتك إنشاء أسئلة اختبارية عالية الجودة مع شروحات مفصلة.

{types_str}

عدد الأسئلة المطلوب: {count}
الموضوع: {topic}

**متطلبات إضافية مهمة:**
1. كل سؤال يجب أن يكون واضحاً ودقيقاً
2. الشروحات يجب أن تكون تعليمية ومفصلة
3. للخيارات الخاطئة: اشرح سبب الخطأ بشكل واضح
4. استخدم لغة عربية سليمة وفصيحة
5. قدم أمثلة توضيحية عندما يكون ذلك مناسباً
6. تأكد من صحة المعلومات المقدمة

**صيغة JSON المطلوبة بدقة:**
{{
  "questions": [
    {{
      "type": "multiple_choice",
      "q": "نص السؤال هنا",
      "options": ["الخيار الأول", "الخيار الثاني", "الخيار الثالث", "الخيار الرابع"],
      "answer": 0,
      "correct_explanation": "شرح مفصّل للإجابة الصحيحة.",
      "wrong_explanations": [
        "شرح مفصّل لسبب خطأ الخيار الأول.",
        "شرح مفصّل لسبب خطأ الخيار الثاني.",
        "شرح مفصّل لسبب خطأ الخيار الثالث.",
        "شرح مفصّل لسبب خطأ الخيار الرابع."
      ]
    }}
  ]
}}
"""

def build_prompt_en(topic: str, count: int, q_types: QuestionTypes):
    type_instructions = []
    
    if q_types.multiple_choice:
        type_instructions.append("""
Multiple Choice Questions:
- Question with 4 options
- Only one correct answer
- Provide detailed explanation for the correct answer (why is this answer correct?)
- Provide detailed explanation for each wrong option (why is this option incorrect?)
- Ensure options are similar and somewhat confusing
- Use educational tone in explanations
""")
    
    if q_types.true_false:
        type_instructions.append("""
True/False Questions:
- A statement that is either true or false
- Provide detailed explanation for the answer (why the statement is true or false)
- If the statement is false, correct it and explain why
- Provide illustrative examples if needed
""")
    
    if q_types.fill_blank:
        type_instructions.append("""
Fill in the Blank Questions:
- A sentence with a missing word or phrase
- Use _____ or (...) to indicate the blank
- Provide the correct answer
- Provide detailed explanation of why this answer is correct
- Mention acceptable alternatives if any
- Provide examples of the word in different contexts
""")

    types_str = "\n".join(type_instructions)
    
    return f"""
You are an intelligent educational question generation system.
Your task is to create high-quality test questions with detailed explanations.

{types_str}

Number of questions required: {count}
Topic: {topic}

**Important Additional Requirements:**
1. Each question must be clear and precise
2. Explanations should be educational and detailed
3. For wrong options: explain the reason clearly
4. Use correct English language
5. Provide illustrative examples when appropriate
6. Ensure the information provided is accurate

**Required JSON Format Exactly:**
{{
  "questions": [
    {{
      "type": "multiple_choice",
      "q": "Question text here",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "answer": 0,
      "correct_explanation": "Detailed explanation of the correct answer.",
      "wrong_explanations": [
        "Detailed explanation of why option 1 is wrong.",
        "Detailed explanation of why option 2 is wrong.",
        "Detailed explanation of why option 3 is wrong.",
        "Detailed explanation of why option 4 is wrong."
      ]
    }},
    {{
      "type": "true_false",
      "q": "Statement here",
      "answer": true,
      "correct_explanation": "Detailed explanation of the answer.",
      "correction": "Correction if applicable"
    }},
    {{
      "type": "fill_blank",
      "q": "Sentence with _____ here",
      "answer": "Correct answer",
      "correct_explanation": "Detailed explanation of the answer.",
      "alternatives": ["alternative1", "alternative2"]
    }}
  ]
}}

**IMPORTANT: All questions and explanations MUST be in English only.**
"""

def build_prompt_from_text(text: str, count: int, q_types: QuestionTypes, lang: str = "ar"):
    if lang == "en":
        return build_prompt_from_text_en(text, count, q_types)
    else:
        return build_prompt_from_text_ar(text, count, q_types)

def build_prompt_from_text_ar(text: str, count: int, q_types: QuestionTypes):
    type_instructions = []
    
    if q_types.multiple_choice:
        type_instructions.append("اختيار من متعدد مع شروحات مفصلة")
    if q_types.true_false:
        type_instructions.append("صح/خطأ مع شروحات مفصلة")
    if q_types.fill_blank:
        type_instructions.append("أكمل الفراغ مع شروحات مفصلة")

    types_str = "، ".join(type_instructions)
    
    return f"""
أنت نظام توليد أسئلة تعليمية ذكي.
مهمتك إنشاء أسئلة اختبارية عالية الجودة مع شروحات مفصلة.

أنواع الأسئلة المطلوبة: {types_str}

النص المصدر:
{text[:4000]}

عدد الأسئلة المطلوب: {count}

**تعليمات مهمة:**
1. جميع الأسئلة يجب أن تكون مبنية على النص المصدر فقط
2. قدم شروحات مفصلة وتعليمية لكل سؤال
3. للخيارات الخاطئة: اشرح سبب الخطأ بوضوح
4. استخرج المعلومات الرئيسية من النص
5. تأكد من دقة المعلومات بناءً على النص

**تنسيق JSON النهائي:**
{{
  "questions": [
    {{
      "type": "multiple_choice",
      "q": "سؤال مبنى على النص",
      "options": ["خيار1", "خيار2", "خيار3", "خيار4"],
      "answer": 0,
      "correct_explanation": "شرح مفصّل",
      "wrong_explanations": ["شرح1", "شرح2", "شرح3", "شرح4"]
    }},
    {{
      "type": "true_false",
      "q": "عبارة عن النص",
      "answer": true,
      "correct_explanation": "شرح مفصّل",
      "correction": "التصحيح إذا لزم"
    }},
    {{
      "type": "fill_blank",
      "q": "جملة عن النص مع _____",
      "answer": "إجابة",
      "correct_explanation": "شرح مفصّل",
      "alternatives": []
    }}
  ]
}}
"""

def build_prompt_from_text_en(text: str, count: int, q_types: QuestionTypes):
    type_instructions = []
    
    if q_types.multiple_choice:
        type_instructions.append("Multiple choice with detailed explanations")
    if q_types.true_false:
        type_instructions.append("True/False with detailed explanations")
    if q_types.fill_blank:
        type_instructions.append("Fill in the blank with detailed explanations")

    types_str = ", ".join(type_instructions)
    
    return f"""
You are an intelligent educational question generation system.
Your task is to create high-quality test questions with detailed explanations.

Required question types: {types_str}

Source text:
{text[:4000]}

Number of questions required: {count}

**Important Instructions:**
1. ALL questions must be based ONLY on the source text
2. Provide detailed educational explanations for each question
3. For wrong options: clearly explain why they're wrong
4. Extract key information from the text
5. Ensure accuracy based on the text
6. ALL questions and explanations MUST be in ENGLISH only

**Final JSON Format:**
{{
  "questions": [
    {{
      "type": "multiple_choice",
      "q": "Question based on text",
      "options": ["option1", "option2", "option3", "option4"],
      "answer": 0,
      "correct_explanation": "Detailed explanation",
      "wrong_explanations": ["expl1", "expl2", "expl3", "expl4"]
    }},
    {{
      "type": "true_false",
      "q": "Statement about the text",
      "answer": true,
      "correct_explanation": "Detailed explanation",
      "correction": "Correction if needed"
    }},
    {{
      "type": "fill_blank",
      "q": "Sentence about text with _____",
      "answer": "answer",
      "correct_explanation": "Detailed explanation",
      "alternatives": []
    }}
  ]
}}
"""

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return text

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
        prompt = build_prompt(req.topic, need, req.question_types, req.language.lang)
        
        try:
            res = model.generate_content(prompt)
            data = safe_json(res.text)
            
            if not data or "questions" not in data:
                # محاولة إصلاح JSON
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
            
            # التحقق من صحة الهيكل وإضافة الحقول الافتراضية
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
    language: str = "ar",
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
        prompt = build_prompt_from_text(text, need, question_types, language)
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

@app.post("/generate/from-pdf")
def generate_from_pdf(
    total_questions: int = 10,
    multiple_choice: bool = True,
    true_false: bool = False,
    fill_blank: bool = False,
    language: str = "ar",
    file: UploadFile = File(...),
    license_key: str = Header(...),
    device_id: str = Header(...)
):
    validate_license(license_key, device_id)

    if not (multiple_choice or true_false or fill_blank):
        raise HTTPException(400, "يجب تحديد نوع سؤال واحد على الأقل")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "نوع الملف غير مدعوم")

    text = extract_text_from_pdf(file.file)
    if not text or len(text) < 50:
        raise HTTPException(400, "فشل استخراج النص من الملف")

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
        prompt = build_prompt_from_text(text, need, question_types, language)
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

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)