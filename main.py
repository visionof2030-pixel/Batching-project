import os
import json
import math
import random
import re
import secrets
import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict, Tuple
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import pdfplumber
import google.generativeai as genai

# إعداد logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ثوابت التطبيق
MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 8  # تقليل الحجم لتحسين الاستجابة
MAX_TOTAL = 200
MAX_TEXT_CHARS = 5000
MAX_RETRIES = 3
RETRY_DELAY = 1  # ثانية

# التحقق من المتغيرات البيئية
ADMIN_SECRET = os.getenv("ADMIN_SECRET")
if not ADMIN_SECRET:
    raise RuntimeError("ADMIN_SECRET not set")

# تحميل مفاتيح Gemini
keys: List[str] = []
for i in range(1, 12):
    key = os.getenv(f"GEMINI_KEY_{i}")
    if key and key.strip():
        keys.append(key.strip())
        logger.info(f"تم تحميل المفتاح GEMINI_KEY_{i}")

if not keys:
    raise RuntimeError("No Gemini API keys found")

logger.info(f"تم تحميل {len(keys)} مفتاح API")

# إدارة قاعدة البيانات
DB_FILE = "licenses.json"

def load_db() -> List[Dict]:
    """تحميل قاعدة بيانات الترخيصات"""
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"خطأ في تحميل قاعدة البيانات: {e}")
        return []

def save_db(data: List[Dict]):
    """حفظ قاعدة بيانات الترخيصات"""
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"خطأ في حفظ قاعدة البيانات: {e}")
        raise

def get_current_time() -> datetime:
    """الحصول على الوقت الحالي"""
    return datetime.utcnow()

# نماذج البيانات
class QuestionTypes(BaseModel):
    multiple_choice: bool = Field(default=True, description="أسئلة اختيار من متعدد")
    true_false: bool = Field(default=False, description="أسئلة صح/خطأ")
    fill_blank: bool = Field(default=False, description="أسئلة أكمل الفراغ")

class LanguageConfig(BaseModel):
    lang: str = Field(default="ar", description="لغة التوليد (ar/en)")

class GenerateReq(BaseModel):
    topic: str = Field(..., description="موضوع الأسئلة")
    total_questions: int = Field(default=10, ge=1, le=50, description="عدد الأسئلة")
    question_types: QuestionTypes = Field(default_factory=QuestionTypes)
    language: LanguageConfig = Field(default_factory=LanguageConfig)

class CreateLicense(BaseModel):
    days: int = Field(default=30, ge=1, le=365, description="عدد أيام الترخيص")
    max_requests: int = Field(default=1000, ge=1, le=10000, description="الحد الأقصى للطلبات")
    owner: str = Field(default="", description="مالك الترخيص")

class UpdateLicense(BaseModel):
    days: Optional[int] = Field(None, ge=1, le=365)
    max_requests: Optional[int] = Field(None, ge=1, le=10000)
    is_active: Optional[bool] = None

# إدارة مفاتيح API
def get_model():
    """الحصول على نموذج Gemini مع مفتاح عشوائي"""
    key = random.choice(keys)
    genai.configure(api_key=key)
    return genai.GenerativeModel(MODEL)

def extract_json_from_text(text: str) -> Optional[Any]:
    """
    استخراج JSON من النص مع معالجة شاملة للأخطاء الشائعة
    """
    if not text:
        return None
    
    # تنظيف النص
    text = text.strip()
    logger.debug(f"محاولة استخراج JSON من نص بطول: {len(text)}")
    
    # محاولة 1: تحليل JSON مباشرة
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"المحاولة 1 فشلت: {e}")
        pass
    
    # محاولة 2: البحث عن كتلة JSON في النص
    patterns = [
        r'```json\s*(.*?)\s*```',  # كتلة كود JSON
        r'```\s*(.*?)\s*```',      # أي كتلة كود
        r'\{.*\}',                  # بحث عن أقواس معقوفة
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                cleaned = match.strip()
                
                # إصلاح المشاكل الشائعة
                cleaned = cleaned.replace("'", '"')
                cleaned = cleaned.replace("True", "true").replace("False", "false")
                cleaned = cleaned.replace("None", "null")
                
                # إصلاح الفواصل الزائدة
                cleaned = re.sub(r',\s*}', '}', cleaned)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                
                # إصلاح الاقتباسات غير المغلقة
                cleaned = re.sub(r'(?<!\\)"', '"', cleaned)
                
                # تحويل أرقام الخيارات إلى أرقام صحيحة
                cleaned = re.sub(r'"answer":\s*"(\d+)"', r'"answer": \1', cleaned)
                
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.debug(f"فشل في نمط {pattern}: {e}")
                continue
    
    # محاولة 3: استخراج يدوي باستخدام regex
    try:
        # إزالة أي نص غير JSON
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start >= 0 and end > start:
            json_str = text[start:end]
            
            # تنظيف شامل
            json_str = json_str.replace('\n', ' ').replace('\r', '')
            
            # إضافة اقتباسات للمفاتيح إذا كانت مفقودة
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            
            # إصلاح باقي المشاكل
            json_str = json_str.replace("'", '"')
            json_str = json_str.replace("True", "true").replace("False", "false")
            json_str = json_str.replace("None", "null")
            
            # إصلاح الفواصل
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            return json.loads(json_str)
    except Exception as e:
        logger.error(f"فشل المحاولة 3: {e}")
    
    logger.warning("لم يتم استخراج JSON من النص")
    return None

# دوال التحقق من الصحة
def validate_question_structure(question: Dict, q_types: QuestionTypes) -> bool:
    """
    التحقق من هيكل السؤال المولد
    """
    if not isinstance(question, dict):
        return False
    
    q_type = question.get("type")
    if q_type not in ["multiple_choice", "true_false", "fill_blank"]:
        return False
    
    # التحقق حسب النوع
    if q_type == "multiple_choice":
        required = ["q", "options", "answer"]
        if not all(key in question for key in required):
            return False
        
        if not isinstance(question["options"], list) or len(question["options"]) != 4:
            return False
        
        if not isinstance(question["answer"], int) or question["answer"] not in [0, 1, 2, 3]:
            return False
        
        # إضافة الحقول المفقودة
        if "correct_explanation" not in question:
            question["correct_explanation"] = "شرح مفصل للإجابة الصحيحة."
        
        if "wrong_explanations" not in question:
            question["wrong_explanations"] = ["تفسير غير متوفر"] * 4
        elif not isinstance(question["wrong_explanations"], list) or len(question["wrong_explanations"]) != 4:
            question["wrong_explanation"] = ["تفسير غير متوفر"] * 4
    
    elif q_type == "true_false":
        required = ["q", "answer"]
        if not all(key in question for key in required):
            return False
        
        if not isinstance(question["answer"], bool):
            return False
        
        if "correct_explanation" not in question:
            question["correct_explanation"] = f"العبارة {'صحيحة' if question['answer'] else 'خاطئة'}."
        
        if "correction" not in question:
            question["correction"] = ""
    
    elif q_type == "fill_blank":
        required = ["q", "answer"]
        if not all(key in question for key in required):
            return False
        
        if "correct_explanation" not in question:
            question["correct_explanation"] = "تفسير مفصل للإجابة الصحيحة."
        
        if "alternatives" not in question:
            question["alternatives"] = []
    
    return True

def validate_license(license_key: str, device_id: str) -> Tuple[List[Dict], int]:
    """
    التحقق من صلاحية الترخيص
    """
    db = load_db()
    for i, license_data in enumerate(db):
        if license_data["license_key"] == license_key:
            if not license_data.get("is_active", True):
                raise HTTPException(status_code=403, detail="الترخيص معطل")
            
            expires_at = datetime.fromisoformat(license_data["expires_at"])
            if get_current_time() > expires_at:
                raise HTTPException(status_code=403, detail="انتهت صلاحية الترخيص")
            
            if license_data["used_requests"] >= license_data["max_requests"]:
                raise HTTPException(status_code=403, detail="تم استنفاذ عدد الطلبات")
            
            if license_data.get("bound_device") is None:
                license_data["bound_device"] = device_id
                save_db(db)  # حفظ عند ربط الجهاز لأول مرة
            elif license_data["bound_device"] != device_id:
                raise HTTPException(status_code=403, detail="الترخيص مستخدم على جهاز آخر")
            
            return db, i
    
    raise HTTPException(status_code=403, detail="ترخيص غير صالح")

def admin_check(key: str):
    """التحقق من صلاحية المسؤول"""
    if key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="غير مصرح")

# دوال بناء الـ Prompts
def build_simplified_prompt_ar(topic: str, count: int, q_types: QuestionTypes) -> str:
    """
    بناء prompt مبسط باللغة العربية
    """
    type_list = []
    if q_types.multiple_choice:
        type_list.append("اختيار من متعدد")
    if q_types.true_false:
        type_list.append("صح/خطأ")
    if q_types.fill_blank:
        type_list.append("أكمل الفراغ")
    
    types_str = " و ".join(type_list)
    
    return f"""
أنت مساعد متخصص في إنشاء أسئلة تعليمية. مهمتك إنشاء {count} سؤالاً تعليمياً عن الموضوع التالي:

**الموضوع:** {topic}

**أنواع الأسئلة المطلوبة:** {types_str}

**التعليمات الهامة:**
1. أعد الإجابة بتنسيق JSON صالح فقط
2. لا تضيف أي نص خارج JSON
3. تأكد من أن الأسئلة متنوعة ومتعلقة بالموضوع
4. استخدم لغة عربية سليمة وفصيحة

**بنية JSON المطلوبة:**
{{
  "questions": [
    {{
      "type": "multiple_choice",
      "q": "نص السؤال هنا",
      "options": ["الخيار الأول", "الخيار الثاني", "الخيار الثالث", "الخيار الرابع"],
      "answer": 0,
      "correct_explanation": "شرح مفصل للإجابة الصحيحة",
      "wrong_explanations": ["شرح الخطأ الأول", "شرح الخطأ الثاني", "شرح الخطأ الثالث", "شرح الخطأ الرابع"]
    }}
  ]
}}

**ملاحظات:**
- الخيارات يجب أن تكون 4 خيارات لكل سؤال اختيار من متعدد
- answer هو رقم (0-3) يشير للخيار الصحيح
- wrong_explanations مصفوفة من 4 عناصر
"""

def build_simplified_prompt_en(topic: str, count: int, q_types: QuestionTypes) -> str:
    """
    Build simplified prompt in English
    """
    type_list = []
    if q_types.multiple_choice:
        type_list.append("multiple choice")
    if q_types.true_false:
        type_list.append("true/false")
    if q_types.fill_blank:
        type_list.append("fill in the blank")
    
    types_str = " and ".join(type_list)
    
    return f"""
You are an expert educational question generator. Your task is to create {count} educational questions about the following topic:

**Topic:** {topic}

**Required question types:** {types_str}

**Important Instructions:**
1. Return your answer in valid JSON format ONLY
2. Do NOT add any text outside the JSON
3. Ensure questions are diverse and relevant to the topic
4. Use correct English language

**Required JSON Structure:**
{{
  "questions": [
    {{
      "type": "multiple_choice",
      "q": "Question text here",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "answer": 0,
      "correct_explanation": "Detailed explanation of correct answer",
      "wrong_explanations": ["Wrong reason 1", "Wrong reason 2", "Wrong reason 3", "Wrong reason 4"]
    }}
  ]
}}

**Notes:**
- Multiple choice questions must have exactly 4 options
- answer is a number (0-3) pointing to correct option
- wrong_explanations is an array of 4 elements
"""

def build_text_based_prompt_ar(text: str, count: int, q_types: QuestionTypes) -> str:
    """
    بناء prompt من نص مصدر بالعربية
    """
    type_list = []
    if q_types.multiple_choice:
        type_list.append("اختيار من متعدد")
    if q_types.true_false:
        type_list.append("صح/خطأ")
    if q_types.fill_blank:
        type_list.append("أكمل الفراغ")
    
    types_str = " و ".join(type_list)
    
    return f"""
أنشئ {count} سؤالاً تعليمياً بناءً على النص التالي:

**النص المصدر:**
{text[:3000]}

**أنواع الأسئلة المطلوبة:** {types_str}

**التعليمات:**
1. الأسئلة يجب أن تستند فقط إلى المعلومات في النص أعلاه
2. أعد الإجابة بتنسيق JSON صالح فقط
3. لا تضيف أي نص خارج JSON

**تنسيق JSON:**
{{
  "questions": [
    {{
      "type": "multiple_choice",
      "q": "سؤال من النص",
      "options": ["خيار1", "خيار2", "خيار3", "خيار4"],
      "answer": 0,
      "correct_explanation": "شرح الإجابة",
      "wrong_explanations": ["شرح1", "شرح2", "شرح3", "شرح4"]
    }}
  ]
}}
"""

def build_text_based_prompt_en(text: str, count: int, q_types: QuestionTypes) -> str:
    """
    Build text-based prompt in English
    """
    type_list = []
    if q_types.multiple_choice:
        type_list.append("multiple choice")
    if q_types.true_false:
        type_list.append("true/false")
    if q_types.fill_blank:
        type_list.append("fill in the blank")
    
    types_str = " and ".join(type_list)
    
    return f"""
Create {count} educational questions based ONLY on the following text:

**Source Text:**
{text[:3000]}

**Required question types:** {types_str}

**Instructions:**
1. Questions must be based ONLY on the information in the text above
2. Return your answer in valid JSON format ONLY
3. Do NOT add any text outside the JSON

**JSON Format:**
{{
  "questions": [
    {{
      "type": "multiple_choice",
      "q": "Question from text",
      "options": ["option1", "option2", "option3", "option4"],
      "answer": 0,
      "correct_explanation": "Explanation",
      "wrong_explanations": ["expl1", "expl2", "expl3", "expl4"]
    }}
  ]
}}
"""

# دوال توليد الأسئلة
def generate_questions_with_retry(
    topic: str, 
    count: int, 
    q_types: QuestionTypes, 
    lang: str = "ar",
    source_text: Optional[str] = None
) -> Dict[str, List]:
    """
    توليد الأسئلة مع إعادة المحاولة عند الفشل
    """
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"محاولة توليد الأسئلة {attempt + 1}/{MAX_RETRIES}")
            
            model = get_model()
            
            # بناء الـ prompt المناسب
            if source_text:
                if lang == "ar":
                    prompt = build_text_based_prompt_ar(source_text, count, q_types)
                else:
                    prompt = build_text_based_prompt_en(source_text, count, q_types)
            else:
                if lang == "ar":
                    prompt = build_simplified_prompt_ar(topic, count, q_types)
                else:
                    prompt = build_simplified_prompt_en(topic, count, q_types)
            
            # إضافة تعليمات نهائية للتنسيق
            prompt += "\n\nتذكر: أعد الإجابة بتنسيق JSON صالح فقط، بدون أي نص إضافي."
            
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # توليد المحتوى
            response = model.generate_content(prompt)
            
            if not response or not response.text:
                raise ValueError("الرد من API فارغ")
            
            logger.debug(f"تم استلام رد بطول: {len(response.text)} حرف")
            
            # استخراج JSON
            data = extract_json_from_text(response.text)
            
            if not data:
                logger.error(f"فشل استخراج JSON. النص المستلم:\n{response.text[:500]}")
                raise ValueError("لم يتم العثور على JSON في الرد")
            
            if "questions" not in data:
                logger.error(f"مفتاح 'questions' مفقود. البيانات: {data}")
                raise ValueError("المفتاح 'questions' مفقود من JSON")
            
            # التحقق من صحة البيانات
            valid_questions = []
            for q in data.get("questions", []):
                if validate_question_structure(q, q_types):
                    valid_questions.append(q)
                else:
                    logger.warning(f"تم تجاهل سؤال بهيكل غير صالح: {q}")
            
            valid_count = len(valid_questions)
            if valid_count >= min(3, count):  # قبول إذا حصلنا على 3 أسئلة صالحة على الأقل
                logger.info(f"تم توليد {valid_count} سؤالاً صالحاً")
                return {"questions": valid_questions[:count]}
            
            logger.warning(f"محاولة {attempt + 1}: حصلنا على {valid_count} سؤالاً صالحاً فقط من {count}")
            
        except Exception as e:
            logger.error(f"محاولة {attempt + 1} فشلت: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # زيادة وقت الانتظار تدريجياً
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"فشل توليد الأسئلة بعد {MAX_RETRIES} محاولات: {str(e)}"
                )
    
    raise HTTPException(status_code=500, detail="فشل توليد الأسئلة")

# دوال معالجة الملفات
def extract_text_from_pdf(file) -> str:
    """استخراج النص من ملف PDF"""
    try:
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        text = text.strip()
        if not text:
            raise ValueError("لم يتم العثور على نص في ملف PDF")
        
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
        
        return text
    except Exception as e:
        logger.error(f"فشل استخراج النص من PDF: {e}")
        raise HTTPException(status_code=400, detail=f"فشل استخراج النص من PDF: {str(e)}")

def extract_text_from_image_ai(file) -> str:
    """استخراج النص من صورة باستخدام Gemini"""
    try:
        # إعادة تعيين مؤشر الملف
        file.seek(0)
        
        # تحميل الصورة
        image = Image.open(file)
        
        # تحويل الصورة إذا لزم الأمر
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # توليد النص
        model = get_model()
        prompt = "استخرج النص الموجود في هذه الصورة بدقة. أعد النص فقط بدون أي تعليقات إضافية."
        
        response = model.generate_content([prompt, image])
        
        if not response.text:
            raise ValueError("لم يتم العثور على نص في الصورة")
        
        text = response.text.strip()
        
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
        
        return text
    except Exception as e:
        logger.error(f"فشل استخراج النص من الصورة: {e}")
        raise HTTPException(status_code=400, detail=f"فشل استخراج النص من الصورة: {str(e)}")
    finally:
        file.seek(0)

# إنشاء تطبيق FastAPI
app = FastAPI(
    title="نظام توليد الأسئلة التعليمية",
    description="نظام ذكي لتوليد أسئلة تعليمية باستخدام Gemini AI",
    version="2.0.0"
)

# إضافة CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# نقاط النهاية
@app.get("/")
def root():
    """الصفحة الرئيسية"""
    return {
        "status": "active",
        "service": "نظام توليد الأسئلة التعليمية",
        "version": "2.0.0",
        "timestamp": get_current_time().isoformat()
    }

@app.get("/health")
def health_check():
    """فحص صحة النظام"""
    return {
        "status": "healthy",
        "database": os.path.exists(DB_FILE),
        "api_keys": len(keys),
        "timestamp": get_current_time().isoformat()
    }

@app.post("/generate/batch")
def generate_manual(
    req: GenerateReq,
    license_key: str = Header(..., alias="License-Key"),
    device_id: str = Header(..., alias="Device-ID")
):
    """
    توليد أسئلة يدوياً عن موضوع معين
    """
    logger.info(f"طلب توليد أسئلة: {req.topic}, عدد: {req.total_questions}")
    
    # التحقق من الترخيص
    db, idx = validate_license(license_key, device_id)
    
    # التحقق من أنواع الأسئلة
    if not (req.question_types.multiple_choice or 
            req.question_types.true_false or 
            req.question_types.fill_blank):
        raise HTTPException(
            status_code=400, 
            detail="يجب تحديد نوع سؤال واحد على الأقل"
        )
    
    # تحديد العدد
    total = min(req.total_questions, MAX_TOTAL)
    
    try:
        # توليد الأسئلة
        result = generate_questions_with_retry(
            topic=req.topic,
            count=total,
            q_types=req.question_types,
            lang=req.language.lang
        )
        
        # تحديث عدد الطلبات
        db[idx]["used_requests"] += 1
        db[idx]["last_request_at"] = get_current_time().isoformat()
        save_db(db)
        
        logger.info(f"تم توليد {len(result['questions'])} سؤالاً بنجاح")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ غير متوقع في توليد الأسئلة: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"خطأ في توليد الأسئلة: {str(e)}"
        )

@app.post("/generate/from-text")
def generate_from_text(
    text: str = Form(..., description="النص المصدر"),
    total_questions: int = Form(10, ge=1, le=50),
    multiple_choice: bool = Form(True),
    true_false: bool = Form(False),
    fill_blank: bool = Form(False),
    language: str = Form("ar"),
    license_key: str = Header(..., alias="License-Key"),
    device_id: str = Header(..., alias="Device-ID")
):
    """
    توليد أسئلة من نص معين
    """
    logger.info(f"طلب توليد أسئلة من نص بطول: {len(text)}")
    
    # التحقق من الترخيص
    db, idx = validate_license(license_key, device_id)
    
    # التحقق من أنواع الأسئلة
    if not (multiple_choice or true_false or fill_blank):
        raise HTTPException(
            status_code=400, 
            detail="يجب تحديد نوع سؤال واحد على الأقل"
        )
    
    # التحقق من النص
    if not text or len(text.strip()) < 50:
        raise HTTPException(
            status_code=400, 
            detail="النص قصير جداً، يجب أن يكون على الأقل 50 حرفاً"
        )
    
    # تحديد العدد
    total = min(total_questions, MAX_TOTAL)
    
    # إعداد أنواع الأسئلة
    q_types = QuestionTypes(
        multiple_choice=multiple_choice,
        true_false=true_false,
        fill_blank=fill_blank
    )
    
    try:
        # توليد الأسئلة
        result = generate_questions_with_retry(
            topic="نص مقدم",
            count=total,
            q_types=q_types,
            lang=language,
            source_text=text[:4000]  # تحديد طول النص
        )
        
        # تحديث عدد الطلبات
        db[idx]["used_requests"] += 1
        db[idx]["last_request_at"] = get_current_time().isoformat()
        save_db(db)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ في توليد الأسئلة من النص: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"خطأ في توليد الأسئلة: {str(e)}"
        )

@app.post("/generate/from-image")
async def generate_from_image(
    total_questions: int = Form(10, ge=1, le=50),
    multiple_choice: bool = Form(True),
    true_false: bool = Form(False),
    fill_blank: bool = Form(False),
    language: str = Form("ar"),
    file: UploadFile = File(...),
    license_key: str = Header(..., alias="License-Key"),
    device_id: str = Header(..., alias="Device-ID")
):
    """
    توليد أسئلة من صورة
    """
    logger.info(f"طلب توليد أسئلة من صورة: {file.filename}")
    
    # التحقق من الترخيص
    db, idx = validate_license(license_key, device_id)
    
    # التحقق من أنواع الأسئلة
    if not (multiple_choice or true_false or fill_blank):
        raise HTTPException(
            status_code=400, 
            detail="يجب تحديد نوع سؤال واحد على الأقل"
        )
    
    # التحقق من نوع الملف
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        raise HTTPException(
            status_code=400, 
            detail="نوع الملف غير مدعوم. الرجاء استخدام صورة (PNG, JPG, JPEG, WEBP)"
        )
    
    total = min(total_questions, MAX_TOTAL)
    
    # إعداد أنواع الأسئلة
    q_types = QuestionTypes(
        multiple_choice=multiple_choice,
        true_false=true_false,
        fill_blank=fill_blank
    )
    
    try:
        # استخراج النص من الصورة
        text = extract_text_from_image_ai(file.file)
        
        if not text or len(text.strip()) < 30:
            raise HTTPException(
                status_code=400, 
                detail="لم يتم العثور على نص كافي في الصورة"
            )
        
        logger.info(f"تم استخراج نص من الصورة بطول: {len(text)} حرف")
        
        # توليد الأسئلة
        result = generate_questions_with_retry(
            topic="نص مستخرج من صورة",
            count=total,
            q_types=q_types,
            lang=language,
            source_text=text
        )
        
        # تحديث عدد الطلبات
        db[idx]["used_requests"] += 1
        db[idx]["last_request_at"] = get_current_time().isoformat()
        save_db(db)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ في معالجة الصورة: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"خطأ في معالجة الصورة: {str(e)}"
        )

@app.post("/generate/from-pdf")
async def generate_from_pdf(
    total_questions: int = Form(10, ge=1, le=50),
    multiple_choice: bool = Form(True),
    true_false: bool = Form(False),
    fill_blank: bool = Form(False),
    language: str = Form("ar"),
    file: UploadFile = File(...),
    license_key: str = Header(..., alias="License-Key"),
    device_id: str = Header(..., alias="Device-ID")
):
    """
    توليد أسئلة من ملف PDF
    """
    logger.info(f"طلب توليد أسئلة من PDF: {file.filename}")
    
    # التحقق من الترخيص
    db, idx = validate_license(license_key, device_id)
    
    # التحقق من أنواع الأسئلة
    if not (multiple_choice or true_false or fill_blank):
        raise HTTPException(
            status_code=400, 
            detail="يجب تحديد نوع سؤال واحد على الأقل"
        )
    
    # التحقق من نوع الملف
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail="نوع الملف غير مدعوم. الرجاء استخدام ملف PDF"
        )
    
    total = min(total_questions, MAX_TOTAL)
    
    # إعداد أنواع الأسئلة
    q_types = QuestionTypes(
        multiple_choice=multiple_choice,
        true_false=true_false,
        fill_blank=fill_blank
    )
    
    try:
        # استخراج النص من PDF
        text = extract_text_from_pdf(file.file)
        
        logger.info(f"تم استخراج نص من PDF بطول: {len(text)} حرف")
        
        # توليد الأسئلة
        result = generate_questions_with_retry(
            topic="نص مستخرج من PDF",
            count=total,
            q_types=q_types,
            lang=language,
            source_text=text
        )
        
        # تحديث عدد الطلبات
        db[idx]["used_requests"] += 1
        db[idx]["last_request_at"] = get_current_time().isoformat()
        save_db(db)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ في معالجة PDF: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"خطأ في معالجة PDF: {str(e)}"
        )

# نقاط نهاية إدارة المسؤول
@app.post("/admin/create")
def admin_create(
    data: CreateLicense,
    x_admin_key: str = Header(..., alias="X-Admin-Key")
):
    """إنشاء ترخيص جديد"""
    admin_check(x_admin_key)
    
    db = load_db()
    
    # إنشاء مفتاح ترخيص
    key = "ST-" + secrets.token_hex(6).upper()
    
    license_data = {
        "license_key": key,
        "expires_at": (get_current_time() + timedelta(days=data.days)).isoformat(),
        "max_requests": data.max_requests,
        "used_requests": 0,
        "bound_device": None,
        "is_active": True,
        "owner": data.owner,
        "created_at": get_current_time().isoformat(),
        "last_request_at": None
    }
    
    db.append(license_data)
    save_db(db)
    
    logger.info(f"تم إنشاء ترخيص جديد للمالك: {data.owner}")
    
    return {
        "license_key": key,
        "expires_at": license_data["expires_at"],
        "max_requests": data.max_requests,
        "owner": data.owner
    }

@app.get("/admin/licenses")
def admin_list(
    x_admin_key: str = Header(..., alias="X-Admin-Key")
):
    """قائمة جميع التراخيص"""
    admin_check(x_admin_key)
    return load_db()

@app.put("/admin/update/{key}")
def admin_update(
    key: str,
    data: UpdateLicense,
    x_admin_key: str = Header(..., alias="X-Admin-Key")
):
    """تحديث ترخيص"""
    admin_check(x_admin_key)
    
    db = load_db()
    updated = False
    
    for license_data in db:
        if license_data["license_key"] == key:
            if data.days is not None:
                license_data["expires_at"] = (get_current_time() + timedelta(days=data.days)).isoformat()
            
            if data.max_requests is not None:
                license_data["max_requests"] = data.max_requests
            
            if data.is_active is not None:
                license_data["is_active"] = data.is_active
            
            save_db(db)
            updated = True
            logger.info(f"تم تحديث الترخيص: {key}")
            break
    
    if not updated:
        raise HTTPException(status_code=404, detail="الترخيص غير موجود")
    
    return {"status": "تم التحديث", "license_key": key}

@app.post("/admin/reset-device/{key}")
def admin_reset(
    key: str,
    x_admin_key: str = Header(..., alias="X-Admin-Key")
):
    """فصل الجهاز عن الترخيص"""
    admin_check(x_admin_key)
    
    db = load_db()
    for license_data in db:
        if license_data["license_key"] == key:
            license_data["bound_device"] = None
            save_db(db)
            logger.info(f"تم فصل الجهاز من الترخيص: {key}")
            return {"status": "تم فصل الجهاز", "license_key": key}
    
    raise HTTPException(status_code=404, detail="الترخيص غير موجود")

@app.delete("/admin/delete/{key}")
def admin_delete(
    key: str,
    x_admin_key: str = Header(..., alias="X-Admin-Key")
):
    """حذف ترخيص"""
    admin_check(x_admin_key)
    
    db = load_db()
    initial_count = len(db)
    
    db = [l for l in db if l["license_key"] != key]
    
    if len(db) == initial_count:
        raise HTTPException(status_code=404, detail="الترخيص غير موجود")
    
    save_db(db)
    logger.info(f"تم حذف الترخيص: {key}")
    
    return {"status": "تم الحذف", "license_key": key}

@app.post("/debug/test-generate")
async def debug_test_generate(
    topic: str = Form("الرياضيات"),
    count: int = Form(3),
    lang: str = Form("ar")
):
    """
    نقطة نهاية للتصحيح - عرض الرد الخام من Gemini
    """
    try:
        model = get_model()
        q_types = QuestionTypes(multiple_choice=True, true_false=False, fill_blank=False)
        
        prompt = build_simplified_prompt_ar(topic, count, q_types)
        
        response = model.generate_content(prompt)
        
        return {
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "raw_response": response.text,
            "json_extracted": extract_json_from_text(response.text)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/license/info/{license_key}")
def license_info(
    license_key: str,
    device_id: str = Header(..., alias="Device-ID")
):
    """
    الحصول على معلومات الترخيص
    """
    try:
        db, idx = validate_license(license_key, device_id)
        license_data = db[idx]
        
        # إزالة بعض الحقول الحساسة
        safe_data = {k: v for k, v in license_data.items() 
                    if k not in ["bound_device"]}
        
        return safe_data
    except HTTPException as e:
        return {"error": e.detail}

# نقطة بدء التطبيق
if __name__ == "__main__":
    import uvicorn
    
    # التحقق من وجود قاعدة البيانات
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        logger.info("تم إنشاء ملف قاعدة البيانات الجديد")
    
    logger.info("بدء تشغيل خادم توليد الأسئلة التعليمية...")
    logger.info(f"عدد مفاتيح API المتاحة: {len(keys)}")
    logger.info(f"نموذج Gemini المستخدم: {MODEL}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )