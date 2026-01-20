import os
import json
import math
import itertools
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ================== إعدادات عامة ==================
MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 10          # عدد الأسئلة لكل دفعة
MAX_RETRY = 3            # عدد المحاولات لكل دفعة
MAX_TOTAL = 200          # الحد الأعلى المسموح

# ================== مفاتيح Gemini ==================
keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12)]
keys = [k for k in keys if k]

if not keys:
    raise RuntimeError("No Gemini API keys found")

key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(MODEL)

# ================== أدوات مساعدة ==================
def safe_json(text: str):
    """
    استخراج JSON آمن من رد النموذج
    """
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return None

def lang_instruction(lang: str):
    return (
        "Write the final output in clear academic English."
        if lang == "en"
        else "اكتب الناتج النهائي باللغة العربية الفصحى الواضحة."
    )

def build_prompt(topic: str, lang: str, count: int):
    return f"""
{lang_instruction(lang)}

أنشئ {count} سؤال اختيار من متعدد من الموضوع التالي.

قواعد صارمة:
- 4 خيارات فقط لكل سؤال
- شرح موسع وعميق للإجابة الصحيحة
- شرح مختصر وواضح لكل خيار خاطئ
- لا تكرر الأفكار أو الأسئلة
- أعد JSON فقط بدون أي نص إضافي

الصيغة:
{{
  "questions": [
    {{
      "q": "",
      "options": ["", "", "", ""],
      "answer": 0,
      "explanations": ["", "", "", ""]
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int = 10

@app.get("/")
def root():
    return {"status": "ok"}

# ================== Endpoint الرئيسي ==================
@app.post("/generate/batch")
def generate_batch(req: GenerateRequest):
    total = min(max(req.total_questions, 1), MAX_TOTAL)

    batches = math.ceil(total / BATCH_SIZE)
    final_questions = []

    for batch_index in range(batches):
        remaining = total - len(final_questions)
        if remaining <= 0:
            break

        needed = min(BATCH_SIZE, remaining)
        success = False

        for attempt in range(MAX_RETRY):
            model = get_model()
            prompt = build_prompt(req.topic, req.language, needed)

            try:
                response = model.generate_content(prompt)
                data = safe_json(response.text)

                if not data or "questions" not in data:
                    raise ValueError("Invalid JSON returned")

                questions = data["questions"]
                if not isinstance(questions, list) or len(questions) == 0:
                    raise ValueError("Empty questions list")

                # 🔒 قص صارم حسب العدد المتبقي
                final_questions.extend(questions[:needed])
                success = True
                break

            except Exception as e:
                if attempt == MAX_RETRY - 1:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Batch {batch_index + 1} failed: {str(e)}"
                    )

        if not success:
            break

    # ================== تحقق نهائي ==================
    if len(final_questions) != total:
        raise HTTPException(
            status_code=500,
            detail=f"Generated {len(final_questions)} questions out of {total}"
        )

    return {
        "questions": final_questions
    }