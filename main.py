
import json
import math
import itertools
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# ================== إعدادات عامة ==================
MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 20          # عدد الأسئلة في كل دفعة
MAX_QUESTIONS = 200
MAX_RETRY = 2
SLEEP_BETWEEN_BATCHES = 1.2  # ثواني لتخفيف الضغط

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
def lang_instruction(lang: str):
    return (
        "Write the final output in clear academic English."
        if lang == "en"
        else "اكتب الناتج النهائي باللغة العربية الفصحى الواضحة."
    )

def extract_json(text: str):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return None

def build_prompt(topic: str, lang: str, count: int):
    return f"""
{lang_instruction(lang)}

أنشئ {count} سؤال اختيار من متعدد من الموضوع التالي.

قواعد صارمة:
- 4 خيارات لكل سؤال
- شرح موسع وواضح للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
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

# ================== منطق الـ Batching ==================
def generate_questions_pro(topic: str, total_questions: int, language: str):
    total_questions = min(max(total_questions, 5), MAX_QUESTIONS)

    batches = math.ceil(total_questions / BATCH_SIZE)
    final_questions = []

    for batch_index in range(batches):
        remaining = total_questions - len(final_questions)
        if remaining <= 0:
            break

        current_batch_size = min(BATCH_SIZE, remaining)
        success = False

        for attempt in range(MAX_RETRY + 1):
            try:
                model = get_model()
                prompt = build_prompt(topic, language, current_batch_size)
                response = model.generate_content(prompt)

                data = extract_json(response.text)
                if not data or "questions" not in data:
                    raise ValueError("Invalid JSON returned")

                questions = data["questions"]
                if not isinstance(questions, list) or len(questions) == 0:
                    raise ValueError("No questions generated")

                final_questions.extend(questions[:current_batch_size])
                success = True
                break

            except Exception as e:
                if attempt == MAX_RETRY:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Batch {batch_index + 1} failed: {str(e)}"
                    )

        time.sleep(SLEEP_BETWEEN_BATCHES)

        if not success:
            break

    if not final_questions:
        raise HTTPException(
            status_code=500,
            detail="No questions generated from model"
        )

    return {
        "questions": final_questions[:total_questions]
    }

# ================== FastAPI ==================
app = FastAPI()

class QuizRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int = 10

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: QuizRequest):
    return generate_questions_pro(
        topic=req.topic,
        total_questions=req.total_questions,
        language=req.language
    )