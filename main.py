import os
import json
import time
import itertools
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ================= CONFIG =================
MODEL_NAME = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
MAX_RETRIES = 2
DELAY_BETWEEN_BATCHES = 2
# ==========================================

# Load API Keys
API_KEYS = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12)]
API_KEYS = [k for k in API_KEYS if k]

if not API_KEYS:
    raise RuntimeError("❌ No Gemini API keys found")

key_cycle = itertools.cycle(API_KEYS)

# ================= FASTAPI =================
app = FastAPI(title="SmartTest API")

# ================= MODELS =================
class QuizRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int = 10

# ================= GEMINI =================
def get_client():
    api_key = next(key_cycle)
    return genai.Client(api_key=api_key)

def extract_json(text: str):
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except:
        return None

def build_prompt(topic: str, lang: str, count: int):
    lang_line = (
        "Write in clear academic English."
        if lang == "en"
        else "اكتب باللغة العربية الفصحى."
    )

    return f"""
{lang_line}

أنشئ {count} سؤال اختيار من متعدد.

قواعد صارمة:
- 4 خيارات لكل سؤال
- خيار صحيح واحد فقط
- شرح للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- لا تكرر الأفكار
- أعد JSON فقط بدون أي نص إضافي

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

def generate_batch(topic, lang, count):
    for attempt in range(MAX_RETRIES + 1):
        try:
            client = get_client()
            prompt = build_prompt(topic, lang, count)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            data = extract_json(response.text)
            if not data or "questions" not in data:
                raise ValueError("Invalid JSON")

            return data["questions"][:count]

        except Exception as e:
            if attempt == MAX_RETRIES:
                raise e
            time.sleep(2)

# ================= ROUTES =================
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/generate")
def generate_quiz(req: QuizRequest):
    total = max(5, min(req.total_questions, 200))
    all_questions = []

    while len(all_questions) < total:
        remaining = total - len(all_questions)
        batch_count = min(BATCH_SIZE, remaining)

        try:
            batch = generate_batch(
                topic=req.topic,
                lang=req.language,
                count=batch_count
            )
            all_questions.extend(batch)
            time.sleep(DELAY_BETWEEN_BATCHES)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

    return {
        "questions": all_questions
    }