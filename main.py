import os
import json
import math
import itertools
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

MODEL_NAME = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
MAX_RETRIES = 2

keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12)]
keys = [k for k in keys if k]
if not keys:
    raise RuntimeError("No Gemini API keys found")

key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(MODEL_NAME)

app = FastAPI()

class QuizRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int = 10

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
- خيار صحيح واحد فقط
- شرح موسع للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- لا تكرر الأسئلة أو الأفكار
- أعد JSON فقط دون أي نص إضافي

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

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: QuizRequest):
    total = min(max(req.total_questions, 1), 200)
    batches = math.ceil(total / BATCH_SIZE)
    final_questions = []

    for batch_index in range(batches):
        needed = min(BATCH_SIZE, total - len(final_questions))
        for attempt in range(MAX_RETRIES + 1):
            try:
                model = get_model()
                prompt = build_prompt(req.topic, req.language, needed)
                response = model.generate_content(prompt)
                data = extract_json(response.text)
                if not data or "questions" not in data:
                    raise ValueError("Invalid JSON")
                questions = data["questions"]
                if len(questions) < needed:
                    raise ValueError("Insufficient questions")
                final_questions.extend(questions[:needed])
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Batch {batch_index + 1} failed: {str(e)}"
                    )

    return {
        "questions": final_questions[:total],
        "count": len(final_questions[:total]),
        "model": MODEL_NAME
    }