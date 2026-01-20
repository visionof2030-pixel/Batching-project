# main.py
import os
import json
import time
import itertools
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

MODEL = "gemini-2.5-flash-lite"
MAX_RETRY = 2
BATCH_LIMIT = 20

keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 12)]
keys = [k for k in keys if k]
if not keys:
    raise RuntimeError("No Gemini API keys found")

key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(MODEL)

def safe_json(text: str):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return None

def lang_instruction(lang: str):
    return "اكتب الناتج النهائي باللغة العربية الفصحى." if lang == "ar" else "Write the final output in clear academic English."

def build_prompt(topic: str, lang: str, count: int):
    return f"""
{lang_instruction(lang)}

أنشئ {count} سؤال اختيار من متعدد من الموضوع التالي.

قواعد صارمة:
- 4 خيارات لكل سؤال
- شرح موسع للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- لا تكرر الأفكار
- أعد JSON فقط

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

class GenerateRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    total = max(1, req.total_questions)
    results: List[dict] = []

    while len(results) < total:
        batch = min(BATCH_LIMIT, total - len(results))
        for attempt in range(MAX_RETRY + 1):
            try:
                model = get_model()
                prompt = build_prompt(req.topic, req.language, batch)
                response = model.generate_content(prompt)
                data = safe_json(response.text)
                if not data or "questions" not in data:
                    raise ValueError("Invalid JSON")
                results.extend(data["questions"][:batch])
                break
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "quota" in msg:
                    raise HTTPException(status_code=429, detail="انتهى الحد المجاني للمفاتيح")
                if attempt == MAX_RETRY:
                    raise HTTPException(status_code=500, detail="فشل التوليد")
                time.sleep(2)

    return {"questions": results}