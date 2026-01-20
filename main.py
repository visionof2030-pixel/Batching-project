from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os, itertools, math, json, time

# ================== إعداد Gemini ==================
MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 20
MAX_TOTAL = 200

keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 20)]
keys = [k for k in keys if k]
if not keys:
    raise RuntimeError("No Gemini API keys found")

key_cycle = itertools.cycle(keys)

def get_model():
    genai.configure(api_key=next(key_cycle))
    return genai.GenerativeModel(MODEL)

# ================== FastAPI ==================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int = 10
    offset: int = 0

@app.get("/")
def root():
    return {"status": "ok"}

# ================== أدوات مساعدة ==================
def extract_json(text: str):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return None

def build_prompt(topic, lang, count, offset):
    lang_line = (
        "Write the final output in clear academic English."
        if lang == "en"
        else "اكتب الناتج النهائي باللغة العربية الفصحى."
    )
    return f"""
{lang_line}

أنشئ {count} سؤال اختيار من متعدد (بدءًا من السؤال رقم {offset + 1}).

قواعد صارمة:
- 4 خيارات
- شرح موسع للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- لا تكرر الأسئلة السابقة
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

# ================== Endpoint مرحلي ==================
@app.post("/generate/batch")
def generate_batch(req: QuizRequest):
    total = min(req.total_questions, MAX_TOTAL)
    if total <= 0:
        raise HTTPException(400, "Invalid total_questions")

    batch_count = min(BATCH_SIZE, total - req.offset)
    if batch_count <= 0:
        return {"questions": [], "done": True}

    model = get_model()
    prompt = build_prompt(req.topic, req.language, batch_count, req.offset)

    try:
        r = model.generate_content(prompt)
        data = extract_json(r.text)
        if not data or "questions" not in data:
            raise ValueError("Invalid JSON from model")

        return {
            "questions": data["questions"],
            "generated": len(data["questions"]),
            "offset": req.offset + len(data["questions"]),
            "done": req.offset + len(data["questions"]) >= total
        }

    except Exception as e:
        raise HTTPException(500, str(e))