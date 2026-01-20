import os, json, math, itertools
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
MAX_RETRY = 3
MAX_TOTAL = 200

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

def lang_instruction(lang):
    return (
        "Write the final output in clear academic English."
        if lang == "en"
        else "اكتب الناتج النهائي باللغة العربية الفصحى الواضحة."
    )

def build_quiz_prompt(topic, lang, count):
    return f"""
{lang_instruction(lang)}

أنشئ {count} سؤال اختيار من متعدد من الموضوع التالي.

قواعد صارمة:
- 4 خيارات فقط
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

def build_flash_prompt(topic, lang, count):
    return f"""
{lang_instruction(lang)}

أنشئ {count} بطاقات تعليمية Flash Cards من الموضوع التالي.

قواعد:
- فكرة واحدة لكل بطاقة
- صياغة تعليمية واضحة
- لا تكرر
- أعد JSON فقط

الصيغة:
{{
 "cards":[
  {{
   "front":"",
   "back":""
  }}
 ]
}}

الموضوع:
{topic}
"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_questions: int = 10

class FlashRequest(BaseModel):
    topic: str
    language: str = "ar"
    total_cards: int = 10

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/generate/batch")
def generate_quiz(req: QuizRequest):
    total = min(max(req.total_questions, 1), MAX_TOTAL)
    batches = math.ceil(total / BATCH_SIZE)
    final_questions = []

    for batch in range(batches):
        remaining = total - len(final_questions)
        if remaining <= 0:
            break

        needed = min(BATCH_SIZE, remaining)

        for attempt in range(MAX_RETRY):
            model = get_model()
            prompt = build_quiz_prompt(req.topic, req.language, needed)
            try:
                r = model.generate_content(prompt)
                data = safe_json(r.text)
                if not data or "questions" not in data:
                    raise ValueError("Invalid JSON")

                final_questions.extend(data["questions"][:needed])
                break
            except:
                if attempt == MAX_RETRY - 1:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Batch failed at {len(final_questions)}"
                    )

    if len(final_questions) != total:
        raise HTTPException(
            status_code=500,
            detail=f"Generated {len(final_questions)} of {total}"
        )

    return {"questions": final_questions}

@app.post("/generate/flashcards")
def generate_flashcards(req: FlashRequest):
    total = min(max(req.total_cards, 5), 60)
    model = get_model()
    prompt = build_flash_prompt(req.topic, req.language, total)
    r = model.generate_content(prompt)
    data = safe_json(r.text)

    if not data or "cards" not in data:
        raise HTTPException(status_code=500, detail="Invalid flashcards JSON")

    return {"cards": data["cards"][:total]}