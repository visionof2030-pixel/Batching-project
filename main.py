# main.py
import os
import json
import math
import itertools
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
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

def lang_instruction(lang: str):
    return "اكتب الناتج النهائي باللغة العربية الفصحى."

def build_prompt_from_text(text: str, lang: str, count: int):
    return f"""
{lang_instruction(lang)}

اعتمد على النص التالي فقط، ثم أنشئ {count} سؤال اختيار من متعدد.

قواعد:
- 4 خيارات
- شرح موسع للإجابة الصحيحة
- شرح مختصر ودقيق لكل خيار خاطئ
- JSON فقط

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

النص:
{text}
"""

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

@app.post("/generate/from-file")
async def generate_from_file(
    file: UploadFile = File(...),
    language: str = Form("ar"),
    total_questions: int = Form(10),
    license_key: str = Header(...),
    device_id: str = Header(...)
):
    total = min(max(total_questions, 1), MAX_TOTAL)
    model = get_model()
    content = await file.read()

    if file.content_type.startswith("image/"):
        response = model.generate_content([
            {
                "inline_data": {
                    "mime_type": file.content_type,
                    "data": content
                }
            },
            {"text": "استخرج النص التعليمي فقط من هذه الصورة."}
        ])
        extracted = response.text
    else:
        extracted = content.decode(errors="ignore")

    prompt = build_prompt_from_text(extracted, language, total)
    response = model.generate_content(prompt)
    data = safe_json(response.text)

    if not data or "questions" not in data:
        raise HTTPException(status_code=500, detail="Invalid AI output")

    return {"questions": data["questions"][:total]}