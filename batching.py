import json, itertools, os
import google.generativeai as genai
from fastapi import HTTPException

MODEL = "gemini-2.5-flash-lite"
MAX_RETRY = 2

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
    return (
        "Write the final output in clear academic English."
        if lang == "en"
        else "اكتب الناتج النهائي باللغة العربية الفصحى."
    )

def build_prompt(topic: str, lang: str, count: int):
    return f"""
{lang_instruction(lang)}

أنشئ {count} سؤال اختيار من متعدد من الموضوع التالي.

قواعد صارمة:
- 4 خيارات لكل سؤال
- شرح موسع وعميق للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- لا تكرر الأفكار
- مستوى تعليمي واضح
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

def generate_batch(
    topic: str,
    batch_size: int,
    language: str
):
    batch_size = min(max(batch_size, 5), 20)

    for attempt in range(MAX_RETRY + 1):
        try:
            model = get_model()
            prompt = build_prompt(topic, language, batch_size)
            response = model.generate_content(prompt)
            data = safe_json(response.text)

            if not data or "questions" not in data:
                raise ValueError("Invalid JSON from model")

            if len(data["questions"]) < batch_size:
                raise ValueError("Insufficient questions")

            return {
                "questions": data["questions"][:batch_size]
            }

        except Exception as e:
            if attempt == MAX_RETRY:
                raise HTTPException(
                    status_code=500,
                    detail=f"Batch generation failed: {str(e)}"
                )