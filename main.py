import json, math, itertools, os
import google.generativeai as genai
from fastapi import HTTPException

MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 10
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

def build_prompt(topic: str, lang: str, count: int):
    lang_line = (
        "Write the final output in clear academic English."
        if lang == "en"
        else "اكتب الناتج النهائي باللغة العربية الفصحى."
    )
    return f"""
{lang_line}

أنشئ {count} سؤال اختيار من متعدد من الموضوع التالي.

قواعد صارمة:
- 4 خيارات لكل سؤال
- شرح موسع وعميق للإجابة الصحيحة
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

def generate_questions_pro(
    topic: str,
    total_questions: int,
    language: str
):
    total_questions = min(max(total_questions, 5), 200)
    batches = math.ceil(total_questions / BATCH_SIZE)
    final_questions = []

    for batch_index in range(batches):
        needed = min(
            BATCH_SIZE,
            total_questions - len(final_questions)
        )

        success = False
        for attempt in range(MAX_RETRY + 1):
            model = get_model()
            prompt = build_prompt(topic, language, needed)

            try:
                response = model.generate_content(prompt)
                data = safe_json(response.text)

                if not data or "questions" not in data:
                    raise ValueError("Invalid JSON")

                questions = data["questions"]
                if len(questions) < needed:
                    raise ValueError("Insufficient questions")

                final_questions.extend(questions[:needed])
                success = True
                break

            except Exception as e:
                if attempt == MAX_RETRY:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Batch {batch_index+1} failed: {str(e)}"
                    )

        if not success:
            break

    return {
        "questions": final_questions[:total_questions]
    }
