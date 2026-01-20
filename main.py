import json
import math
import time
import itertools
import os

import google.generativeai as genai
from fastapi import HTTPException

MODEL = "gemini-2.5-flash-lite"

BATCH_SIZE = 20
MAX_RETRY = 3
RETRY_SLEEP_SECONDS = 8

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
    except Exception:
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
- 4 خيارات لكل سؤال
- شرح موسع وعميق للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- لا تكرر الأفكار
- مستوى تعليمي واضح
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


def generate_single_batch(topic: str, batch_size: int, language: str):
    for attempt in range(MAX_RETRY):
        try:
            model = get_model()
            prompt = build_prompt(topic, language, batch_size)
            response = model.generate_content(prompt)
            data = safe_json(response.text)

            if not data or "questions" not in data:
                raise ValueError("Invalid JSON returned")

            questions = data["questions"]
            if not isinstance(questions, list) or len(questions) < batch_size:
                raise ValueError("Insufficient questions returned")

            return questions[:batch_size]

        except Exception:
            if attempt == MAX_RETRY - 1:
                raise
            time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError("Unreachable")


def generate_questions(
    topic: str,
    total_questions: int,
    language: str = "ar"
):
    total_questions = min(max(total_questions, 5), 200)

    total_batches = math.ceil(total_questions / BATCH_SIZE)
    final_questions = []

    for batch_index in range(total_batches):
        remaining = total_questions - len(final_questions)
        current_batch_size = min(BATCH_SIZE, remaining)

        try:
            batch_questions = generate_single_batch(
                topic=topic,
                batch_size=current_batch_size,
                language=language
            )
            final_questions.extend(batch_questions)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Batch {batch_index + 1} failed: {str(e)}"
            )

        time.sleep(2)

    return {
        "questions": final_questions[:total_questions],
        "total": len(final_questions[:total_questions]),
        "model": MODEL
    }