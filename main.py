from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os

from key_logic import validate_time_key

MODEL = "gemini-1.5-flash"

genai.configure(api_key=os.getenv("GEMINI_KEY"))

app = FastAPI()


class GenerateRequest(BaseModel):
    access_key: str
    topic: str
    total_questions: int = 5
    language: str = "ar"


def lang_instruction(lang: str):
    return (
        "Write the final output in clear academic English."
        if lang == "en"
        else "اكتب الناتج النهائي باللغة العربية الفصحى الواضحة."
    )


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    validate_time_key(req.access_key)

    prompt = f"""
{lang_instruction(req.language)}

أنشئ {req.total_questions} سؤال اختيار من متعدد من الموضوع التالي.

قواعد:
- 4 خيارات لكل سؤال
- شرح موسع للإجابة الصحيحة
- شرح مختصر لكل خيار خاطئ
- لا تكرر الأفكار
- أعد الناتج بشكل منسق وواضح

الموضوع:
{req.topic}
"""

    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(prompt)

    return {
        "result": response.text
    }