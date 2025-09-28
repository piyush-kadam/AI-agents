import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # loads GEMINI_API_KEY from .env

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

for m in genai.list_models():
    print(m.name)
