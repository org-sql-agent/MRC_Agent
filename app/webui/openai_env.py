import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# ========= 基本設定 =========
SD_API = os.getenv("SD_API", "http://mrc-agent:8000")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # 具備vision+tool calling
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
