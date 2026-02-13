import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

# Create client (v1 API)
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1"),
)

print("\nAvailable models for your API key:\n")

# List models
for m in client.models.list():
    print("-", m.name)