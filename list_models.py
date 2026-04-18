import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY not found in .env")

client = anthropic.Anthropic(api_key=api_key)

print("\nAvailable Claude models for your API key:\n")

for m in client.models.list():
    print(f"- {m.id}  ({m.display_name})")
