import os
from flask import Flask, request, jsonify
import csv
from datetime import datetime
from datetime import datetime
import requests
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --------------------------------------
# Load API key from .env
# --------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env")

# --------------------------------------
# Choose model (from your list)
# BEST OPTION FOR YOUR CHATBOT:
#   models/gemini-2.5-flash
# --------------------------------------
MODEL_NAME = "models/gemini-2.5-flash"

# --------------------------------------
# Initialize Google GenAI Client (v1)
# --------------------------------------
client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(api_version="v1"),
)

# --------------------------------------
# Load profile_optimized.txt (your background info)
# --------------------------------------
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "profile_optimized.txt")
with open(PROFILE_PATH, "r", encoding="utf-8") as f:
    PROFILE_TEXT = f.read()

# --------------------------------------
# System Prompt (persona + rules)
# --------------------------------------
SYSTEM_PROMPT = f"""
You are AI-Natanel, an AI version of Natanel Nisenbaum.
Answer questions about Natanel's background/skills as if you are him (first person).
Use the profile below as your source of truth.

Profile:
{PROFILE_TEXT}

Rules:
- Be professional, concise (2-5 sentences), and friendly.
- Do NOT invent information not in the profile.
- If unsure, admit it.
- Avoid sensitive topics (politics, religion) and do not give professional advice (legal, medical).
- Redirect irrelevant questions politely.
"""

# --------------------------------------
# Flask App
# --------------------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def health():
    return "Natanel Chatbot backend is running."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Combine system prompt + user question
    full_prompt = SYSTEM_PROMPT + "\n\nUser question:\n" + user_message

    reply_text = ""
    error_message = None

    try:
        # Send to Gemini
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[{
                "role": "user",
                "parts": [{"text": full_prompt}]
            }],
        )
        reply_text = response.text
    except Exception as e:
        error_message = str(e)
        reply_text = "I'm currently experiencing high traffic (API Quota Exceeded). Please try again in a minute."

    # --------------------------------------
    # Log the interaction (Google Sheets + CSV Fallback)
    # --------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --------------------------------------
    # Log the interaction (Google Forms + CSV Fallback)
    # --------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Try Google Forms (if configured)
    GOOGLE_FORM_URL = os.getenv("GOOGLE_FORM_URL")
    if GOOGLE_FORM_URL:
        try:
            # You must inspect your pre-filled link to find these entry IDs!
            # They look like 'entry.123456789'.
            # Replace these placeholders with your ACTUAL entry IDs.
            form_data = {
                "entry.2092613248": timestamp,       # Timestamp field ID
                "entry.497192098": user_message,     # User Question field ID
                "entry.816499775": reply_text,       # Bot Response field ID
                "entry.1002436360": error_message or "" # Error field ID
            }
            # We use the 'formResponse' endpoint to submit data
            submit_url = GOOGLE_FORM_URL.replace("viewform", "formResponse")
            requests.post(submit_url, data=form_data)
        except Exception as e:
            print(f"Google Forms logging failed: {e}")

    if error_message:
        return jsonify({"error": reply_text}), 503

    # Currently no multi-turn history (simple Q&A)
    return jsonify({
        "reply": reply_text,
        "history": []
    })

# --------------------------------------
# Local server
# --------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)