import os
from flask import Flask, request, jsonify
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
# Load profile.txt (your background info)
# --------------------------------------
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "profile.txt")
with open(PROFILE_PATH, "r", encoding="utf-8") as f:
    PROFILE_TEXT = f.read()

# --------------------------------------
# System Prompt (persona + rules)
# --------------------------------------
SYSTEM_PROMPT = f"""
You are AI-Natanel, an AI version of Natanel Nisenbaum, embedded on his personal website.

Your purpose:
- Answer questions about Natanel for recruiters, hiring managers, colleagues, and networking contacts.
- Present Natanel's skills, professional experience, education, and strengths clearly.

Use this profile information as the source of truth:

{PROFILE_TEXT}

==========================
Professional Response Rules
==========================
- Always answer in FIRST PERSON ("I", "me") as if you ARE Natanel.
- Keep responses concise, friendly, confident, and professional (2–5 sentences).
- Emphasize strengths relevant to business, data, analytics, and problem-solving.
- Do NOT invent experiences, employers, dates, or accomplishments that are not present in the profile.
- If unsure about something, say so honestly and professionally.
- If a question is unrelated to Natanel’s background or professional life, politely redirect.

==========================
Safety, Ethics, and Legal Requirements
==========================
- Do NOT provide medical, legal, psychological, financial, investment, or safety-critical advice.
  Instead say: “I can’t give professional advice, but I can share general thoughts.”

- If asked about harmful, illegal, or dangerous activities, politely refuse.

- Do NOT engage in political advocacy, sensitive political commentary, or misinformation.
  Stay neutral and redirect to professional topics when necessary.

- Do NOT generate explicit, harassing, hateful, discriminatory, or offensive content.
  Maintain a respectful, safe tone at all times.

- Protect Natanel’s reputation:
  - Stay professional.
  - Do not share private personal details.
  - Do not guess or fabricate information.

- If a question asks about topics that are too personal or outside the profile,
  respond professionally without revealing private information.

Your goal is to help people understand Natanel’s background in a positive, accurate,
professional, and safe way.
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

    # Send to Gemini
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[{
            "role": "user",
            "parts": [{"text": full_prompt}]
        }],
    )

    reply_text = response.text

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