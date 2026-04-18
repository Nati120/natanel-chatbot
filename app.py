import os
import threading
from flask import Flask, request, jsonify
from datetime import datetime
import requests
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
import anthropic

# --------------------------------------
# Load API key from .env
# --------------------------------------
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ ANTHROPIC_API_KEY not found in .env")

# --------------------------------------
# Model + cost caps
# Haiku 4.5: cheapest Claude model, fine for 2-5 sentence Q&A.
# max_tokens caps the reply length so one request can't run up a bill.
# --------------------------------------
MODEL_NAME = "claude-haiku-4-5"
MAX_TOKENS = 512
MAX_MESSAGE_LENGTH = 1000  # characters — reject anything longer before calling Claude

client = anthropic.Anthropic(api_key=API_KEY)

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
# Helper: Log to Google Forms
# --------------------------------------
def log_to_google_forms(user_message, reply_text, error_message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    GOOGLE_FORM_URL = os.getenv("GOOGLE_FORM_URL")
    if not GOOGLE_FORM_URL:
        return

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
        # Set a timeout so we don't hang if Google is slow
        requests.post(submit_url, data=form_data, timeout=3)
        print(f"Google Forms logging successful for message: {user_message[:20]}...")
    except Exception as e:
        print(f"Google Forms logging failed: {e}")

# --------------------------------------
# Flask App
# --------------------------------------
app = Flask(__name__)

# Trust one layer of proxy (Heroku / Render / Cloudflare set X-Forwarded-For).
# Without this, rate limits would see the proxy's IP and cap all traffic together.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# CORS: only allow requests from your website.
CORS(
    app,
    origins=["https://natanelnisenbaum.com"],
)

# Rate limit: per-IP caps. Defaults apply to every route;
# /chat gets a tighter per-minute cap via the decorator below.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
)

@app.route("/")
def health():
    return "Natanel Chatbot backend is running."

@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    data = request.get_json() or {}
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    if len(user_message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Message too long (max {MAX_MESSAGE_LENGTH} characters)."}), 413

    reply_text = ""
    error_message = None

    try:
        # cache_control marks the system prompt as cacheable. On repeat requests
        # the profile portion is served from cache at ~10% cost. Note: caching
        # only kicks in once the cached prefix clears Haiku's 4096-token minimum,
        # so a short profile won't actually cache — grow it or it's a no-op.
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )
        reply_text = next((b.text for b in response.content if b.type == "text"), "")
    except anthropic.APIError as e:
        error_message = str(e)
        reply_text = "I'm currently experiencing high traffic. Please try again in a minute."

    # Log to Google Forms in a background thread so we don't block the response
    threading.Thread(
        target=log_to_google_forms,
        args=(user_message, reply_text, error_message),
    ).start()

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