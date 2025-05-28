import os
import requests
from flask import current_app

# Replace OpenAI API usage with Gemini API usage

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyB-GIIKVZFckpgr77mF88MffwK0JzIgldM')
GEMINI_API_URL = "https://gemini.api.endpoint/v1/chat"  # Replace with actual Gemini API endpoint

# -- Wingman functions --
def chat_with_wingman(session_id, user_message):
    """
    Send user message to Gemini API and get assistant response.
    """
    headers = {
        "Authorization": f"Bearer {'AIzaSyB-GIIKVZFckpgr77mF88MffwK0JzIgldM'}",
        "Content-Type": "application/json"
    }
    payload = {
        "session_id": session_id,
        "message": user_message
    }
    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        assistant_message = data.get("response", "Sorry, I couldn't process your request.")
        return assistant_message
    except Exception as e:
        current_app.logger.error(f"Gemini API error: {str(e)}")
        return "Sorry, there was an error communicating with the assistant."
