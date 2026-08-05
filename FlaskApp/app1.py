import os
from typing import Dict

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)

FAQ_BANK: Dict[str, str] = {
    "what is flask": "Flask is a lightweight Python web framework used to build web apps and APIs quickly.",
    "what is ai": "AI means artificial intelligence, where machines perform tasks that usually require human intelligence.",
    "how do i deploy the app": "You can deploy the app to a hosting platform such as Render, Railway, or Azure App Service.",
    "what is api": "An API is a way for different software systems to communicate with each other.",
    "how can i manage api keys": "Store API keys as environment variables and never hard-code them into your source files.",
}


def get_fallback_answer(question: str) -> str:
    text = question.lower().strip()
    for key, answer in FAQ_BANK.items():
        if key in text:
            return answer

    if "flask" in text:
        return "Flask is a lightweight Python web framework suited for small and medium web applications."
    if "ai" in text:
        return "AI can help automate tasks, provide recommendations, and generate helpful responses for users."
    return "I can help with Flask, AI, APIs, deployment, and secure API key management. Ask me anything related to the workshop topics."


def get_ai_answer(question: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return get_fallback_answer(question)

    try:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful workshop assistant for a Flask and AI FAQ. Answer briefly and clearly.",
                },
                {"role": "user", "content": question},
            ],
            "temperature": 0.4,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return get_fallback_answer(question)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({"answer": "Please type a question so I can help you."})

    answer = get_ai_answer(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
