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
    ollama_url = os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434/api/generate"

    preferred_models = []
    configured_model = os.getenv("OLLAMA_MODEL")
    if configured_model:
        preferred_models.append(configured_model)
    preferred_models.extend(["llama3.2:1b","gemma3:270m", "smollm2:360m"])

    seen_models = set()
    for model in preferred_models:
        if not model or model in seen_models:
            continue
        seen_models.add(model)

        try:
            payload = {
                "model": model,
                "prompt": question,
                "stream": False,
            }
            response = requests.post(ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            answer = (data.get("response") or "").strip()
            if answer:
                return answer
        except Exception:
            continue

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
