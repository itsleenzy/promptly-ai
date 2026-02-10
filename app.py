import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

client = Groq(api_key=api_key)

def build_system_message(tone: str, format_hint: str, constraints: str) -> str:
    constraint_line = f"Additional constraints: {constraints}." if constraints else ""
    format_line = f"Desired formatting style: {format_hint}." if format_hint else ""
    return (
        "You are an AI Prompt Engineer. Use the RTF framework (Role, Task, Format) "
        f"and a {tone} tone. First infer the user's intent and desired vibe internally, "
        "then produce a structured prompt with explicit delimiters. "
        "Output format:\n"
        "[ROLE] ...\n"
        "[TASK] ...\n"
        "[FORMAT] ...\n"
        "[CONSTRAINTS] ...\n"
        f"{format_line} {constraint_line} "
        "Do not include your reasoning or analysis. Output only the structured prompt."
    ).strip()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/improve', methods=['POST'])
def improve():
    data = request.json
    raw_prompt = data.get("prompt", "")
    tone = data.get("tone", "Professional") # Get the tone from the UI
    format_hint = data.get("format_hint", "")
    constraints = data.get("constraints", "")

    system_message = build_system_message(tone, format_hint, constraints)

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": raw_prompt}
            ]
        )
        return jsonify({"improved_prompt": completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/improve-bulk', methods=['POST'])
def improve_bulk():
    data = request.json
    prompts = data.get("prompts", [])
    tone = data.get("tone", "Professional")
    format_hint = data.get("format_hint", "")
    constraints = data.get("constraints", "")

    if not isinstance(prompts, list) or len(prompts) == 0:
        return jsonify({"error": "prompts must be a non-empty array"}), 400

    system_message = build_system_message(tone, format_hint, constraints)

    results = []
    for raw_prompt in prompts:
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": raw_prompt}
                ]
            )
            results.append({
                "input": raw_prompt,
                "improved": completion.choices[0].message.content
            })
        except Exception as e:
            results.append({
                "input": raw_prompt,
                "error": str(e)
            })

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(port=5000)
    

