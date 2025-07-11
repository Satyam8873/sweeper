from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load API keys from environment
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")

# NVIDIA NIM client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Supported language translation modes
LANG_MODES = {
    "en_to_it": "Translate the user's English into perfect Italian.",
    "it_to_en": "Translate the user's Italian into perfect English."
}

@app.route("/translate", methods=["POST"])
def translate():
    # 🔐 Authorization check
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_SECRET_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    # 🔤 Parse request
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    lang_mode = data.get("lang", "").strip().lower()

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400
    if lang_mode not in LANG_MODES:
        return jsonify({"error": "Invalid lang. Use 'en_to_it' or 'it_to_en'."}), 400

    try:
        # 🧠 Call NVIDIA NIM for translation
        system_prompt = f"You are a translator. {LANG_MODES[lang_mode]}"

        response = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=1,
            max_tokens=300
        )

        translated = response.choices[0].message.content.strip()
        return jsonify({
            "lang": lang_mode,
            "translated": translated
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
