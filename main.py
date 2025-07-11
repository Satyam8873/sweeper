from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# NVIDIA NIM
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Supported translation modes
LANG_MODES = {
    "en_to_it": "Translate the user's English into perfect Italian.",
    "it_to_en": "Translate the user's Italian into perfect English."
}

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    lang_mode = data.get("lang", "").strip().lower()

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400
    if lang_mode not in LANG_MODES:
        return jsonify({"error": "Invalid lang. Use 'en_to_it' or 'it_to_en'."}), 400

    try:
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
