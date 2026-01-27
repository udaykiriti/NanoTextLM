from flask import Flask, render_template, request, jsonify
from inference import generate
import os

app = Flask(__name__)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.pt")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer.json")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def generate_api():
    data = request.json
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
        
    try:
        response = generate(MODEL_PATH, TOKENIZER_PATH, prompt)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting TinyTechLM Web App...")
    app.run(host="0.0.0.0", port=5000, debug=True)
