from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import torch
from config import ModelConfig
from model import NanoTextLM
from tokenizers import Tokenizer
import os
import json

app = Flask(__name__)

# Global Model & Tokenizer (Lazy Loading)
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_resources():
    global model, tokenizer
    if model is None:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.pt")
        # Fallback
        if not os.path.exists(model_path):
             model_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
             
        tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer.json")
        
        m_conf = ModelConfig()
        model = NanoTextLM(m_conf).to(device)
        
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("Warning: No checkpoint found. Using random weights.")
            
        model.eval()
        # Compilation might be too slow for first request, optional
        # model = torch.compile(model)
        tokenizer = Tokenizer.from_file(tokenizer_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate_stream", methods=["POST"])
def generate_stream_api():
    load_resources()
    data = request.json
    prompt = data.get("prompt", "")
    temperature = data.get("temperature", 1.0)
    top_p = data.get("top_p", 0.9)
    max_tokens = data.get("max_tokens", 150)
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    def generate():
        # Encode
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        
        # Stream
        with torch.no_grad():
            for token_idx in model.generate_stream(idx, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p):
                token_val = token_idx.item()
                text_chunk = tokenizer.decode([token_val])
                yield text_chunk

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == "__main__":
    print("Starting NanoTextLM Web App on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
