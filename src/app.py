from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
from config import ModelConfig
from model import NanoTextLM
from tokenizers import Tokenizer
import os
import asyncio

# Setup
app = FastAPI(title="NanoTextLM API")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "src", "templates"))

# Global State
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 1.0
    top_p: float = 0.9
    max_tokens: int = 150

def load_resources():
    global model, tokenizer
    if model is None:
        print(f"Loading NanoTextLM on {device}...")
        model_path = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.pt")
        if not os.path.exists(model_path):
             model_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
             
        tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer.json")
        
        m_conf = ModelConfig()
        model = NanoTextLM(m_conf).to(device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)
        else:
            print("Warning: No checkpoint found. Using random weights.")
            
        model.eval()
        if hasattr(torch, "compile"):
             # Compilation might add startup latency but speeds up serving
             model = torch.compile(model)
             
        tokenizer = Tokenizer.from_file(tokenizer_path)

@app.on_event("startup")
async def startup_event():
    load_resources()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/generate_stream")
async def generate_stream_api(req: GenerateRequest):
    if not req.prompt:
        return {"error": "No prompt provided"}

    async def generate():
        # Offload blocking tensor ops to thread pool if needed, 
        # but for simplicity we run direct since generate_stream yields.
        # Ideally: await loop.run_in_executor(...) for blocking parts
        
        ids = tokenizer.encode(req.prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        
        # We need a wrapper to make the synchronous generator async-friendly
        # or just iterate it. FastAPI handles iterators in StreamingResponse.
        for token_idx in model.generate_stream(
            idx,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p
        ):
            token_val = token_idx.item()
            text_chunk = tokenizer.decode([token_val])
            yield text_chunk
            await asyncio.sleep(0)

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
