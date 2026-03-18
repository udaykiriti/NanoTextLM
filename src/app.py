from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import torch
import os
import asyncio
from runtime import PROJECT_ROOT, load_inference_resources

# Setup
app = FastAPI(title="NanoTextLM API")
templates = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "src", "templates"))

# Global State
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = Field(default=1.0, ge=0.0)
    top_k: int | None = Field(default=None, gt=0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=150, ge=0)

def load_resources():
    global model, tokenizer
    if model is None:
        print(f"Loading NanoTextLM on {device}...")
        model, tokenizer, _, model_path, checkpoint_exists = load_inference_resources()
        if not checkpoint_exists:
            print("Warning: No checkpoint found. Using random weights.")
        else:
            print(f"Loaded model from {model_path}")

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
            top_k=req.top_k,
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
