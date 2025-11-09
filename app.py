# app.py
import uuid
import time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from prompt_optimizer import PromptOptimizer   # Your updated optimizer
from token_manager import TokenManager         # Your existing file
from semantic_utils import SemanticHelper      # Your existing file

# ------------------------------------------------------------------
# Model Pricing Logic for UI
# ------------------------------------------------------------------
MODEL_PRICING = {
    'gpt-4.1': 0.03, # Use the exact model names from your UI
    'gpt-4': 0.03,
    'gpt-5': 0.05,
    'claude:haiku': 0.024,
    'claude:opus': 0.03,
    'claude:sonet4.5': 0.028,
    'gemini': 0.004
}

def calculate_cost(tokens: int, model: str) -> float:
    """Calculate cost based on tokens and model pricing"""
    price_per_k = 0.03 # default
    # Find the key that best matches the model name
    for key, price in MODEL_PRICING.items():
        if key in model.lower():
            price_per_k = price
            break
    return (tokens / 1000) * price_per_k

# ------------------------------------------------------------------
# The SimpleCompressor (Tuned for 25 keywords)
# ------------------------------------------------------------------
class SimpleCompressor:
    def compress(self, text: str, rate: float = 0.5, preserve_keywords: Optional[List[str]] = None) -> str:
        """
        Compresses the prompt by *replacing* it with a keyword-only instruction.
        'rate' is now a fallback, as preserve_keywords is the main method.
        """
        if not preserve_keywords:
            # Fallback for no keywords: just truncate
            keep = max(40, int(len(text) * rate))
            return text[:keep].rstrip()

        kws = [str(k).strip() for k in preserve_keywords if k]
        if not kws:
            # Fallback if keywords are empty: just truncate
            keep = max(40, int(len(text) * rate))
            return text[:keep].rstrip()

        # Use up to 25 keywords (matching your optimizer's 'balanced' mode)
        kw_line = "Important keywords: " + ", ".join(kws[:25])
        return kw_line

# ------------------------------------------------------------------
# Pydantic Schemas (The "Matching Output System")
# ------------------------------------------------------------------
class OptimizeRequest(BaseModel):
    prompt: str
    target_model: str = "gpt-4"
    mode: str = "balanced"  # <-- CHANGED: Now accepts mode
    preserve_keywords: Optional[List[str]] = None
    async_job: bool = False

class OptimizeResponse(BaseModel):
    optimized_id: str
    optimized_prompt: Dict[str, Any]
    adapted_payload: Dict[str, Any]
    compression_rate: float
    cached: bool = False
    
    # --- NEW FIELDS FOR UI ---
    origTokens: int
    optTokens: int
    costSaved: float
    qualityScore: float

# --- Other models are unchanged ---
class GenerateRequest(BaseModel):
    optimized_id: Optional[str] = None
    optimized_prompt: Optional[Dict[str, Any]] = None
    model: str = "gpt-4"
    model_params: Optional[Dict[str, Any]] = None

class GenerateResponse(BaseModel):
    request_id: str
    response: str
    tokens_used: int
    model: str

class EvaluateRequest(BaseModel):
    original_prompt: str
    optimized_prompt: Dict[str, Any]
    model: str = "gpt-4"

class EvaluateResponse(BaseModel):
    quality_score: float
    tokens_original: int
    tokens_optimized: int
    token_savings_pct: float

# ------------------------------------------------------------------
# In-memory stores
# ------------------------------------------------------------------
optimized_store: Dict[str, Dict[str, Any]] = {}
metrics_store: Dict[str, Dict[str, Any]] = {}

# ------------------------------------------------------------------
# Core objects
# ------------------------------------------------------------------
compressor = SimpleCompressor()
prompt_optimizer = PromptOptimizer(compressor=compressor, token_model="gpt-4")
token_manager = TokenManager(gpt_model="gpt-4")
semantic_helper = SemanticHelper(model_name="all-MiniLM-L6-v2")

app = FastAPI(title="Prompt Optimizer API", version="0.3")

# ------------------------------------------------------------------
# Background task helpers (Unchanged)
# ------------------------------------------------------------------
def _run_optimize_and_store(
    prompt: str, target_model: str, mode: str, preserve_keywords: Optional[List[str]], opt_id: str
):
    try:
        # Pass 'mode' instead of 'compression_ratio'
        result = prompt_optimizer.optimize(prompt, target_model, mode, preserve_keywords)
        # ... (rest is the same, but we must add the new fields)
        
        orig_tokens = token_manager.count_tokens(prompt, target_model)
        opt_tokens = result["token_count"]
        tokens_saved = orig_tokens - opt_tokens
        cost_saved = calculate_cost(tokens_saved, target_model)
        quality_score = semantic_helper.similarity(prompt, result["optimized_prompt"]["flat"])

        optimized_store[opt_id] = {
            "id": opt_id,
            "original_prompt": prompt,
            "target_model": target_model,
            "mode": mode,
            "preserve_keywords": preserve_keywords,
            "result": result, # This holds the core optimizer output
            "created_at": time.time(),
            "status": "done",
            # Store the new UI fields as well
            "origTokens": orig_tokens,
            "optTokens": opt_tokens,
            "costSaved": cost_saved,
            "qualityScore": quality_score
        }
    except Exception as e:
        optimized_store[opt_id] = {
            "id": opt_id,
            "original_prompt": prompt,
            "status": "error",
            "error": str(e),
            "created_at": time.time(),
        }

# ------------------------------------------------------------------
# Model call stub (Unchanged)
# ------------------------------------------------------------------
def stub_model_call(adapted_payload: Dict[str, Any], model: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # ... (this function is unchanged)
    if "messages" in adapted_payload:
        content = " ".join(m.get("content", "") for m in adapted_payload["messages"])
    elif "input" in adapted_payload:
        content = str(adapted_payload["input"])
    elif "system" in adapted_payload and "messages" in adapted_payload:
        content = adapted_payload["system"] + " " + " ".join(m.get("content", "") for m in adapted_payload["messages"])
    else:
        content = str(adapted_payload)

    resp_text = f"[{model} stub reply] summary of: {content[:400]}"
    tokens_est = token_manager.count_tokens(resp_text, model)
    return {"text": resp_text, "tokens": tokens_est}

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest, background_tasks: BackgroundTasks):
    
    if req.async_job:
        opt_id = str(uuid.uuid4())
        optimized_store[opt_id] = {
            "id": opt_id,
            "status": "pending",
            "created_at": time.time(),
            "original_prompt": req.prompt,
        }
        background_tasks.add_task(
            _run_optimize_and_store,
            req.prompt,
            req.target_model,
            req.mode, # Pass mode
            req.preserve_keywords,
            opt_id,
        )
        # Return a temporary response
        return OptimizeResponse(
            optimized_id=opt_id,
            optimized_prompt={},
            adapted_payload={},
            compression_rate=0.0,
            cached=False,
            origTokens=0,
            optTokens=0,
            costSaved=0.0,
            qualityScore=0.0
        )

    # --- THIS IS THE UPDATED SYNCHRONOUS LOGIC ---
    
    # 1. Run the optimizer (pass mode)
    result = prompt_optimizer.optimize(
        req.prompt, 
        req.target_model, 
        mode=req.mode,  # Pass the new mode
        preserve_keywords=req.preserve_keywords
    )

    # 2. Get the new data for the UI
    orig_tokens = token_manager.count_tokens(req.prompt, req.target_model)
    opt_tokens = result["token_count"] # This is the optimized count
    
    # Calculate quality score
    quality_score = semantic_helper.similarity(
        req.prompt, 
        result["optimized_prompt"]["flat"]
    )
    
    tokens_saved = orig_tokens - opt_tokens
    cost_saved = calculate_cost(tokens_saved, req.target_model)

    opt_id = str(uuid.uuid4())
    
    # 3. Store all data in the in-memory store
    optimized_store[opt_id] = {
        "id": opt_id,
        "status": "done",
        "created_at": time.time(),
        "original_prompt": req.prompt,
        "result": result,
        "origTokens": orig_tokens,
        "optTokens": opt_tokens,
        "costSaved": cost_saved,
        "qualityScore": quality_score
    }

    # 4. Return the new response object
    return OptimizeResponse(
        optimized_id=opt_id,
        optimized_prompt=result["optimized_prompt"],
        adapted_payload=result["adapted_payload"],
        compression_rate=result["compression_rate"],
        cached=False,
        
        # --- NEW DATA FOR THE UI ---
        origTokens=orig_tokens,
        optTokens=opt_tokens,
        costSaved=cost_saved,
        qualityScore=round(quality_score, 2)
    )

# --- Other endpoints are unchanged ---

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    # ... (this endpoint is unchanged)
    if req.optimized_id:
        entry = optimized_store.get(req.optimized_id)
        if not entry:
            raise HTTPException(status_code=404, detail="optimized_id not found")
        if entry.get("status") != "done":
            raise HTTPException(status_code=409, detail=f"optimized prompt status: {entry.get('status')}")
        adapted = entry["result"]["adapted_payload"]
    elif req.optimized_prompt:
        adapted = prompt_optimizer.adapter.adapt(req.optimized_prompt, req.model)
    else:
        raise HTTPException(status_code=400, detail="Provide optimized_id or optimized_prompt")

    model_resp = stub_model_call(adapted, req.model, req.model_params)
    request_id = str(uuid.uuid4())
    metrics_store[request_id] = {
        "id": request_id,
        "model": req.model,
        "tokens_used": model_resp["tokens"],
        "timestamp": time.time(),
        "adapted_payload": adapted,
    }

    return GenerateResponse(
        request_id=request_id,
        response=model_resp["text"],
        tokens_used=model_resp["tokens"],
        model=req.model,
    )

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    # ... (this endpoint is unchanged)
    tokens_orig = token_manager.count_tokens(req.original_prompt, req.model)
    if "flat" in req.optimized_prompt and req.optimized_prompt["flat"]:
        optimized_flat = req.optimized_prompt["flat"]
    else:
        msgs = req.optimized_prompt.get("messages", [])
        optimized_flat = " ".join(m.get("content", "") for m in msgs)

    tokens_opt = token_manager.count_tokens(optimized_flat, req.model)
    token_savings_pct = 100.0 * (tokens_orig - tokens_opt) / tokens_orig if tokens_orig else 0.0

    # semantic similarity (0..1)
    quality = semantic_helper.similarity(req.original_prompt, optimized_flat)

    return EvaluateResponse(
        quality_score=quality,
        tokens_original=tokens_orig,
        tokens_optimized=tokens_opt,
        token_savings_pct=token_savings_pct,
    )

@app.get("/prompts/{opt_id}")
async def get_optimized(opt_id: str):
    # ... (this endpoint is unchanged)
    entry = optimized_store.get(opt_id)
    if not entry:
        raise HTTPException(status_code=404, detail="not found")
    return entry

@app.get("/metrics/{request_id}")
async def get_metrics(request_id: str):
    # ... (this endpoint is unchanged)
    entry = metrics_store.get(request_id)
    if not entry:
        raise HTTPException(status_code=404, detail="not found")
    return entry

# ------------------------------------------------------------------
# Main entrypoint for Render
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)