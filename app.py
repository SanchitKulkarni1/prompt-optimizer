# app.py
import uuid
import time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from prompt_optimizer import PromptOptimizer   # patched
from token_manager import TokenManager         # your existing file
from semantic_utils import SemanticHelper

# ------------------------------------------------------------------
# Improved SimpleCompressor — explicitly injects Important keywords
# ------------------------------------------------------------------

# class SimpleCompressor:
#     def compress(self, text: str, rate: float = 0.5, preserve_keywords: Optional[List[str]] = None) -> str:
#         """
#         Compresses the prompt by *replacing* it with a keyword-only instruction.
#         """
#         if not preserve_keywords:
#             # Fallback for no keywords: just truncate
#             keep = max(40, int(len(text) * rate))
#             return text[:keep].rstrip()

#         kws = [str(k).strip() for k in preserve_keywords if k]
#         if not kws:
#             # Fallback if keywords are empty: just truncate
#             keep = max(40, int(len(text) * rate))
#             return text[:keep].rstrip()

#         kw_line = "Important keywords: " + ", ".join(kws[:15])
#         return kw_line

# In app.py
class SimpleCompressor:
    def compress(self, text: str, rate: float = 0.5, preserve_keywords: Optional[List[str]] = None) -> str:
        """
        Compresses the prompt by *replacing* it with a keyword-only instruction.
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

        # THIS IS THE CRITICAL FIX:
        # Return *only* the keyword line.
        kw_line = "Important keywords: " + ", ".join(kws[:25])
        return kw_line

# ------------------------------------------------------------------
# Pydantic request/response schemas
# ------------------------------------------------------------------
class OptimizeRequest(BaseModel):
    prompt: str
    target_model: str = "gpt-4"
    compression_ratio: float = 0.5
    preserve_keywords: Optional[List[str]] = None
    async_job: bool = False

class OptimizeResponse(BaseModel):
    optimized_id: str
    optimized_prompt: Dict[str, Any]
    adapted_payload: Dict[str, Any]
    token_count: int
    compression_rate: float
    cached: bool = False

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

app = FastAPI(title="Prompt Optimizer API", version="0.2")

# ------------------------------------------------------------------
# Background task helpers
# ------------------------------------------------------------------
def _run_optimize_and_store(
    prompt: str, target_model: str, compression_ratio: float, preserve_keywords: Optional[List[str]], opt_id: str
):
    try:
        result = prompt_optimizer.optimize(prompt, target_model, compression_ratio, preserve_keywords)
        optimized_store[opt_id] = {
            "id": opt_id,
            "original_prompt": prompt,
            "target_model": target_model,
            "compression_ratio": compression_ratio,
            "preserve_keywords": preserve_keywords,
            "result": result,
            "created_at": time.time(),
            "status": "done",
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
# Model call stub (unchanged)
# ------------------------------------------------------------------
def stub_model_call(adapted_payload: Dict[str, Any], model: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            req.compression_ratio,
            req.preserve_keywords,
            opt_id,
        )
        return OptimizeResponse(
            optimized_id=opt_id,
            optimized_prompt={},
            adapted_payload={},
            token_count=0,
            compression_rate=0.0,
            cached=False,
        )

    result = prompt_optimizer.optimize(req.prompt, req.target_model, req.compression_ratio, req.preserve_keywords)
    opt_id = str(uuid.uuid4())
    optimized_store[opt_id] = {
        "id": opt_id,
        "status": "done",
        "created_at": time.time(),
        "original_prompt": req.prompt,
        "result": result,
    }

    return OptimizeResponse(
        optimized_id=opt_id,
        optimized_prompt=result["optimized_prompt"],
        adapted_payload=result["adapted_payload"],
        token_count=result["token_count"],
        compression_rate=result["compression_rate"],
        cached=False,
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
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
    entry = optimized_store.get(opt_id)
    if not entry:
        raise HTTPException(status_code=404, detail="not found")
    return entry

@app.get("/metrics/{request_id}")
async def get_metrics(request_id: str):
    entry = metrics_store.get(request_id)
    if not entry:
        raise HTTPException(status_code=404, detail="not found")
    return entry
