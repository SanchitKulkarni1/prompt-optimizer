# ✂️ Prompt Optimizer

A FastAPI service that **compresses LLM prompts without losing their meaning**, then reports the token and cost savings. It extracts the key concepts with KeyBERT, compresses the rest of the prompt around them, and adapts the result to the target model's format.

## How it works

1. **Analyse:** KeyBERT (`all-MiniLM-L6-v2`) extracts the keywords that must survive. The mode controls how many:
   - `conservative`: more keywords kept, lighter compression
   - `balanced`: the default
   - `aggressive`: fewer keywords, heavier compression
2. **Compress:** sentences are ranked and trimmed while the keywords are preserved.
3. **Adapt:** `ModelAdapter` reshapes the optimised prompt for the target model (chat-message format for GPT, system-prompt placement for Claude).
4. **Measure:** `TokenManager` counts tokens before and after with `tiktoken` and estimates the cost saved for each model. Sentence-transformer similarity checks that the compressed prompt still means the same thing.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/optimize` | Compress a prompt → optimised prompt, token counts, savings |
| `POST` | `/generate` | Adapt the optimised prompt for a target model and run it (model call is currently a stub) |
| `POST` | `/evaluate` | Compare the original vs. optimised output |
| `GET` | `/prompts/{opt_id}` | Fetch a stored optimisation |
| `GET` | `/metrics/{request_id}` | Token and cost metrics for a request |

## Tech stack

Python · FastAPI · KeyBERT · sentence-transformers · tiktoken · Pydantic

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
# open http://localhost:8000/docs
```
