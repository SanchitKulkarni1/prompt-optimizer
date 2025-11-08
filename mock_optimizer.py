# mock_optimizer.py
# Simple mock optimizer + preview server for development/demo
# Run:
#   python3 -m venv venv
#   source venv/bin/activate
#   pip install flask
#   python mock_optimizer.py
#
# For production hosting (Render/Heroku) use gunicorn:
#   gunicorn mock_optimizer:app --bind 0.0.0.0:$PORT

from flask import Flask, request, jsonify
from datetime import datetime
import math

app = Flask(__name__)

# ---------- Demo dataset (6 items) ----------
DEMO_PACK = [
    {
        "id": "demo1",
        "category": "Flask API",
        "original_prompt": "I want you to act as a senior Python developer. Please write me a Flask backend that takes a POST request containing two numbers called \"a\" and \"b\", computes their sum, and returns the result in JSON. Also explain each line of code in detail and provide an example curl command that demonstrates sending the request and reading the response.",
        "original_tokens": 33,
        "candidates": [
            {"name":"conservative","prompt":"Role: Senior Python dev. Task: Create a Flask POST endpoint that accepts JSON {a, b}, returns {\"sum\": a+b}. Include brief line comments and one example curl. Output: code + short explanation. Max words: 140.","tokens":26,"quality":0.90},
            {"name":"balanced","prompt":"Role: Senior Python dev. Task: Create a Flask POST endpoint that accepts JSON {a, b}, returns {\"sum\": a+b}. Include brief line comments and one example curl. Output: code + short explanation. Max words: 120.","tokens":23,"quality":0.93},
            {"name":"aggressive","prompt":"Task: Flask POST endpoint: input {a,b}, return JSON {sum}. Provide concise code + 1-line comments + example curl. Output: JSON + code. Max words: 90.","tokens":20,"quality":0.86}
        ]
    },
    {
        "id": "demo2",
        "category": "React + Flask",
        "original_prompt": "Please give me a detailed plan for building a React frontend and Flask backend architecture. Include folder structure, key packages to install, commands for setup, an example API route, environment variables, and deployment steps for deploying to Render. Provide sample code snippets and brief explanation for each step.",
        "original_tokens": 34,
        "candidates":[
            {"name":"conservative","prompt":"Provide a compact React + Flask starter. Return JSON with keys: folders, packages, commands, example_api, env_vars, deploy_steps. Max words: 180.","tokens":26,"quality":0.88},
            {"name":"balanced","prompt":"Task: Provide a compact React + Flask starter. Return JSON with keys: folders, packages, commands, example_api, env_vars, deploy_steps. Each value short. Max words:150.","tokens":21,"quality":0.90},
            {"name":"aggressive","prompt":"Task: React+Flask starter. Output: JSON keys folders/packages/commands/env/deploy. Keep very short.","tokens":18,"quality":0.85}
        ]
    },
    {
        "id": "demo3",
        "category": "Instagram caption",
        "original_prompt": "Write a compelling Instagram caption for a Gen Z audience announcing a new feature that lets creators sell 30-second shoutouts. Keep it fun and include 20 relevant hashtags, a 1-line CTA, and two variations (short and long).",
        "original_tokens": 24,
        "candidates": [
            {"name":"conservative","prompt":"Task: Write 2 IG caption variations announcing 30s shoutouts. Include 1-line CTA and 12 hashtags. Output: short & long.","tokens":24,"quality":0.94},
            {"name":"balanced","prompt":"Task: Create 2 Instagram caption variations (short, long) announcing 30s shoutouts for creators. Include a 1-line CTA and 12 relevant hashtags. Output: JSON {short,long,cta,hashtags}. Max words: 80.","tokens":23,"quality":0.95},
            {"name":"aggressive","prompt":"Task: IG captions x2, 1-line CTA, 8 hashtags, very concise.","tokens":20,"quality":0.90}
        ]
    },
    {
        "id": "demo4",
        "category": "Resume feedback",
        "original_prompt": "I have a software engineering resume. Please analyze and give me specific improvements for bullet points, skill keywords to add for an ML engineer role, and a 2-line summary I can put at the top. Use examples and show before/after for two bullets.",
        "original_tokens": 30,
        "candidates":[
            {"name":"conservative","prompt":"Task: Analyze resume. Return suggested 2-line summary, 2 before/after bullets, and skills to add. Max words: 160.","tokens":26,"quality":0.89},
            {"name":"balanced","prompt":"Task: Analyze resume. Return JSON {summary_2line, bullets_before_after:[{before,after}], skills_to_add}. Provide concise examples. Max words: 120.","tokens":22,"quality":0.91},
            {"name":"aggressive","prompt":"Task: Resume micro-feedback: 2-line summary + 2 before/after bullets + 3 skills. Very concise.","tokens":19,"quality":0.86}
        ]
    },
    {
        "id": "demo5",
        "category": "Summarization",
        "original_prompt": "Summarize the following article into a 200-word executive summary, then provide 5 key takeaways in bullets with one-sentence explanations each. Keep the tone formal and use headings.",
        "original_tokens": 22,
        "candidates":[
            {"name":"conservative","prompt":"Task: Produce a 200-word executive summary and 5 bullet takeaways (one sentence each). Tone: formal. Output: JSON {summary, takeaways}. Max words: 220.","tokens":19,"quality":0.94},
            {"name":"balanced","prompt":"Task: 200-word executive summary + 5 key takeaways. Output: JSON {summary,takeaways}. Max words: 200.","tokens":18,"quality":0.92},
            {"name":"aggressive","prompt":"Task: Short executive summary (150 words) + 3 bullets. Max words: 150.","tokens":15,"quality":0.88}
        ]
    },
    {
        "id": "demo6",
        "category": "SQL query",
        "original_prompt": "Generate a SQL query that returns, for each user, the count of orders in the last 30 days, the total order value, and the average order value. Use appropriate JOINs between users and orders tables and alias columns clearly. Also explain edge cases and performance tips.",
        "original_tokens": 29,
        "candidates":[
            {"name":"conservative","prompt":"Task: Provide SQL query and 3 performance tips. Output: JSON {query,tips,edge_case}. Max words: 180.","tokens":26,"quality":0.90},
            {"name":"balanced","prompt":"Task: Provide SQL query: per-user count_orders_30d, total_value, avg_value. Include JOINs and final query only. Then provide 3 bullet performance tips and 1 edge-case note. Output: JSON {query, tips, edge_case}. Max words: 120.","tokens":25,"quality":0.92},
            {"name":"aggressive","prompt":"Task: SQL query only; plus 1 performance tip. Output succinct.","tokens":20,"quality":0.86}
        ]
    }
]

# utility
def find_demo_by_prompt_text(prompt_text):
    if not prompt_text:
        return None
    low = prompt_text.strip().lower()
    for d in DEMO_PACK:
        if d["original_prompt"].strip().lower() == low:
            return d
    return None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "time": datetime.utcnow().isoformat()})

@app.route("/optimize", methods=["POST"])
def optimize():
    payload = request.get_json(force=True) or {}
    prompt = payload.get("prompt", "").strip()
    model = payload.get("model", "gpt:4.1")
    mode = payload.get("mode", "balanced")
    threshold = float(payload.get("threshold", 0.8))
    demo_id = payload.get("demo_id", None)

    matched = None
    if demo_id:
        matched = next((d for d in DEMO_PACK if d["id"] == demo_id), None)
    else:
        matched = find_demo_by_prompt_text(prompt)

    if not matched:
        # fallback: simple transform
        words = prompt.split()
        orig_tokens = max(1, math.ceil(len(words) * 0.75))
        optimized_prompt = "Task: " + " ".join(words[:20]) + ("..." if len(words) > 20 else "") + " Output: concise."
        opt_tokens = max(1, int(orig_tokens * 0.6))
        chosen = {
            "name": "balanced",
            "prompt": optimized_prompt,
            "tokens": opt_tokens,
            "tokens_saved": orig_tokens - opt_tokens,
            "percent_saved": round((orig_tokens - opt_tokens) / max(1, orig_tokens) * 100, 6),
            "quality": 0.85,
            "est_cost_saved_usd": round(((orig_tokens - opt_tokens)/1000.0) * 0.03, 8),
            "flag_quality_low": False
        }
        return jsonify({"model": model, "mode": mode, "original_tokens": orig_tokens, "candidates":[chosen], "chosen": chosen})

    # Build response from matched demo
    orig_tokens = matched["original_tokens"]
    # pick candidate by mode
    candidate = next((c for c in matched["candidates"] if c["name"] == mode), matched["candidates"][1])
    chosen = dict(candidate)  # copy
    chosen["tokens_saved"] = orig_tokens - chosen["tokens"]
    chosen["percent_saved"] = round((chosen["tokens_saved"] / max(1, orig_tokens)) * 100, 6)
    chosen["est_cost_saved_usd"] = round((chosen["tokens_saved"] / 1000.0) * 0.03, 8)
    chosen["flag_quality_low"] = False if chosen.get("quality", 0) >= threshold else True

    return jsonify({"model": model, "mode": mode, "original_tokens": orig_tokens, "candidates": matched["candidates"], "chosen": chosen})

@app.route("/preview", methods=["POST"])
def preview():
    payload = request.get_json(force=True) or {}
    demo_id = payload.get("demo_id")
    chosen_prompt = payload.get("prompt", "")
    model = payload.get("model", "gpt:4.1")
    max_tokens = int(payload.get("max_tokens", 150))

    matched = None
    if demo_id:
        matched = next((d for d in DEMO_PACK if d["id"] == demo_id), None)

    if matched:
        # canned outputs mapped by category
        output_map = {
            "Flask API": "```python\nfrom flask import Flask, request, jsonify\napp = Flask(__name__)\n@app.route('/sum', methods=['POST'])\ndef sum_route():\n    data = request.get_json()\n    a = data.get('a',0)\n    b = data.get('b',0)\n    return jsonify({'sum': a + b})\n```",
            "React + Flask": '{"folders":["client/","server/"], "packages":["react","flask"], "commands":["npm install","pip install -r requirements.txt"] }',
            "Instagram caption": '{"short":"30s shoutout! Sell quick shoutouts and earn. CTA: DM to book!","long":"Creators can now sell 30-second shoutouts..."}',
            "Resume feedback": '{"summary_2line":"ML engineer with X yrs ...","bullets_before_after":[{"before":"Worked on models","after":"Designed and deployed ML pipelines..."}]}',
            "Summarization": '{"summary":"Executive summary text...","takeaways":["T1","T2","T3","T4","T5"]}',
            "SQL query": '{"query":"SELECT u.id, COUNT(o.id) AS cnt, SUM(o.value) AS total, AVG(o.value) AS avg FROM users u LEFT JOIN orders o ON ... GROUP BY u.id;","tips":["Index created_at","Use partitioning"],"edge_case":"NULL orders"}'
        }
        out_text = output_map.get(matched["category"], "Demo model output text.")
        prompt_tokens = matched["candidates"][1]["tokens"]
        output_tokens = min(max_tokens, 120)
        total = prompt_tokens + output_tokens
        usage = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "cost_usd": round((total / 1000.0) * 0.03, 6)
        }
        return jsonify({"model": model, "prompt": matched["candidates"][1]["prompt"], "output": out_text, "usage": usage})

    # fallback echo
    words = chosen_prompt.split()
    prompt_tokens = max(1, math.ceil(len(words) * 0.75))
    output_tokens = min(max_tokens, 80)
    total = prompt_tokens + output_tokens
    usage = {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens, "total_tokens": total, "cost_usd": round((total / 1000.0) * 0.03, 6)}
    out = "Simulated preview output: " + (" ".join(words[:40]) + ("..." if len(words) > 40 else ""))
    return jsonify({"model": model, "prompt": chosen_prompt, "output": out, "usage": usage})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
