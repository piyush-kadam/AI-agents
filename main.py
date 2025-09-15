# main.py
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
from google.api_core import exceptions as google_exceptions
import re
from pathlib import Path
from datetime import datetime
import uuid
import logging

# ------------------ setup ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agents-pipeline")

# load environment variables from .env (if present)
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    logger.warning("GEMINI_API_KEY not set in .env (calls to Gemini will fail)")
else:
    genai.configure(api_key=GEMINI_KEY)

app = FastAPI(title="Stratify AI Backend - Pipeline")

# outputs directory (for logs, saved JSON results, optional static files)
OUTPUTS_DIR = Path("./outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# serve outputs as static files at /outputs (useful when deployed)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# ------------------ CORS ------------------
# Allow origins from env var CORS_ORIGINS (comma-separated) or allow all (*) by default.
cors_origins = os.getenv("CORS_ORIGINS", "*")
if cors_origins.strip() == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ models ------------------
class IdeaRequest(BaseModel):
    idea: str

class IdeaResponse(BaseModel):
    problem: str
    solution: str
    target_audience: str
    why_now: str

class MVPRequest(BaseModel):
    problem: str
    solution: str
    target_audience: str
    why_now: str

class MVPResponse(BaseModel):
    mvp_flow: str

class TechStackRequest(BaseModel):
    mvp_flow: str

class TechStackResponse(BaseModel):
    tech_stack: str
    implementation_plan: str

# ------------------ constants ------------------
FANTASY_KEYWORDS = {
    "invisible", "teleport", "time travel", "magic", "fly", "superpower", "telepathy",
    "teleportation", "warp", "resurrect", "immortality"
}

# ------------------ helpers ------------------
def clean_text(text: str) -> str:
    """Strip common code fences and return trimmed text."""
    if text is None:
        return ""
    text = re.sub(r"^```[\w-]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()

import concurrent.futures

def generate_with_fallback(prompt: str, timeout_sec: int = 60):
    """
    Call Gemini models with fallback (pro -> flash).
    Returns tuple (text, model_name).
    Raises RuntimeError if all models fail or timeout.
    """
    for model_name in ("gemini-1.5-pro", "gemini-1.5-flash"):
        try:
            logger.info(f"Calling model: {model_name}")
            model = genai.GenerativeModel(model_name)

            # run with timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(model.generate_content, prompt)
                response = future.result(timeout=timeout_sec)

            text = clean_text(response.text)
            logger.info(f"Model {model_name} returned {len(text)} chars")
            return text, model_name

        except concurrent.futures.TimeoutError:
            logger.error(f"Model {model_name} timed out after {timeout_sec}s")
            if model_name == "gemini-1.5-flash":
                raise RuntimeError("All models timed out")
            continue
        except google_exceptions.ResourceExhausted:
            logger.warning(f"{model_name} quota exhausted — trying fallback")
            continue
        except Exception as e:
            logger.exception(f"Model {model_name} error: {e}")
            if model_name == "gemini-1.5-flash":
                raise
            continue

    raise RuntimeError("All models failed")


def parse_json_keys(text: str, keys):
    """
    Try JSON parse. If that fails, use a regex extraction fallback for each key.
    """
    try:
        data = json.loads(text)
        for k in keys:
            if k not in data:
                data[k] = "Parsing failed"
        return data
    except Exception:
        data = {}
        for k in keys:
            m = re.search(f'"{k}"\\s*:\\s*"([\\s\\S]*?)"', text)
            data[k] = m.group(1).strip() if m else "Parsing failed"
        return data

def is_fantasy_idea(idea: str) -> bool:
    lower = idea.lower()
    return any(k in lower for k in FANTASY_KEYWORDS)

# ------------------ internal agents (core prompts) ------------------
def refine_idea_internal(idea: str):
    prompt = f"""
You are a startup mentor. Take this rough idea and refine it into a structured startup format.

Idea: {idea}

IMPORTANT:
1) If the idea is impossible, physically implausible, fantasy, or clearly unrealistic, respond ONLY with this exact JSON:
{{"problem":"Idea not feasible or unclear","solution":"Idea cannot be implemented realistically","target_audience":"N/A","why_now":"N/A"}}

2) Otherwise respond ONLY in valid JSON with these keys:
{{"problem":"...","solution":"...","target_audience":"...","why_now":"..."}}
"""
    text, model_used = generate_with_fallback(prompt)
    data = parse_json_keys(text, ["problem", "solution", "target_audience", "why_now"])
    return data, text, model_used

def generate_mvp_internal(refined: dict):
    prompt = f"""
You are a startup mentor. Given a refined idea, generate a clear, prioritized MVP feature flow (JSON).

Refined idea:
Problem: {refined.get('problem')}
Solution: {refined.get('solution')}
Target Audience: {refined.get('target_audience')}
Why Now: {refined.get('why_now')}

Respond ONLY in JSON:
{{
  "mvp_flow": [ 
     {{ "step":1, "feature":"...", "description":"..." }},
     ...
  ]
}}
"""
    text, model_used = generate_with_fallback(prompt)
    try:
        parsed = json.loads(text)
        if "mvp_flow" not in parsed:
            parsed = {"mvp_flow": text}
    except Exception:
        parsed = {"mvp_flow": text}
    return parsed, text, model_used

def generate_tech_stack_internal(mvp_flow_text: str):
    prompt = f"""
You are a tech architect. Given an MVP flow (JSON or text), recommend a detailed tech stack and a step-by-step implementation plan.

MVP Flow:
{mvp_flow_text}

Respond ONLY in JSON:
{{
  "tech_stack": {{ "frontend":"", "backend":"", "database":"", "cloud":"", "other":[] }},
  "implementation_plan": [ {{ "step":1, "title":"", "description":"", "tasks":[...] }}, ... ]
}}
"""
    text, model_used = generate_with_fallback(prompt)
    try:
        parsed = json.loads(text)
        if "tech_stack" not in parsed or "implementation_plan" not in parsed:
            parsed = {"tech_stack": parsed, "implementation_plan": parsed}
    except Exception:
        parsed = {"tech_stack": text, "implementation_plan": text}
    return parsed, text, model_used

# ------------------ endpoints ------------------
@app.get("/")
async def health():
    return {"status": "ok", "message": "Stratify AI Backend is running."}

@app.post("/refine_idea", response_model=IdeaResponse)
async def refine_idea(req: IdeaRequest):
    """
    Refine a rough idea into problem/solution/target_audience/why_now
    """
    if is_fantasy_idea(req.idea):
        return {
            "problem": "Idea not feasible or unclear",
            "solution": "Idea cannot be implemented realistically",
            "target_audience": "N/A",
            "why_now": "N/A"
        }
    data, raw, model_used = refine_idea_internal(req.idea)
    return data

@app.post("/generate_mvp", response_model=MVPResponse)
async def generate_mvp(req: MVPRequest):
    parsed, raw, model_used = generate_mvp_internal({
        "problem": req.problem,
        "solution": req.solution,
        "target_audience": req.target_audience,
        "why_now": req.why_now
    })
    return parsed

@app.post("/generate_tech_stack", response_model=TechStackResponse)
async def generate_tech_stack(req: TechStackRequest):
    parsed, raw, model_used = generate_tech_stack_internal(req.mvp_flow)
    tech = parsed.get("tech_stack", parsed)
    impl = parsed.get("implementation_plan", parsed)
    # Return JSON strings so the FastAPI response_model matches (string fields)
    return {"tech_stack": json.dumps(tech), "implementation_plan": json.dumps(impl)}

@app.post("/run_pipeline")
async def run_pipeline(req: IdeaRequest, request: Request):
    """
    Runs the pipeline:
      1) refine idea
      2) generate mvp
      3) generate tech stack
    Saves outputs JSON to ./outputs/<run_id>.json and returns combined object.
    """
    idea = req.idea.strip()
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.utcnow().isoformat()

    # quick deterministic fantasy check
    if is_fantasy_idea(idea):
        result = {
            "id": run_id,
            "timestamp": timestamp,
            "status": "rejected",
            "reason": "idea flagged as fantasy/impossible",
            "idea": idea,
            "refined": {
                "problem": "Idea not feasible or unclear",
                "solution": "Idea cannot be implemented realistically",
                "target_audience": "N/A",
                "why_now": "N/A"
            }
        }
        out_file = OUTPUTS_DIR / f"{run_id}.json"
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    # 1) refine
    try:
        refined, raw_refine_text, model_refine = refine_idea_internal(idea)
    except Exception as e:
        logger.exception("Refine step failed")
        return {"error": f"Refine step failed: {str(e)}"}

    if refined.get("problem") == "Idea not feasible or unclear":
        result = {
            "id": run_id,
            "timestamp": timestamp,
            "status": "rejected_by_model",
            "idea": idea,
            "refined": refined,
            "raw_refine": raw_refine_text,
            "model_used_refine": model_refine
        }
        out_file = OUTPUTS_DIR / f"{run_id}.json"
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    # 2) mvp
    try:
        mvp_parsed, raw_mvp_text, model_mvp = generate_mvp_internal(refined)
    except Exception as e:
        logger.exception("MVP generation failed")
        return {"error": f"MVP generation failed: {str(e)}"}

    # 3) tech stack + implementation plan
    try:
        tech_parsed, raw_tech_text, model_tech = generate_tech_stack_internal(json.dumps(mvp_parsed))
    except Exception as e:
        logger.exception("Tech stack generation failed")
        return {"error": f"Tech stack generation failed: {str(e)}"}

    # Build output object
    out_obj = {
        "id": run_id,
        "timestamp": timestamp,
        "idea": idea,
        "refined": refined,
        "raw_refine": raw_refine_text,
        "model_refine": model_refine,
        "mvp": mvp_parsed,
        "raw_mvp": raw_mvp_text,
        "model_mvp": model_mvp,
        "tech_stack": tech_parsed,
        "raw_tech": raw_tech_text,
        "model_tech": model_tech
    }

    # save
    out_file = OUTPUTS_DIR / f"{run_id}.json"
    out_file.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")

    return out_obj

# ------------------ run server (for local dev / simple deployment) ------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    # Point uvicorn to this module's app: main:app
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
