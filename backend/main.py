from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import asyncio

from neuvera_ai.llm_explainer import ollama_explain_session
from neuvera_ai.service import (
    run_full_analysis,
    run_movement_analysis_from_video,
    run_tremor_analysis_from_video,
    run_voice_analysis_from_file,
)

from . import db

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "uploads"
FRONTEND_DIR = ROOT / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Neuvera MVP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


def _save_upload(file: UploadFile, suffix_default: str) -> Path:
    suffix = Path(file.filename or "").suffix or suffix_default
    out = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with out.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)
    return out


@app.post("/analyze/voice")
async def analyze_voice(file: UploadFile = File(...)):
    path = _save_upload(file, ".webm")
    try:
        result = run_voice_analysis_from_file(path)
    except Exception as e:
        raise HTTPException(500, f"voice analysis failed: {e}")
    sid = db.save_result("voice", result)
    result["session_id"] = sid
    return result


@app.post("/analyze/tremor")
async def analyze_tremor(file: UploadFile = File(...)):
    path = _save_upload(file, ".webm")
    try:
        result = run_tremor_analysis_from_video(path)
    except Exception as e:
        raise HTTPException(500, f"tremor analysis failed: {e}")
    sid = db.save_result("tremor", result)
    result["session_id"] = sid
    return result


@app.post("/analyze/movement")
async def analyze_movement(file: UploadFile = File(...)):
    path = _save_upload(file, ".webm")
    try:
        result = run_movement_analysis_from_video(path)
    except Exception as e:
        raise HTTPException(500, f"movement analysis failed: {e}")
    sid = db.save_result("movement", result)
    result["session_id"] = sid
    return result


@app.post("/analyze/session")
async def analyze_session(
    voice: UploadFile | None = File(None),
    hand: UploadFile | None = File(None),
    movement: UploadFile | None = File(None),
):
    if not any([voice, hand, movement]):
        raise HTTPException(400, "at least one file required")
    audio_path = _save_upload(voice, ".webm") if voice else None
    hand_path = _save_upload(hand, ".webm") if hand else None
    move_path = _save_upload(movement, ".webm") if movement else None
    try:
        result = run_full_analysis(
            audio_path=audio_path,
            hand_video_path=hand_path,
            movement_video_path=move_path,
        )
    except Exception as e:
        raise HTTPException(500, f"session analysis failed: {e}")
    try:
        explanation = await asyncio.to_thread(ollama_explain_session, result)
        if explanation:
            result["llm_explanation"] = explanation
    except Exception:
        pass
    sid = db.save_result("session", result)
    result["session_id"] = sid
    return result


@app.get("/history")
def history(user_id: str = "demo-user", limit: int = 50):
    return {"items": db.list_history(user_id=user_id, limit=limit)}


@app.get("/history/{session_id}")
def history_item(session_id: int):
    row = db.get_result(session_id)
    if not row:
        raise HTTPException(404, "not found")
    return row


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve frontend at /
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/{page}.html")
    def page(page: str):
        f = FRONTEND_DIR / f"{page}.html"
        if not f.exists():
            raise HTTPException(404)
        return FileResponse(f)
