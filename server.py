"""
FastAPI backend for the ANPR Weighbridge dashboard.

Endpoints:
  GET  /api/stream        → MJPEG live camera feed
  GET  /api/stats         → today/total/unique KPIs
  GET  /api/detections    → detection history (SQLite)
  WS   /ws/live           → real-time detection events
  GET  /                  → dashboard (web/static/index.html)

Run:
  source llmenv/bin/activate
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import io
import json
import os
import queue
import sqlite3
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

# load .env if present (so GROQ_API_KEY is available when detect.py is imported)
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip()

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import detect

# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    detect._groq_client = None   # force re-init with freshly loaded key
    detect.start_pipeline()
    relay = asyncio.create_task(_relay_loop())
    print("[SERVER] Ready → http://0.0.0.0:8000")
    yield
    relay.cancel()

app    = FastAPI(lifespan=lifespan)
STATIC = Path(__file__).parent / "web" / "static"

# ── WebSocket broadcast ────────────────────────────────────────────────────────
_clients: set[WebSocket] = set()

async def _broadcast(msg: dict) -> None:
    text = json.dumps(msg)
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)

async def _relay_loop() -> None:
    """Forward events from detect.event_queue to all WebSocket clients."""
    while True:
        try:
            evt = detect.event_queue.get_nowait()
            await _broadcast(evt)
        except queue.Empty:
            await asyncio.sleep(0.1)

# ── MJPEG stream ───────────────────────────────────────────────────────────────
def _make_placeholder() -> bytes:
    img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()

_PLACEHOLDER = _make_placeholder()

async def _mjpeg_gen():
    while True:
        with detect.frame_lock:
            frame = detect.latest_frame.copy() if detect.latest_frame is not None else None

        if frame is not None:
            img = Image.fromarray(frame)   # frame is RGB from GStreamer
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=75)
            jpeg = buf.getvalue()
        else:
            jpeg = _PLACEHOLDER

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg
            + b"\r\n"
        )
        await asyncio.sleep(1 / 10)   # 10 fps

@app.get("/api/stream")
async def stream():
    return StreamingResponse(
        _mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

# ── Stats ──────────────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def stats():
    today = date.today().isoformat()
    conn  = sqlite3.connect(detect.DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE date(detected_at) = ?)    AS today_detections,
            COUNT(*)                                          AS total_detections,
            COUNT(DISTINCT plate) FILTER
                (WHERE date(detected_at) = ?)                 AS unique_plates_today,
            0                                                 AS open_weigh_events,
            (SELECT plate       FROM detections ORDER BY id DESC LIMIT 1) AS last_plate,
            (SELECT detected_at FROM detections ORDER BY id DESC LIMIT 1) AS last_detected_at
        FROM detections
        """,
        (today, today),
    ).fetchone()
    conn.close()
    return JSONResponse(dict(row) if row else {})

# ── Detection history ──────────────────────────────────────────────────────────
@app.get("/api/detections")
async def detections(limit: int = 100):
    conn = sqlite3.connect(detect.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    # confidence stored as 0.0-1.0; dashboard JS does Math.round(d.confidence * 100)
    data = []
    for r in rows:
        d = dict(r)
        if d["confidence"] is not None and d["confidence"] > 1:
            d["confidence"] = d["confidence"] / 100.0   # normalise if stored as %
        data.append(d)
    return JSONResponse(data)

# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        while True:
            await ws.receive_text()   # keep-alive; client doesn't send anything
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)

# ── Static dashboard — mount last so API routes take priority ──────────────────
app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
