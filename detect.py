"""
GStreamer RTSP capture → Groq LLaMA-vision OCR → license plate detection.

Flow:
  rtspsrc → rtph265depay → h265parse → avdec_h265 → videoconvert → appsink
                                                                        ↓
                                                           frame_processor thread
                                                                        ↓
                                                      Groq API  llama-3.2-11b-vision
                                                                        ↓
                                                           SQLite  +  event_queue

Run standalone:  python detect.py
Import:          from detect import start_pipeline, event_queue, latest_frame, frame_lock

Requires:  GROQ_API_KEY environment variable  (free at console.groq.com)
"""

import base64
import io
import os
import queue
import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
from groq import Groq
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
RTSP_URL       = "rtsp://192.168.96.31:554/rtsp/streaming?channel=01&subtype=A"
DB_PATH        = str(Path(__file__).parent / "detections.db")
OCR_MODEL      = "meta-llama/llama-4-scout-17b-16e-instruct"   # best free vision model on Groq
OCR_INTERVAL_S = 15      # seconds between OCR attempts
OCR_RESIZE_W   = 640     # shrink frame to this width before OCR
COOLDOWN_S     = 30      # min gap (seconds) before re-logging the same plate

_groq_client: Groq | None = None

def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")
        _groq_client = Groq(api_key=key)
    return _groq_client

# ── Shared state ───────────────────────────────────────────────────────────────
latest_frame: np.ndarray | None = None   # RGB uint8 from GStreamer
frame_lock   = threading.Lock()
event_queue  = queue.Queue()             # dict events consumed by server.py

# ── Database ───────────────────────────────────────────────────────────────────
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            plate        TEXT    NOT NULL,
            confidence   REAL,
            gross_weight REAL,
            net_weight   REAL,
            detected_at  TEXT    NOT NULL,
            modbus_sent  INTEGER DEFAULT 0,
            webhook_sent INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def save_detection(plate: str, confidence: float) -> str:
    ts = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO detections (plate, confidence, detected_at) VALUES (?,?,?)",
        (plate, confidence, ts),
    )
    conn.commit()
    conn.close()
    return ts

# ── OCR ────────────────────────────────────────────────────────────────────────
_OCR_PROMPT = (
    "You are a license plate recognition system. "
    "Examine this image carefully.\n"
    "If a vehicle license plate is clearly visible, reply in EXACTLY this format:\n"
    "PLATE: <plate_text>\n"
    "CONFIDENCE: <high|medium|low>\n\n"
    "If no license plate is visible, reply with exactly: NO_PLATE"
)

def _encode_jpeg(frame_rgb: np.ndarray) -> bytes:
    h, w = frame_rgb.shape[:2]
    if w > OCR_RESIZE_W:
        scale = OCR_RESIZE_W / w
        img = Image.fromarray(frame_rgb).resize(
            (OCR_RESIZE_W, int(h * scale)), Image.LANCZOS
        )
    else:
        img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()

def run_ocr(frame_rgb: np.ndarray) -> tuple[str | None, float]:
    """
    Pass a frame through Groq LLaMA vision OCR.
    Returns (plate_text, confidence 0.0-1.0) or (None, 0.0) if no plate found.
    """
    jpeg = _encode_jpeg(frame_rgb)
    b64  = base64.b64encode(jpeg).decode("utf-8")
    try:
        resp = _get_groq().chat.completions.create(
            model=OCR_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",      "text": _OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=100,
        )
        text = resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[OCR] Error: {exc}")
        return None, 0.0

    if "NO_PLATE" in text.upper():
        print("[OCR] No plate detected")
        return None, 0.0

    m_plate = re.search(r"PLATE:\s*([A-Z0-9 \-]+)", text, re.IGNORECASE)
    m_conf  = re.search(r"CONFIDENCE:\s*(high|medium|low)", text, re.IGNORECASE)

    if not m_plate:
        print(f"[OCR] Unexpected response: {text!r}")
        return None, 0.0

    plate    = m_plate.group(1).strip().upper()
    conf_str = m_conf.group(1).lower() if m_conf else "medium"
    conf     = {"high": 0.92, "medium": 0.68, "low": 0.42}[conf_str]

    print(f"[OCR] Plate: {plate}  Confidence: {conf_str} ({conf:.0%})")
    return plate, conf

# ── Frame processor thread ─────────────────────────────────────────────────────
_plate_last_seen: dict[str, float] = {}

def _frame_processor() -> None:
    last_ocr = 0.0
    while True:
        time.sleep(0.1)
        now = time.time()
        if now - last_ocr < OCR_INTERVAL_S:
            continue

        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            continue

        last_ocr = now
        plate, conf = run_ocr(frame)
        if not plate:
            continue

        if now - _plate_last_seen.get(plate, 0) < COOLDOWN_S:
            print(f"[DETECT] Cooldown active for {plate}")
            continue

        _plate_last_seen[plate] = now
        ts = save_detection(plate, conf)
        print(f"[DETECT] Saved: {plate} @ {ts}")

        event_queue.put({
            "type": "detection",
            "data": {
                "plate":        plate,
                "confidence":   round(conf * 100, 1),   # 0-100 for dashboard WS
                "gross_weight": None,
                "net_weight":   None,
                "detected_at":  ts,
                "modbus_sent":  False,
                "webhook_sent": False,
            },
        })

# ── GStreamer pipeline ─────────────────────────────────────────────────────────
Gst.init(None)

def _build_pipeline() -> Gst.Pipeline:
    pipe   = Gst.Pipeline.new("anpr")
    source = Gst.ElementFactory.make("rtspsrc",      "source")
    depay  = Gst.ElementFactory.make("rtph265depay", "depay")
    parser = Gst.ElementFactory.make("h265parse",    "parser")
    decode = Gst.ElementFactory.make("avdec_h265",   "decoder")
    conv   = Gst.ElementFactory.make("videoconvert", "convert")
    sink   = Gst.ElementFactory.make("appsink",      "sink")

    source.set_property("location", RTSP_URL)
    source.set_property("latency",  200)

    sink.set_property("emit-signals", True)
    sink.set_property("max-buffers",  1)
    sink.set_property("drop",         True)
    sink.set_property("sync",         False)
    sink.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))

    for el in [source, depay, parser, decode, conv, sink]:
        pipe.add(el)

    def on_pad(src, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        st   = caps.get_structure(0)
        if st.get_string("media") != "video" or st.get_string("encoding-name") != "H265":
            print(f"[PAD] Skipping pad: media={st.get_string('media')}")
            return
        sp = depay.get_static_pad("sink")
        if not sp.is_linked():
            ret = pad.link(sp)
            print(f"[PAD] Linked H265 pad: {ret}")

    source.connect("pad-added", on_pad)
    depay.link(parser)
    parser.link(decode)
    decode.link(conv)
    conv.link(sink)

    def on_sample(appsink):
        global latest_frame
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        st  = sample.get_caps().get_structure(0)
        w   = st.get_value("width")
        h   = st.get_value("height")

        ok, mi = buf.map(Gst.MapFlags.READ)
        if ok:
            f = np.frombuffer(mi.data, np.uint8).reshape(h, w, 3).copy()
            buf.unmap(mi)
            with frame_lock:
                latest_frame = f

        return Gst.FlowReturn.OK

    sink.connect("new-sample", on_sample)
    return pipe

def start_pipeline() -> tuple[Gst.Pipeline, GLib.MainLoop]:
    """
    Initialise DB, build GStreamer pipeline, start GLib loop + frame processor.
    Non-blocking — returns (pipeline, glib_loop).
    """
    init_db()
    pipe = _build_pipeline()
    loop = GLib.MainLoop()

    def on_bus(bus, msg):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[GST] Error: {err}\n{dbg}")
            loop.quit()
        elif t == Gst.MessageType.EOS:
            print("[GST] End of stream")
            loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED and msg.src == pipe:
            _, new, _ = msg.parse_state_changed()
            print(f"[GST] State → {new.value_nick}")

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus)
    pipe.set_state(Gst.State.PLAYING)

    threading.Thread(target=loop.run,         daemon=True, name="glib-loop").start()
    threading.Thread(target=_frame_processor, daemon=True, name="frame-proc").start()

    print(f"[DETECT] Pipeline started  RTSP={RTSP_URL}")
    return pipe, loop


# ── Standalone entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    pipe, loop = start_pipeline()
    print("Running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.set_state(Gst.State.NULL)
        loop.quit()
