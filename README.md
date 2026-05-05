# LLM-Anpr-weighbridge

AI-powered ANPR system for weighbridge automation. Captures live H265 RTSP camera feed via GStreamer on Raspberry Pi, extracts vehicle license plates using LLaMA vision (Groq API), and displays real-time detections on a live dashboard with WebSocket updates, MJPEG stream, and SQLite logging.

## Architecture

```
RTSP Camera (H265)
      │
      ▼
GStreamer Pipeline  (detect.py)
rtspsrc → depay → parse → decode → convert → appsink
                                                  │
                                         latest_frame (RGB)
                                                  │
                                    frame_processor thread
                                      (every 15 seconds)
                                                  │
                                    Groq API — LLaMA Vision
                                                  │
                               ┌──────────────────┴──────────────────┐
                           SQLite DB                           event_queue
                        (detections.db)                               │
                                                           WebSocket /ws/live
                                                                      │
                                                             Dashboard 🖥️
```

## Stack

| Component | Technology |
|---|---|
| Camera | H265 RTSP via GStreamer |
| OCR / Vision | LLaMA 4 Scout (Groq API — free tier) |
| Backend | FastAPI + Uvicorn |
| Realtime | WebSocket |
| Camera stream | MJPEG |
| Storage | SQLite |
| Hardware | Raspberry Pi 5 |

## Project Structure

```
├── detect.py          # GStreamer capture + Groq OCR + plate detection
├── server.py          # FastAPI backend + WebSocket + MJPEG stream
├── web/static/
│   └── index.html     # ANPR Weighbridge dashboard UI
├── requirements.txt   # Python dependencies
└── ai-engineering-hub/
    └── llama-ocr/     # Reference implementation
```

## Setup

### 1. System dependencies
```bash
sudo apt install python3-gi python3-gst-1.0 gstreamer1.0-plugins-good
sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-libav
```

### 2. Python environment
```bash
python3 -m venv llmenv --system-site-packages
source llmenv/bin/activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com)
```

### 4. Run
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://<raspberry-pi-ip>:8000` in your browser.

## Configuration

Edit these values in `detect.py`:

| Variable | Default | Description |
|---|---|---|
| `RTSP_URL` | `rtsp://...` | Your camera RTSP URL |
| `OCR_INTERVAL_S` | `15` | Seconds between OCR calls |
| `COOLDOWN_S` | `30` | Min gap before re-logging same plate |
| `OCR_RESIZE_W` | `640` | Frame width sent to model |

## License

MIT — see [ai-engineering-hub/LICENSE](ai-engineering-hub/LICENSE)
