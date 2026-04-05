from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from sign_language_app.web.analyzer import WebGestureAnalyzer


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT / "models" / "asl_model.pkl")

app = FastAPI(title="ASL Recognition Web App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = WebGestureAnalyzer(MODEL_PATH)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ASL Recognition</title>
  <style>
    :root {
      --bg: #f5f5f7;
      --panel: rgba(255,255,255,0.78);
      --panel-border: rgba(0,0,0,0.08);
      --text: #111114;
      --muted: rgba(17,17,20,0.56);
      --accent: #0071e3;
      --good: #34c759;
      --shadow: 0 20px 60px rgba(0,0,0,0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.9), transparent 35%),
        linear-gradient(180deg, #fbfbfd 0%, #eef1f6 100%);
      min-height: 100vh;
    }
    .shell {
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 20px 36px;
    }
    .hero {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 24px;
      margin-bottom: 18px;
    }
    .brand {
      font-size: 13px;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }
    h1 {
      margin: 0;
      font-size: clamp(38px, 5vw, 64px);
      line-height: 0.98;
      letter-spacing: -0.05em;
    }
    .sub {
      margin-top: 10px;
      max-width: 620px;
      font-size: 18px;
      line-height: 1.45;
      color: var(--muted);
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 12px 16px;
      border-radius: 999px;
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--panel-border);
      box-shadow: var(--shadow);
      font-size: 14px;
      white-space: nowrap;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--good);
      box-shadow: 0 0 0 6px rgba(52,199,89,0.14);
    }
    .grid {
      display: grid;
      grid-template-columns: 1.4fr .9fr;
      gap: 18px;
      margin-top: 22px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
    }
    .video-card { padding: 18px; }
    .camera-wrap {
      position: relative;
      aspect-ratio: 16 / 10;
      border-radius: 22px;
      overflow: hidden;
      background: #0b0b0d;
    }
    video, canvas.overlay {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: scaleX(-1);
    }
    canvas.overlay { pointer-events: none; }
    .chips {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }
    .chip {
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255,255,255,0.85);
      border: 1px solid var(--panel-border);
      font-size: 13px;
      color: var(--muted);
    }
    .side {
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .panel-title {
      font-size: 12px;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .prediction {
      font-size: clamp(56px, 8vw, 92px);
      font-weight: 700;
      letter-spacing: -0.08em;
      line-height: .92;
    }
    .confidence {
      margin-top: 10px;
      color: var(--muted);
      font-size: 16px;
    }
    .sentence {
      min-height: 100px;
      padding: 16px;
      border-radius: 22px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--panel-border);
      font-size: 22px;
      line-height: 1.5;
      letter-spacing: -0.02em;
    }
    .controls {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 6px;
    }
    button {
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      cursor: pointer;
      transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
    }
    button:hover { transform: translateY(-1px); }
    .primary {
      color: white;
      background: var(--accent);
      box-shadow: 0 14px 30px rgba(0,113,227,0.24);
    }
    .secondary {
      background: rgba(255,255,255,0.85);
      color: var(--text);
      border: 1px solid var(--panel-border);
    }
    .footer-note {
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    @media (max-width: 960px) {
      .grid { grid-template-columns: 1fr; }
      .hero { align-items: start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div>
        <div class="brand">Sign Language Recognition</div>
        <h1>Clean, real-time ASL recognition.</h1>
        <div class="sub">A simple browser-based interface with the same MediaPipe landmark pipeline and CNN classifier as the desktop app.</div>
      </div>
      <div class="status"><span class="dot"></span><span id="statusText">Ready</span></div>
    </div>

    <div class="grid">
      <div class="card video-card">
        <div class="camera-wrap">
          <video id="video" autoplay playsinline muted></video>
          <canvas id="overlay" class="overlay"></canvas>
        </div>
        <div class="chips">
          <div class="chip">Live camera</div>
          <div class="chip">Apple-like minimal UI</div>
          <div class="chip">CNN inference</div>
        </div>
      </div>

      <div class="card side">
        <div>
          <div class="panel-title">Current letter</div>
          <div id="prediction" class="prediction">--</div>
          <div id="confidence" class="confidence">Confidence: 0%</div>
        </div>

        <div>
          <div class="panel-title">Sentence</div>
          <div id="sentence" class="sentence"></div>
        </div>

        <div class="controls">
          <button id="startBtn" class="primary">Start Camera</button>
          <button id="clearBtn" class="secondary">Clear Sentence</button>
        </div>

        <div class="footer-note">
          Hold a gesture steady. The browser sends captured frames to the local backend for prediction.
        </div>
      </div>
    </div>
  </div>

  <script>
    const video = document.getElementById('video');
    const overlay = document.getElementById('overlay');
    const ctx = overlay.getContext('2d');
    const predictionEl = document.getElementById('prediction');
    const confidenceEl = document.getElementById('confidence');
    const statusText = document.getElementById('statusText');
    const sentenceEl = document.getElementById('sentence');
    const startBtn = document.getElementById('startBtn');
    const clearBtn = document.getElementById('clearBtn');

    let stream = null;
    let timer = null;
    let sentence = [];
    let lastStableLabel = '';
    let lastStableSince = 0;
    let lastLabel = '';

    function resizeOverlay() {
      overlay.width = overlay.clientWidth * window.devicePixelRatio;
      overlay.height = overlay.clientHeight * window.devicePixelRatio;
      ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    }

    async function startCamera() {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      video.srcObject = stream;
      await video.play();
      resizeOverlay();
      window.addEventListener('resize', resizeOverlay);
      statusText.textContent = 'Scanning';
      startBtn.textContent = 'Camera On';
      startBtn.disabled = true;
      timer = setInterval(captureAndPredict, 280);
    }

    async function captureAndPredict() {
      if (!video.videoWidth || !video.videoHeight) return;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const c = canvas.getContext('2d');
      c.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.82);

      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      });
      const data = await response.json();

      predictionEl.textContent = data.label || '--';
      confidenceEl.textContent = `Confidence: ${Math.round((data.confidence || 0) * 100)}%`;

      const now = Date.now();
      if (data.label && data.label !== 'UNKNOWN' && (data.confidence || 0) >= 0.7) {
        if (data.label !== lastLabel) {
          lastLabel = data.label;
          lastStableSince = now;
          lastStableLabel = data.label;
        } else if (now - lastStableSince >= 700) {
          const last = sentence[sentence.length - 1];
          if (last !== data.label) {
            sentence.push(data.label);
            sentenceEl.textContent = sentence.join(' ');
          }
        }
      } else {
        lastLabel = '';
      }

      const top3 = (data.top3 || []).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(' • ');
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      ctx.fillStyle = 'rgba(0,0,0,0.45)';
      ctx.fillRect(16, 16, 220, 58);
      ctx.fillStyle = '#fff';
      ctx.font = '600 18px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText(`Live: ${data.label || '--'}`, 28, 40);
      ctx.font = '13px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText(top3 || 'Waiting for hand...', 28, 62);
      statusText.textContent = data.label && data.label !== 'UNKNOWN' ? 'Tracking' : 'Searching';
    }

    startBtn.addEventListener('click', () => startCamera().catch(err => {
      statusText.textContent = 'Camera unavailable';
      console.error(err);
    }));

    clearBtn.addEventListener('click', () => {
      sentence = [];
      sentenceEl.textContent = '';
      lastLabel = '';
      lastStableSince = 0;
      statusText.textContent = 'Ready';
    });

    window.addEventListener('beforeunload', () => {
      if (timer) clearInterval(timer);
      if (stream) stream.getTracks().forEach(track => track.stop());
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/api/health")
def health() -> Dict[str, object]:
    return {"ok": True, "model_loaded": analyzer.classifier.model is not None or analyzer.classifier.cnn_classifier is not None}


@app.post("/api/predict")
async def predict(request: Request) -> JSONResponse:
    payload = await request.json()
    image_value = payload.get("image")
    if not image_value:
        raise HTTPException(status_code=400, detail="Missing image payload")

    if "," in image_value:
        image_value = image_value.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload") from exc

    result = analyzer.predict_image(image_bytes)
    return JSONResponse(result)
