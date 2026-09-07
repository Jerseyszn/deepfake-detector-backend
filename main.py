"""
Multimodal Deepfake & Misinformation Detector - RENDER-COMPATIBLE Backend
===========================================================================
This is a variant of backend/main.py built to run on Render's free tier
(512 MB RAM, 0.1 CPU). It does NOT load PyTorch or Transformers models
locally - that needs far more memory than the free tier gives you.

Instead:
  - OpenCV / FFmpeg (light, run locally on Render) still do the real
    preprocessing named in Chapter 4.4.1: face-cropping images/frames and
    normalising audio.
  - The actual AI inference (RoBERTa / image classifier / Wav2Vec2) is
    delegated to Hugging Face's hosted Inference API over HTTPS, so the
    heavy model weights and compute live on Hugging Face's servers, not
    on your Render instance.

You need a free Hugging Face account and API token for this to work.
Set it as an environment variable on Render named HF_API_TOKEN.
"""

import io
import os
import time
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Deepfake Detector API (Render + HF Inference API proxy)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

TEXT_MODEL = "openai-community/roberta-large-openai-detector"
IMAGE_MODEL = "aaronespasa/deepfake-detection-resnetinceptionv1"
AUDIO_MODEL = "facebook/wav2vec2-base"

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class DetectionResult(BaseModel):
    analysis_id: str
    is_fake: bool
    confidence_score: float
    visual_score: Optional[float] = None
    acoustic_score: Optional[float] = None
    textual_score: Optional[float] = None
    latency_ms: int
    explanation: str


def decision(prob_fake: float):
    return prob_fake >= 0.5, (prob_fake if prob_fake >= 0.5 else 1 - prob_fake)


def call_hf_inference(model: str, data: bytes = None, json_payload: dict = None, retries: int = 3):
    """Calls Hugging Face's hosted Inference API. Free-tier models can be
    'cold' (unloaded) - HF returns a 503 with an estimated_time while it
    spins the model up, so we retry a couple of times."""
    url = f"https://api-inference.huggingface.co/models/{model}"
    for attempt in range(retries):
        if json_payload is not None:
            resp = requests.post(url, headers=HF_HEADERS, json=json_payload, timeout=30)
        else:
            resp = requests.post(url, headers=HF_HEADERS, data=data, timeout=30)
        if resp.status_code == 503:
            wait = resp.json().get("estimated_time", 5)
            time.sleep(min(wait, 15))
            continue
        resp.raise_for_status()
        return resp.json()
    raise HTTPException(503, "Hugging Face model is still loading — try again in a few seconds")


@app.get("/health")
def health():
    return {"status": "ok", "mode": "render-proxy", "hf_token_configured": bool(HF_API_TOKEN)}


@app.post("/api/analyze/image", response_model=DetectionResult)
async def analyze_image(file: UploadFile = File(...)):
    t0 = time.time()
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
    crop = img
    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = img[y:y + h, x:x + w]
    crop = cv2.resize(crop, (224, 224))
    ok, encoded = cv2.imencode(".jpg", crop)
    if not ok:
        raise HTTPException(500, "Could not encode cropped image")

    preds = call_hf_inference(IMAGE_MODEL, data=encoded.tobytes())
    fake_prob = next((p["score"] for p in preds if "fake" in p["label"].lower()), preds[0]["score"])
    is_fake, conf = decision(fake_prob)

    return DetectionResult(
        analysis_id=str(uuid.uuid4()), is_fake=is_fake, confidence_score=round(conf, 4),
        visual_score=round(fake_prob, 4), latency_ms=int((time.time() - t0) * 1000),
        explanation="Face cropped locally with OpenCV; scored remotely via Hugging Face Inference API.",
    )


@app.post("/api/analyze/video", response_model=DetectionResult)
async def analyze_video(file: UploadFile = File(...)):
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / file.filename
        video_path.write_bytes(await file.read())
        frame_path = Path(tmp) / "frame.jpg"
        audio_path = Path(tmp) / "audio.wav"

        subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-ss", "00:00:01", "-frames:v", "1", str(frame_path)], capture_output=True)
        has_audio = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "16000", "-ac", "1", str(audio_path)],
            capture_output=True,
        ).returncode == 0 and audio_path.exists()

        visual_score = None
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
            crop = img
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                crop = img[y:y + h, x:x + w]
            crop = cv2.resize(crop, (224, 224))
            ok, encoded = cv2.imencode(".jpg", crop)
            preds = call_hf_inference(IMAGE_MODEL, data=encoded.tobytes())
            visual_score = next((p["score"] for p in preds if "fake" in p["label"].lower()), preds[0]["score"])

        acoustic_score = None
        if has_audio:
            preds = call_hf_inference(AUDIO_MODEL, data=audio_path.read_bytes())
            if isinstance(preds, list) and preds:
                acoustic_score = next((p["score"] for p in preds if "fake" in p.get("label", "").lower()), preds[0].get("score", 0.5))

        scores = [s for s in (visual_score, acoustic_score) if s is not None]
        fused = sum(scores) / len(scores) if scores else 0.5
        is_fake, conf = decision(fused)

        return DetectionResult(
            analysis_id=str(uuid.uuid4()), is_fake=is_fake, confidence_score=round(conf, 4),
            visual_score=round(visual_score, 4) if visual_score is not None else None,
            acoustic_score=round(acoustic_score, 4) if acoustic_score is not None else None,
            latency_ms=int((time.time() - t0) * 1000),
            explanation="FFmpeg extracted frame + audio locally; both scored remotely via Hugging Face Inference API.",
        )


@app.post("/api/analyze/audio", response_model=DetectionResult)
async def analyze_audio(file: UploadFile = File(...)):
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / file.filename
        in_path.write_bytes(await file.read())
        wav_path = Path(tmp) / "norm.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(in_path), "-ar", "16000", "-ac", "1", str(wav_path)], capture_output=True)
        target = wav_path if wav_path.exists() else in_path

        preds = call_hf_inference(AUDIO_MODEL, data=target.read_bytes())
        fake_prob = 0.5
        if isinstance(preds, list) and preds:
            fake_prob = next((p["score"] for p in preds if "fake" in p.get("label", "").lower()), preds[0].get("score", 0.5))
        is_fake, conf = decision(fake_prob)

        return DetectionResult(
            analysis_id=str(uuid.uuid4()), is_fake=is_fake, confidence_score=round(conf, 4),
            acoustic_score=round(fake_prob, 4), latency_ms=int((time.time() - t0) * 1000),
            explanation="FFmpeg normalised the clip locally; Wav2Vec2 scored it via Hugging Face Inference API.",
        )


class TextPayload(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None


def scrape_url(url: str) -> str:
    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        if art.text.strip():
            return art.text
    except Exception:
        pass
    from bs4 import BeautifulSoup
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    return "\n".join(p.get_text() for p in soup.find_all("p"))


@app.post("/api/analyze/text", response_model=DetectionResult)
async def analyze_text(payload: TextPayload):
    t0 = time.time()
    text = scrape_url(payload.url) if payload.url else (payload.text or "")
    if not text.strip():
        raise HTTPException(400, "No text extracted/provided")
    text = text[:5000]

    result = call_hf_inference(TEXT_MODEL, json_payload={"inputs": text})
    scores = result[0] if isinstance(result, list) else result
    fake_prob = next((s["score"] for s in scores if s["label"].lower() in ("fake", "label_1")), scores[0]["score"])
    is_fake, conf = decision(fake_prob)

    return DetectionResult(
        analysis_id=str(uuid.uuid4()), is_fake=is_fake, confidence_score=round(conf, 4),
        textual_score=round(fake_prob, 4), latency_ms=int((time.time() - t0) * 1000),
        explanation="Text scored remotely via Hugging Face Inference API (RoBERTa detector).",
    )
