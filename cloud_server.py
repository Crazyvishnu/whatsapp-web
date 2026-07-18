"""Jarvis Cloud STT/TTS Server — deploy this on a free-tier cloud VM
(Oracle Cloud Always Free, or similar) to offload Whisper STT and Kokoro TTS
CPU load off your local PC. Jarvis's local stt.py/tts.py call this instead
of loading the models locally.

Setup on the VM:
  pip install fastapi uvicorn faster-whisper kokoro numpy python-multipart

Run:
  uvicorn cloud_server:app --host 0.0.0.0 --port 8000

IMPORTANT SECURITY NOTE: this has zero authentication. Anyone who finds
the VM's IP:port can use your server for free. At minimum, set API_KEY
below to a random string and restrict the VM's firewall/security group
to only your home IP if possible.
"""
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="Jarvis Cloud STT/TTS")

# ─── Simple shared-secret auth — set this to something random ───
API_KEY = "CHANGE-THIS-TO-A-RANDOM-SECRET"


def _check_auth(x_api_key: str | None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ─── STT (Whisper) ───
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper_model


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    x_api_key: str | None = Header(None),
):
    _check_auth(x_api_key)
    raw = await audio.read()
    # Expecting raw float32 16kHz mono PCM bytes (matches Jarvis's local format)
    samples = np.frombuffer(raw, dtype=np.float32)
    model = _get_whisper()
    segments, _ = model.transcribe(samples, beam_size=5, language="en")
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"text": text}


# ─── TTS (Kokoro) ───
_tts_pipeline = None


def _get_tts_pipeline():
    global _tts_pipeline
    if _tts_pipeline is None:
        from kokoro import KPipeline
        _tts_pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return _tts_pipeline


@app.post("/speak")
async def speak(
    text: str = Form(...),
    voice: str = Form("af_heart"),
    speed: float = Form(1.1),
    x_api_key: str | None = Header(None),
):
    _check_auth(x_api_key)
    pipeline = _get_tts_pipeline()

    audio_chunks = []
    for _, _, audio_chunk in pipeline(text, voice=voice, speed=speed):
        if audio_chunk is not None:
            audio_chunks.append(audio_chunk)

    if not audio_chunks:
        raise HTTPException(status_code=500, detail="TTS produced no audio")

    combined = np.concatenate(audio_chunks)
    pcm = (combined * 32767).astype(np.int16)

    return StreamingResponse(
        io.BytesIO(pcm.tobytes()),
        media_type="application/octet-stream",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
