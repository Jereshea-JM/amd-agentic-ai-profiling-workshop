"""
kokoro_server.py

FastAPI + uvicorn server for Kokoro-82M on AMD ROCm. The model loads once at
startup and stays warm in VRAM. The caller picks the mode per request:

    mode = "sequential"  -> native per-sentence inference (reference audio)
    mode = "batched"     -> length-bucketed batched inference (uses GPU width)

Run:
    cd /home/sabiras/hermes-audio-notebook/workspace
    ../env/bin/python kokoro_server.py     # :8092 (single worker only)

Endpoints:
    GET  /health
    GET  /v1/models
    GET  /v1/audio/voices
    POST /v1/audio/speech   (OpenAI-compatible; accepts optional "mode")
    POST /tts               -> JSON: base64 wav + timing metadata
    POST /tts/wav           -> raw audio/wav bytes + timing headers
"""

import os
import io
import re
import time
import base64
import asyncio
import logging
import functools
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal, Optional, Dict

# ROCm/MIOpen overrides must be set before torch is imported.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")

# MIOpen needs <user_db>/miopen-lockfiles to persist its kernel DB; recreate
# it defensively so a cleared cache dir cannot break the server.
_miopen_db = (os.environ.get("MIOPEN_USER_DB_PATH")
              or os.path.expanduser("~/.config/miopen"))
os.makedirs(os.path.join(_miopen_db, "miopen-lockfiles"), exist_ok=True)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from kokoro import KPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("kokoro-server")

LANG = "a"
DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
DEFAULT_MODE = "sequential"
DEFAULT_BATCH_SIZE = 16
DEFAULT_SPLIT_PATTERN = r"\n+|(?<=[.?!]) +"
SAMPLE_RATE = 24000
PORT = int(os.environ.get("KOKORO_PORT", "8092"))

VOICES = [
    "af_heart", "af_sky", "af_bella", "af_sarah", "af_nicole",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
]


# Batched replacement for KModel.forward_with_tokens (B > 1). Inlined here so
# the server has no dependency on the benchmark script batched_inference.py.
@torch.no_grad()
def _batched_f0n_train(predictor, en, s, frame_counts):
    """Batched F0Ntrain with the shared LSTM packed by length."""
    xt = en.transpose(-1, -2)                       # [B, F_max, C]
    lengths = frame_counts.clamp(min=1).detach().cpu()
    packed = nn.utils.rnn.pack_padded_sequence(
        xt, lengths, batch_first=True, enforce_sorted=False)
    predictor.shared.flatten_parameters()
    h, _ = predictor.shared(packed)
    h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
    f_max = en.shape[-1]
    if h.shape[1] < f_max:
        h = F.pad(h, (0, 0, 0, f_max - h.shape[1]))
    base = h.transpose(-1, -2)                      # [B, H, F_max]

    F0 = base
    for block in predictor.F0:
        F0 = block(F0, s)
    F0 = predictor.F0_proj(F0)

    N = base
    for block in predictor.N:
        N = block(N, s)
    N = predictor.N_proj(N)
    return F0.squeeze(1), N.squeeze(1)              # [B, 2*F_max] each


@torch.no_grad()
def batched_forward(model, input_ids, input_lengths, ref_s, speed=1.0):
    """Batched KModel.forward_with_tokens (B > 1).

    input_ids:     [B, T] padded with 0
    input_lengths: [B] real token counts (incl. the two 0 sentinels)
    ref_s:         [B, 256] voice style rows (pack[len(ps)-1] per row)
    Returns: audio [B, samples], pred_dur [B, T], frame_counts [B]
    """
    device = model.device
    input_ids = input_ids.to(device)
    input_lengths = input_lengths.to(device)
    ref_s = ref_s.to(device)
    B, T = input_ids.shape

    # True where position is padding.
    text_mask = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    text_mask = (text_mask + 1) > input_lengths.unsqueeze(1)   # [B, T] bool

    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)      # [B, H, T]
    s = ref_s[:, 128:]                                         # [B, 128]

    # DurationEncoder packs its own LSTMs by length (mask-safe).
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

    # Duration LSTM: pack by length (stock path leaves it unpacked, which
    # leaks across the batch under padding).
    lengths_cpu = input_lengths.detach().cpu()
    packed = nn.utils.rnn.pack_padded_sequence(
        d, lengths_cpu, batch_first=True, enforce_sorted=False)
    model.predictor.lstm.flatten_parameters()
    x, _ = model.predictor.lstm(packed)
    x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
    if x.shape[1] < T:
        x = F.pad(x, (0, 0, 0, T - x.shape[1]))
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(dim=-1) / speed     # [B, T]
    pred_dur = torch.round(duration).clamp(min=1).long()
    pred_dur = pred_dur.masked_fill(text_mask, 0)   # padded -> 0 frames

    # Per-sample alignment matrix [B, T, F_max] (B, T small; cheap).
    frame_counts = pred_dur.sum(dim=1)              # [B]
    f_max = int(frame_counts.max().item())
    pred_aln = torch.zeros((B, T, f_max), device=device)
    pd = pred_dur.tolist()
    for i in range(B):
        c = 0
        for t, di in enumerate(pd[i]):
            if di > 0:
                pred_aln[i, t, c:c + di] = 1.0
                c += di

    en = d.transpose(-1, -2) @ pred_aln             # [B, C, F_max]
    F0_pred, N_pred = _batched_f0n_train(
        model.predictor, en, s, frame_counts)

    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln                           # [B, C, F_max]

    audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128])
    audio = audio.squeeze(1)                        # [B, samples]
    return audio, pred_dur, frame_counts


class KokoroEngine:
    def __init__(self, lang=LANG, device=None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pipeline = KPipeline(lang_code=lang, device=device)
        self.model = self.pipeline.model
        self._voice_cache: Dict[str, torch.Tensor] = {}

    def get_pack(self, voice: str) -> torch.Tensor:
        if voice not in self._voice_cache:
            pack = self.pipeline.load_voice(voice).to(self.device)
            self._voice_cache[voice] = pack
        return self._voice_cache[voice]

    def build_items(self, text: str, split_pattern: str):
        items = []
        for seg in re.split(split_pattern, text.strip()):
            if not seg.strip():
                continue
            _, tokens = self.pipeline.g2p(seg)
            for _gs, ps, _tks in self.pipeline.en_tokenize(tokens):
                if not ps:
                    continue
                if len(ps) > 510:
                    ps = ps[:510]
                ids = [self.model.vocab.get(p) for p in ps]
                ids = [v for v in ids if v is not None]
                if not ids:
                    continue
                items.append({"idx": len(items), "ps": ps, "ids": ids})
        return items

    @torch.no_grad()
    def _run_sequential(self, items, pack, speed):
        out = {}
        for it in items:
            result = KPipeline.infer(self.model, it["ps"], pack, speed)
            out[it["idx"]] = result.audio.detach().cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out

    @torch.no_grad()
    def _run_batched(self, items, pack, speed, batch_size):
        out = {}
        order = sorted(items, key=lambda it: len(it["ids"]))
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            seqs = [[0, *it["ids"], 0] for it in chunk]
            lengths = [len(s) for s in seqs]
            t_max = max(lengths)

            input_ids = torch.zeros((len(chunk), t_max), dtype=torch.long)
            for i, s in enumerate(seqs):
                input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            input_lengths = torch.tensor(lengths, dtype=torch.long)
            ref_s = torch.cat(
                [pack[min(len(it["ps"]) - 1, pack.shape[0] - 1)]
                 for it in chunk],
                dim=0,
            )

            audio, _pred_dur, frame_counts = batched_forward(
                self.model, input_ids, input_lengths, ref_s, speed=speed)
            total_samples = audio.shape[-1]
            f_max = int(frame_counts.max().item())
            ratio = total_samples / max(f_max, 1)
            for i, it in enumerate(chunk):
                n = min(total_samples,
                        int(round(frame_counts[i].item() * ratio)))
                out[it["idx"]] = audio[i, :n].detach().cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out

    def synthesize(self, text, mode, voice, speed, batch_size,
                   split_pattern):
        pack = self.get_pack(voice)

        t = time.perf_counter()
        items = self.build_items(text, split_pattern)
        g2p_time = time.perf_counter() - t
        if not items:
            raise ValueError("No speakable text after phonemization.")

        t = time.perf_counter()
        if mode == "sequential":
            out = self._run_sequential(items, pack, speed)
        elif mode == "batched":
            out = self._run_batched(items, pack, speed, batch_size)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
        inference_time = time.perf_counter() - t

        audio = np.concatenate([out[i] for i in sorted(out)], axis=0)
        audio_seconds = len(audio) / SAMPLE_RATE
        rtf = (round(inference_time / audio_seconds, 4)
               if audio_seconds else None)
        rt_factor = (round(audio_seconds / inference_time, 1)
                     if inference_time else None)
        meta = {
            "mode": mode,
            "voice": voice,
            "speed": speed,
            "batch_size": batch_size if mode == "batched" else None,
            "num_sentences": len(items),
            "sample_rate": SAMPLE_RATE,
            "audio_seconds": round(audio_seconds, 3),
            "g2p_time": round(g2p_time, 3),
            "inference_time": round(inference_time, 3),
            "total_time": round(g2p_time + inference_time, 3),
            "rtf": rtf,
            "realtime_factor": rt_factor,
        }
        return audio, meta

    def warmup(self):
        text = "This is a warmup sentence. And another for batching."
        for mode in ("sequential", "batched"):
            self.synthesize(
                text, mode, DEFAULT_VOICE, DEFAULT_SPEED,
                DEFAULT_BATCH_SIZE, DEFAULT_SPLIT_PATTERN,
            )


ENGINE: Optional[KokoroEngine] = None

# One dedicated GPU thread for all inference (model init, warmup, requests).
# A single model on one GPU serializes anyway, so this keeps the event loop
# free and avoids races on the shared module without needing a lock.
_gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro-gpu")


def _on_gpu(fn, *args):
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(_gpu, functools.partial(fn, *args))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    log.info("Loading Kokoro-82M into VRAM (once)...")
    t = time.perf_counter()
    ENGINE = await _on_gpu(KokoroEngine)
    log.info("Loaded on %s in %.2fs", ENGINE.device,
             time.perf_counter() - t)
    log.info("Warming up both modes...")
    await _on_gpu(ENGINE.warmup)
    log.info("Ready.")
    yield
    _gpu.shutdown(wait=True)
    log.info("Shutting down.")


app = FastAPI(title="Kokoro TTS Server", version="2.0", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mode: Literal["sequential", "batched"] = DEFAULT_MODE
    voice: str = DEFAULT_VOICE
    speed: float = Field(DEFAULT_SPEED, gt=0.0, le=5.0)
    batch_size: int = Field(DEFAULT_BATCH_SIZE, ge=1, le=128)
    split_pattern: str = DEFAULT_SPLIT_PATTERN


class TTSResponse(BaseModel):
    mode: str
    voice: str
    speed: float
    batch_size: Optional[int]
    num_sentences: int
    sample_rate: int
    audio_seconds: float
    g2p_time: float
    inference_time: float
    total_time: float
    rtf: Optional[float]
    realtime_factor: Optional[float]
    audio_base64: str


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1)
    voice: str = DEFAULT_VOICE
    speed: float = Field(DEFAULT_SPEED, gt=0.0, le=5.0)
    response_format: str = "wav"
    model: str = "kokoro"
    mode: Literal["sequential", "batched"] = DEFAULT_MODE
    batch_size: int = Field(DEFAULT_BATCH_SIZE, ge=1, le=128)


async def _synthesize(text, mode, voice, speed, batch_size,
                      split_pattern=DEFAULT_SPLIT_PATTERN):
    if ENGINE is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded yet.")
    try:
        return await _on_gpu(
            ENGINE.synthesize, text, mode, voice, speed, batch_size,
            split_pattern)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("synthesis failed")
        raise HTTPException(
            status_code=500, detail=f"{type(e).__name__}: {e}")


def _to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def _timing_headers(meta: dict) -> dict:
    return {
        "X-Mode": str(meta["mode"]),
        "X-Num-Sentences": str(meta["num_sentences"]),
        "X-G2P-Time": str(meta["g2p_time"]),
        "X-Inference-Time": str(meta["inference_time"]),
        "X-Audio-Seconds": str(meta["audio_seconds"]),
        "X-RTF": str(meta["rtf"]),
    }


@app.get("/health")
async def health():
    gpu = (torch.cuda.get_device_name(0)
           if torch.cuda.is_available() else "cpu")
    return {
        "status": "ok" if ENGINE is not None else "loading",
        "device": ENGINE.device if ENGINE else None,
        "gpu": gpu,
        "modes": ["sequential", "batched"],
        "default_mode": DEFAULT_MODE,
    }


@app.get("/v1/models")
async def get_models():
    return {
        "object": "list",
        "data": [
            {"id": "kokoro", "object": "model", "owned_by": "kokoro"}
        ],
    }


@app.get("/v1/audio/voices")
async def get_voices():
    return {"voices": VOICES}


@app.post("/tts", response_model=TTSResponse)
async def tts(req: TTSRequest):
    audio, meta = await _synthesize(
        req.text, req.mode, req.voice, req.speed, req.batch_size,
        req.split_pattern)
    log.info("/tts mode=%s sents=%s infer=%ss audio=%ss",
             meta["mode"], meta["num_sentences"],
             meta["inference_time"], meta["audio_seconds"])
    wav_b64 = base64.b64encode(_to_wav_bytes(audio)).decode()
    return TTSResponse(audio_base64=wav_b64, **meta)


@app.post("/tts/wav")
async def tts_wav(req: TTSRequest):
    audio, meta = await _synthesize(
        req.text, req.mode, req.voice, req.speed, req.batch_size,
        req.split_pattern)
    return Response(
        content=_to_wav_bytes(audio),
        media_type="audio/wav",
        headers=_timing_headers(meta),
    )


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if req.response_format.lower() != "wav":
        raise HTTPException(
            status_code=400,
            detail="Only response_format='wav' is supported.")
    audio, meta = await _synthesize(
        req.input, req.mode, req.voice, req.speed, req.batch_size)
    log.info("/v1/audio/speech mode=%s infer=%ss audio=%ss",
             meta["mode"], meta["inference_time"],
             meta["audio_seconds"])
    return Response(
        content=_to_wav_bytes(audio),
        media_type="audio/wav",
        headers=_timing_headers(meta),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1,
                log_level="info")
