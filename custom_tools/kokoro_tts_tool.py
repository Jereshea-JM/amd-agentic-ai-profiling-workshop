"""
kokoro_tts_tool.py

Single, unified custom TTS tool for the Hermes agent, backed by the local
Kokoro TTS server (kokoro_server.py).

One registered tool -- `kokoro_tts` -- with a `mode` argument that selects the
inference strategy:

  - mode="sequential" : native Kokoro path, one sentence per GPU forward.
                        Baseline / reference audio; slower on long text.
  - mode="batched"    : optimized path, many sentences packed into a single
                        GPU forward (length-bucketing + padding + masking).
                        Faster on long, multi-sentence text; same audio.

This file merges what were previously three files (a shared HTTP client plus
separate seq and batched tools) into one module for a unified structure.
The Kokoro server must be running (helper.sh starts it).
"""
from __future__ import annotations

import datetime
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from tools.registry import registry, tool_error

log = logging.getLogger("kokoro_tts")

SERVER_URL = "http://localhost:8092"
DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_MODE = "sequential"
MODES = ("sequential", "batched")
_DEBUG_LOG = Path.home() / ".hermes" / "kokoro_tts_calls.log"

VOICES = [
    "af_heart", "af_sky", "af_bella", "af_sarah", "af_nicole",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _arg_summary(args: dict) -> dict:
    return {
        k: (f"<str:{len(v)}>" if isinstance(v, str) else v)
        for k, v in (args or {}).items()
    }


def _log_call(mode: str, summary: dict, note: str = "") -> None:
    try:
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{ts} mode={mode} args={summary} {note}\n")
    except Exception:
        pass
    log.info("kokoro_tts [%s] call: args=%s %s", mode, summary, note)


def _resolve_text(args: dict) -> tuple[str, str]:
    for key in ("text", "input"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip(), f"inline '{key}' ({len(v)} chars)"
    for key in ("text_file", "input_file", "file", "path"):
        p = args.get(key)
        if isinstance(p, str) and p.strip():
            try:
                data = Path(p).expanduser().read_text(encoding="utf-8")
            except Exception as e:
                return "", f"file '{p}' unreadable: {e}"
            if data.strip():
                return data.strip(), f"file '{p}' ({len(data)} chars)"
            return "", f"file '{p}' is empty"
    return "", "no text/input/file key with content"


def _as_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Core: resolve args -> call the Kokoro server -> save WAV
# ---------------------------------------------------------------------------
def run(args: dict) -> str:
    args = args or {}
    mode = str(args.get("mode") or DEFAULT_MODE).lower()
    if mode not in MODES:
        return tool_error(
            f"Unknown mode '{mode}'. Use one of: {', '.join(MODES)}.")

    summary = _arg_summary(args)
    text, how = _resolve_text(args)
    _log_call(mode, summary, note=f"-> {how}")
    if not text:
        keys = list(args.keys())
        return tool_error(
            f"No text provided. Received keys={keys} ({summary}). "
            "Put the full text in the 'text' field, or a file path "
            "in 'text_file'."
        )
    return synthesize(
        mode=mode,
        text=text,
        voice=args.get("voice") or DEFAULT_VOICE,
        speed=_as_float(args.get("speed"), DEFAULT_SPEED),
        batch_size=_as_int(args.get("batch_size"), DEFAULT_BATCH_SIZE),
        output_path=args.get("output_path"),
    )


def synthesize(
    mode: str,
    text: str,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_path: Optional[str] = None,
) -> str:
    if voice not in VOICES:
        return tool_error(
            f"Unknown voice '{voice}'. Choose from: {', '.join(VOICES)}")

    file_prefix = f"kokoro_tts_{mode}"
    payload = json.dumps({
        "text": text,
        "mode": mode,
        "voice": voice,
        "speed": speed,
        "batch_size": batch_size,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/tts/wav",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            wav_bytes = resp.read()
            headers = resp.headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        return tool_error(
            f"kokoro_tts [{mode}]: server error {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        return tool_error(
            f"kokoro_tts [{mode}]: cannot reach kokoro server at "
            f"{SERVER_URL} - start kokoro_server.py first. ({exc})"
        )
    except Exception as exc:
        return tool_error(f"kokoro_tts [{mode}]: request failed: {exc}")

    if not wav_bytes:
        return tool_error(f"kokoro_tts [{mode}]: server returned empty audio.")

    elapsed = time.perf_counter() - t0
    if not output_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path.home() / ".hermes" / "audio_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{file_prefix}_{ts}.wav")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(wav_bytes)

    nsent = headers.get("X-Num-Sentences", "?")
    try:
        infer_f = float(headers.get("X-Inference-Time"))
        audio_f = float(headers.get("X-Audio-Seconds"))
        rt = audio_f / infer_f if infer_f else 0
        timing = (
            f"{infer_f:.2f}s inference, {audio_f:.1f}s audio "
            f"({rt:.1f}x real-time)"
        )
    except (TypeError, ValueError):
        timing = f"{elapsed:.2f}s round-trip"

    return (
        f"MEDIA:{output_path}\n"
        f"Done [{mode}] {nsent} sentence(s) - {timing}"
    )


# ---------------------------------------------------------------------------
# Tool schema + registration (single unified tool)
# ---------------------------------------------------------------------------
KOKORO_TTS_SCHEMA = {
    "name": "kokoro_tts",
    "description": (
        "Local text-to-speech via the Kokoro server on the AMD MI300X GPU. "
        "A single tool with two strategies selected by 'mode': "
        "'sequential' (native Kokoro, one sentence per GPU forward - the "
        "baseline/reference) and 'batched' (optimized: packs many sentences "
        "into one GPU forward for faster generation on long text; same "
        "audio). No length limit: pass the full text in 'text' (or a file "
        "path in 'text_file') in one call; do not pre-split. If you already have the text, put it in 'text' directly and do NOT write it to a file first. "
        "kokoro_server.py must be running."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to synthesize directly. Use this whenever you already have the text - do not write it to a file first.",
            },
            "text_file": {
                "type": "string",
                "description": (
                    "Path to a UTF-8 text file to synthesize "
                    "(use ONLY when the user gives you a file path; never create or write a new file)."
                ),
            },
            "mode": {
                "type": "string",
                "enum": list(MODES),
                "description": (
                    "Inference strategy. 'sequential' = baseline "
                    "(one sentence per GPU forward); 'batched' = optimized "
                    f"(many sentences per forward). Default: {DEFAULT_MODE}."
                ),
            },
            "voice": {
                "type": "string",
                "description": (
                    f"Voice name. Default: {DEFAULT_VOICE}. "
                    f"Options: {', '.join(VOICES)}."
                ),
            },
            "batch_size": {
                "type": "integer",
                "description": (
                    "Sentences per GPU forward when mode='batched'. "
                    "Default 16 (sweet spot 16-24). Ignored for sequential."
                ),
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Output WAV path. Defaults to "
                    "~/.hermes/audio_cache/kokoro_tts_<mode>_<timestamp>.wav"
                ),
            },
        },
        "required": ["text"],
    },
}

registry.register(
    name="kokoro_tts",
    toolset="tts",
    schema=KOKORO_TTS_SCHEMA,
    handler=lambda args, **kw: run(args),
)
