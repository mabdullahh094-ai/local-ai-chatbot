from __future__ import annotations

from collections import deque
import html
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import re
import tempfile
import threading
import time
from typing import Any, Iterator
from uuid import uuid4

import av
from docx import Document
from faster_whisper import WhisperModel
import numpy as np
from pypdf import PdfReader
import pyttsx3
import requests
import streamlit as st
from streamlit_webrtc import (
    WebRtcMode,
    create_audio_sink_track,
    create_pcm_audio_source_track,
    webrtc_streamer,
)


APP_RUNTIME_VERSION = 19
INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
SILENCE_SECONDS = 0.7          # how long a pause ends your turn (lower = faster replies)
MAX_UTTERANCE_SECONDS = 20.0
MIN_VOICE_SECONDS = 0.22
VAD_RMS = 180.0               # lower threshold for normal laptop microphones
ECHO_GUARD = 1.18              # browser echo cancellation already reduces speaker feedback
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_WHISPER = "tiny"

# Document settings
MAX_DOC_CHARS = 200_000      # per file, extra text is cut off
DOC_FULL_LIMIT = 6_000       # if all docs are smaller than this, send everything
DOC_CONTEXT_CHARS = 6_000    # otherwise send only the best matching chunks
CHUNK_SIZE = 900
NUM_CTX = 4096               # keep constant so Ollama does not reload the model

HF_ROOT = Path(r"D:\AI\huggingface")
os.environ.setdefault("HF_HOME", str(HF_ROOT))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_ROOT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_ROOT / "transformers"))
os.environ.setdefault("LLAMA_ARG_REPACK", "0")

SYSTEM_PROMPT = """You are a fast, friendly voice assistant running fully offline.
Reply in the same language as the user. For Urdu or Roman Urdu, always use
Roman Urdu in Latin letters, never Urdu script. For English, reply in English.
This is a spoken conversation, so answer naturally in one or two short
sentences. Do not use markdown, lists, emoji, citations, or stage directions.
Answer only the user's latest request. If it is unclear, ask one short question.
Never mention speech recognition, transcription, prompts, or internal tools.
"""

TEXT_SYSTEM_PROMPT = """You are a helpful, friendly assistant running fully offline.
Reply in the same language as the user. For Urdu or Roman Urdu, always use
Roman Urdu in Latin letters, never Urdu script. For English, reply in English.
Be clear and reasonably concise. You may use simple markdown (short lists,
bold) when it helps. Answer only the user's latest request. If it is unclear,
ask one short question.
"""

DOC_INSTRUCTIONS = """

The user has uploaded documents. Relevant excerpts are below. If the question
is about the documents, answer using only this content and say so if the answer
is not in it. Otherwise answer normally.

--- DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---
"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _spoken_text(text: str) -> str:
    """Remove formatting that an offline speech engine should not read aloud."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+", "link", text)
    text = re.sub(r"[*_#>`~|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:900]


def _wav_to_pcm(path: Path) -> np.ndarray:
    """Decode a SAPI WAV and resample to mono s16 at 24 kHz."""
    chunks: list[np.ndarray] = []
    resampler = av.AudioResampler(format="s16", layout="mono", rate=OUTPUT_RATE)
    with av.open(str(path), mode="r") as container:
        stream = next(item for item in container.streams if item.type == "audio")
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1).astype(np.int16, copy=False))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1).astype(np.int16, copy=False))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)


def _split_sentences(buffer: str, final: bool = False) -> tuple[list[str], str]:
    """Cut streamed LLM text into speakable sentences as soon as they are complete."""
    out: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?؟۔]+[\"”')\]]*(?=\s|$)", buffer):
        end = match.end()
        if end - start < 12 and not final:
            continue  # too short to speak on its own, keep collecting
        out.append(buffer[start:end].strip())
        start = end
    rest = buffer[start:]
    if len(rest) > 180:  # very long sentence: speak it in pieces
        cut = rest.rfind(" ", 0, 180)
        if cut > 60:
            out.append(rest[:cut].strip())
            rest = rest[cut:]
    if final and rest.strip():
        out.append(rest.strip())
        rest = ""
    return [s for s in out if s], rest.lstrip()


def extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("PDF password se protected hai")
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[Page {number}]\n{text}")
    return "\n\n".join(pages)


def extract_docx_text(data: bytes) -> str:
    document = Document(io.BytesIO(data))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def recorded_audio_to_pcm(data: bytes) -> np.ndarray:
    """Decode a browser-recorded audio note into mono 16 kHz PCM for Whisper."""
    chunks: list[np.ndarray] = []
    resampler = av.AudioResampler(format="s16", layout="mono", rate=INPUT_RATE)
    with av.open(io.BytesIO(data), mode="r") as container:
        stream = next(item for item in container.streams if item.type == "audio")
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                chunks.append(
                    converted.to_ndarray().reshape(-1).astype(np.int16, copy=False)
                )
        for converted in resampler.resample(None):
            chunks.append(
                converted.to_ndarray().reshape(-1).astype(np.int16, copy=False)
            )
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)


def extract_document_text(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        text = extract_pdf_text(data)
    elif suffix == ".docx":
        text = extract_docx_text(data)
    else:
        raise ValueError("Sirf .pdf aur .docx files supported hain")
    return text.strip()[:MAX_DOC_CHARS]


def _chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    chunks: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in text.splitlines()):
        if not paragraph:
            continue
        while len(paragraph) > size:  # very long line: slice it
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:size])
            paragraph = paragraph[size:]
        if current and len(current) + len(paragraph) + 1 > size:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w{3,}", text.lower()))


# ----------------------------------------------------------------------------
# Voice + text bridge
# ----------------------------------------------------------------------------
class RealtimeVoiceBridge:
    """Browser WebRTC audio -> local AI -> browser WebRTC audio (+ text chat)."""

    runtime_version = APP_RUNTIME_VERSION

    def __init__(self, source: Any) -> None:
        self.source = source
        self._state_lock = threading.RLock()
        self._audio_lock = threading.Lock()
        self._jobs: queue.Queue[tuple[str, int, Any]] = queue.Queue(maxsize=4)
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._active = False
        self._muted = False
        self._generation = 0
        self._phase = "Call band hai"
        self._error = ""
        self._speaking_until = 0.0
        self._live_user = ""
        self._live_ai = ""
        self._history: list[dict[str, str]] = []   # what the model sees
        self._log: list[dict[str, str]] = []        # what the user sees
        self._documents: dict[str, str] = {}
        self._model_name = DEFAULT_MODEL
        self._whisper_name = DEFAULT_WHISPER
        self._temperature = 0.25
        self._loaded_whisper_name = ""
        self._whisper: WhisperModel | None = None
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=INPUT_RATE)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=12)
        self._chunks: list[np.ndarray] = []
        self._in_speech = False
        self._voice_run = 0
        self._voiced_samples = 0
        self._silence_samples = 0
        self._total_samples = 0
        self._audio_frames = 0
        self._last_audio_at = 0.0
        self._last_audio_rms = 0.0
        self._audio_error = ""
        self._noise_floor = 55.0

    # ---- configuration ----------------------------------------------------
    def configure(self, model: str, whisper_name: str, temperature: float) -> None:
        with self._state_lock:
            self._model_name = model
            self._whisper_name = whisper_name
            self._temperature = temperature

    def sync_documents(self, documents: dict[str, str]) -> None:
        with self._state_lock:
            self._documents = dict(documents)

    def set_muted(self, muted: bool) -> None:
        with self._state_lock:
            changed = muted != self._muted
            self._muted = muted
        if changed and muted:
            self._reset_utterance()

    def messages(self) -> list[dict[str, str]]:
        with self._state_lock:
            return list(self._log)

    def export_conversation(self) -> dict[str, list[dict[str, str]]]:
        """Return a safe copy of the current chat for local history storage."""
        with self._state_lock:
            return {
                "history": [dict(item) for item in self._history],
                "messages": [dict(item) for item in self._log],
            }

    def load_conversation(self, data: dict[str, Any] | None) -> None:
        """Switch the visible/model context to a previously saved conversation."""
        data = data or {}
        history = data.get("history", [])
        messages = data.get("messages", [])
        with self._state_lock:
            self._history = [
                {"role": str(x.get("role", "user")), "content": str(x.get("content", ""))}
                for x in history if isinstance(x, dict) and x.get("role") in ("user", "assistant")
            ][-10:]
            self._log = [
                {"role": str(x.get("role", "assistant")), "content": str(x.get("content", "")), "source": str(x.get("source", "text"))}
                for x in messages if isinstance(x, dict) and x.get("role") in ("user", "assistant")
            ][-200:]
            self._generation += 1
            self._live_user = ""
            self._live_ai = ""
            self._speaking_until = 0.0
        self.source.clear()
        self._reset_utterance()

    def live_captions(self) -> tuple[str, str]:
        with self._state_lock:
            return self._live_user, self._live_ai

    def log_message(self, role: str, content: str, source: str = "text") -> None:
        self._log_add(role, content, source)

    def _log_add(self, role: str, content: str, source: str) -> None:
        with self._state_lock:
            self._log.append({"role": role, "content": content, "source": source})
            self._log = self._log[-200:]

    # ---- state helpers ----------------------------------------------------
    def _set_phase(self, phase: str, error: str = "") -> None:
        with self._state_lock:
            self._phase = phase
            self._error = error

    def _current_generation(self) -> int:
        with self._state_lock:
            return self._generation

    def _is_current(self, token: int) -> bool:
        with self._state_lock:
            return self._active and token == self._generation

    def _bump_generation(self) -> int:
        with self._state_lock:
            self._generation += 1
            return self._generation

    def _ensure_worker(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(
            target=self._worker, name="offline-webrtc-voice", daemon=True
        )
        self._worker_thread.start()

    # ---- call lifecycle ---------------------------------------------------
    def start(self) -> None:
        with self._state_lock:
            if self._active:
                return
            self._active = True
            self._generation += 1
            token = self._generation
            self._phase = "Call connect ho gayi"
            self._error = ""
            self._live_user = ""
            self._live_ai = ""
        self.source.clear()
        self._reset_utterance()
        self._ensure_worker()
        # Start in listening mode immediately; an opening TTS greeting can mask
        # the user's first words on laptop microphones and delay VAD activation.

    def pause(self) -> None:
        with self._state_lock:
            if not self._active:
                return
            self._active = False
            self._muted = False
            self._generation += 1
            self._phase = "Call band hai"
            self._speaking_until = 0.0
            self._live_user = ""
            self._live_ai = ""
        self.source.clear()
        self._reset_utterance()

    def shutdown(self) -> None:
        """Stop this bridge when code/settings create a replacement instance."""
        self.pause()
        self._stop_event.set()
        self._put_job(("stop", self._current_generation(), None))

    def forget(self) -> None:
        with self._state_lock:
            self._history.clear()
            self._log.clear()
            self._generation += 1
            self._speaking_until = 0.0
            self._live_user = ""
            self._live_ai = ""
            active = self._active
            token = self._generation
            self._phase = "Sun raha hoon" if active else "Call band hai"
            self._error = ""
        self.source.clear()
        self._reset_utterance()
        if active:
            self._put_job(("say", token, "Theek hai, purani baat bhool gaya. Aap bolain."))

    def snapshot(self) -> tuple[str, str]:
        with self._state_lock:
            if not self._active:
                return "Voice call band hai", "off"
            if self._error:
                return self._error, "error"
            if self._muted:
                return "Mic mute hai — Unmute karke bolain", "muted"
            if self._audio_error:
                return self._audio_error, "error"
            if time.monotonic() < self._speaking_until:
                return "Bol raha hoon — aap beech mein bol kar rok sakte hain", "speaking"
            phase = self._phase
            last_audio_at = self._last_audio_at
            frames = self._audio_frames
        if phase in {"Call connect ho gayi", "Sun raha hoon"}:
            if frames == 0 or time.monotonic() - last_audio_at > 4.0:
                return "Mic se audio nahi aa rahi — browser permission aur input device check karein", "error"
            return "Sun raha hoon — bolain", "listening"
        return phase, "working"

    def _put_job(self, job: tuple[str, int, Any]) -> None:
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            try:
                self._jobs.get_nowait()
                self._jobs.put_nowait(job)
            except queue.Empty:
                pass

    # ---- audio input / VAD ------------------------------------------------
    def _reset_utterance(self) -> None:
        with self._audio_lock:
            self._pre_roll.clear()
            self._chunks = []
            self._in_speech = False
            self._voice_run = 0
            self._voiced_samples = 0
            self._silence_samples = 0
            self._total_samples = 0

    def on_audio_frame(self, frame: av.AudioFrame) -> None:
        """Receive browser microphone frames from the WebRTC audio sink."""
        with self._state_lock:
            if not self._active or self._muted:
                return
        try:
            with self._audio_lock:
                converted_frames = self._resampler.resample(frame)
                for converted in converted_frames:
                    raw = converted.to_ndarray()
                    samples = np.asarray(raw).reshape(-1).astype(np.int16, copy=False)
                    if samples.size:
                        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                        with self._state_lock:
                            self._audio_frames += 1
                            self._last_audio_at = time.monotonic()
                            self._last_audio_rms = rms
                            self._audio_error = ""
                        self._consume_samples(samples.copy())
        except Exception as exc:
            detail = str(exc).strip().splitlines()[0][:100] if str(exc).strip() else "audio frame error"
            with self._state_lock:
                self._audio_error = f"Mic audio error: {detail}"
            # Recreate the resampler after an invalid frame so the next frame can recover.
            with self._audio_lock:
                self._resampler = av.AudioResampler(
                    format="s16", layout="mono", rate=INPUT_RATE
                )

    def _consume_samples(self, samples: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        # Adapt gently to room noise. Browser echoCancellation/noiseSuppression
        # are enabled, so do not demand an unrealistically loud interruption.
        if not self._in_speech and rms < VAD_RMS:
            self._noise_floor = (0.96 * self._noise_floor) + (0.04 * rms)
        adaptive_threshold = max(105.0, min(VAD_RMS, self._noise_floor * 2.4))
        threshold = adaptive_threshold * ECHO_GUARD if time.monotonic() < self._speaking_until else adaptive_threshold
        voiced = rms >= threshold
        self._voice_run = self._voice_run + 1 if voiced else 0
        if not self._in_speech:
            self._pre_roll.append(samples)
            if self._voice_run < 2:
                return
            self._in_speech = True
            self._chunks = list(self._pre_roll)
            self._total_samples = sum(chunk.size for chunk in self._chunks)
            self._voiced_samples = min(self._total_samples, self._voice_run * samples.size)
            self._silence_samples = 0
            self._bump_generation()
            with self._state_lock:
                self._speaking_until = 0.0
            self.source.clear()  # real barge-in: stop the current answer immediately
            self._set_phase("Sun raha hoon")
            return

        self._chunks.append(samples)
        self._total_samples += samples.size
        if voiced:
            self._voiced_samples += samples.size
            self._silence_samples = 0
        else:
            self._silence_samples += samples.size

        long_enough = self._voiced_samples >= int(MIN_VOICE_SECONDS * INPUT_RATE)
        finished = self._silence_samples >= int(SILENCE_SECONDS * INPUT_RATE)
        timed_out = self._total_samples >= int(MAX_UTTERANCE_SECONDS * INPUT_RATE)
        if timed_out or (finished and long_enough):
            audio = np.concatenate(self._chunks).astype(np.int16, copy=False)
            token = self._current_generation()
            self._chunks = []
            self._pre_roll.clear()
            self._in_speech = False
            self._voice_run = 0
            self._voiced_samples = 0
            self._silence_samples = 0
            self._total_samples = 0
            self._set_phase("Samajh raha hoon")
            self._put_job(("audio", token, audio))
        elif finished:
            self._chunks = []
            self._in_speech = False
            self._voice_run = 0
            self._voiced_samples = 0
            self._silence_samples = 0
            self._total_samples = 0

    # ---- speech to text ---------------------------------------------------
    def _load_whisper(self) -> WhisperModel:
        with self._state_lock:
            requested = self._whisper_name
        if self._whisper is not None and requested == self._loaded_whisper_name:
            return self._whisper
        self._set_phase("Offline voice model load ho raha hai")
        self._whisper = WhisperModel(
            requested,
            device="cpu",
            compute_type="int8",
            cpu_threads=max(1, min(4, (os.cpu_count() or 2) - 1)),
            download_root=str(HF_ROOT / "hub"),
            local_files_only=True,
        )
        self._loaded_whisper_name = requested
        return self._whisper

    def _transcribe(self, pcm: np.ndarray) -> str:
        audio = pcm.astype(np.float32) / 32768.0
        segments, _ = self._load_whisper().transcribe(
            audio,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    # ---- documents --------------------------------------------------------
    def _document_context(self, query: str) -> str:
        with self._state_lock:
            docs = dict(self._documents)
        if not docs:
            return ""
        total = sum(len(text) for text in docs.values())
        if total <= DOC_FULL_LIMIT:
            return "\n\n".join(f"[{name}]\n{text}" for name, text in docs.items())

        query_words = _words(query)
        scored: list[tuple[int, int, str, str]] = []
        order = 0
        for name, text in docs.items():
            for chunk in _chunk_text(text):
                scored.append((len(query_words & _words(chunk)), order, name, chunk))
                order += 1
        # best score first; ties keep document order (so "summarize" gets the start)
        scored.sort(key=lambda item: (-item[0], item[1]))
        picked: list[tuple[int, str, str]] = []
        used = 0
        for _, position, name, chunk in scored:
            if used + len(chunk) > DOC_CONTEXT_CHARS:
                continue
            picked.append((position, name, chunk))
            used += len(chunk)
        picked.sort(key=lambda item: item[0])  # back to reading order
        return "\n\n".join(f"[{name}]\n{chunk}" for _, name, chunk in picked)

    # ---- language model ---------------------------------------------------
    def _chat_payload(self, user_text: str, voice: bool, stream: bool) -> dict[str, Any]:
        with self._state_lock:
            model_name = self._model_name
            temperature = self._temperature
            history = list(self._history[-8:])
        system = SYSTEM_PROMPT if voice else TEXT_SYSTEM_PROMPT
        context = self._document_context(user_text)
        if context:
            system += DOC_INSTRUCTIONS.format(context=context)
        messages = [{"role": "system", "content": system}]
        messages.extend({"role": h["role"], "content": h["content"][:1200]} for h in history)
        messages.append({"role": "user", "content": user_text})
        return {
            "model": model_name,
            "messages": messages,
            "stream": stream,
            "keep_alive": "15m",
            "options": {
                "temperature": temperature,
                "num_ctx": NUM_CTX,
                "num_predict": 120 if voice else 700,
                "num_thread": max(1, min(4, (os.cpu_count() or 2) - 1)),
            },
        }

    def _remember(self, user_text: str, answer: str) -> None:
        with self._state_lock:
            self._history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": answer},
                ]
            )
            self._history = self._history[-10:]

    def _stream_sentences(self, user_text: str, token: int) -> Iterator[str]:
        """Stream the Ollama reply and yield each sentence the moment it is complete."""
        payload = self._chat_payload(user_text, voice=True, stream=True)
        with requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=(3, 90)
        ) as response:
            response.raise_for_status()
            buffer = ""
            for line in response.iter_lines():
                if not self._is_current(token):
                    return  # interrupted: closing the response stops generation
                if not line:
                    continue
                data = json.loads(line)
                buffer += data.get("message", {}).get("content", "")
                sentences, buffer = _split_sentences(buffer)
                for sentence in sentences:
                    yield sentence
                if data.get("done"):
                    break
            sentences, _ = _split_sentences(buffer, final=True)
            for sentence in sentences:
                yield sentence

    def _answer(self, user_text: str) -> str:
        """Complete (non-streaming) text-chat answer."""
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=self._chat_payload(user_text, voice=False, stream=False),
            timeout=(3, 240),
        )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "").strip()
        if not answer:
            answer = "Mujhe jawab nahin mila. Dobara try karein."
        self._remember(user_text, answer)
        return answer

    def ask_text(self, user_text: str, speak: bool = False) -> str:
        """Typed message -> Ollama -> text answer (optionally spoken in the call)."""
        self._log_add("user", user_text, "text")
        try:
            answer = self._answer(user_text)
        except requests.RequestException:
            answer = "⚠️ Ollama offline hai — Ollama start karke dobara try karein."
            self._log_add("assistant", answer, "text")
            return answer
        except Exception as exc:
            detail = str(exc).strip().splitlines()[0][:150] if str(exc).strip() else "unknown error"
            answer = f"⚠️ Local error: {detail}"
            self._log_add("assistant", answer, "text")
            return answer
        self._log_add("assistant", answer, "text")
        if speak:
            with self._state_lock:
                active, token = self._active, self._generation
            if active:
                self._put_job(("say", token, answer))
        return answer

    # ---- text to speech ---------------------------------------------------
    @staticmethod
    def _synthesize(engine: Any, text: str) -> np.ndarray:
        path = Path(tempfile.gettempdir()) / f"offline-chatbot-{uuid4().hex}.wav"
        try:
            engine.save_to_file(_spoken_text(text), str(path))
            engine.runAndWait()
            for _ in range(20):
                if path.exists() and path.stat().st_size > 44:
                    break
                time.sleep(0.05)
            return _wav_to_pcm(path)
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _speak(self, engine: Any, token: int, text: str, first: bool) -> bool:
        """Synthesize one piece of speech and queue it. Returns False if interrupted."""
        pcm = self._synthesize(engine, text)
        if not self._is_current(token) or pcm.size == 0:
            return False
        if first:
            self.source.clear()
        self.source.push(pcm)
        with self._state_lock:
            now = time.monotonic()
            self._speaking_until = max(now, self._speaking_until) + pcm.size / OUTPUT_RATE
            self._error = ""
        return True

    # ---- worker thread ----------------------------------------------------
    def _worker(self) -> None:
        pythoncom = None
        engine = None
        try:
            try:
                import pythoncom as _pythoncom

                pythoncom = _pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass
            engine = pyttsx3.init()
            engine.setProperty("rate", 184)
            engine.setProperty("volume", 1.0)
            while not self._stop_event.is_set():
                try:
                    kind, token, payload = self._jobs.get(timeout=0.5)
                except queue.Empty:
                    continue
                if kind == "stop":
                    break
                if not self._is_current(token):
                    continue
                try:
                    if kind == "audio":
                        user_text = self._transcribe(payload)
                        if not user_text or not self._is_current(token):
                            if self._is_current(token):
                                self._set_phase("Sun raha hoon")
                            continue
                        self._log_add("user", user_text, "voice")
                        with self._state_lock:
                            self._live_user = user_text
                            self._live_ai = ""
                        self._set_phase("Jawab soch raha hoon")

                        # Realtime pipeline: LLM tokens -> sentence -> speech, while the
                        # model is still writing the rest of the answer.
                        spoken: list[str] = []
                        try:
                            for sentence in self._stream_sentences(user_text, token):
                                clean = _spoken_text(sentence)
                                if not clean:
                                    continue
                                if not self._speak(engine, token, clean, first=not spoken):
                                    break
                                spoken.append(clean)
                                with self._state_lock:
                                    self._live_ai = " ".join(spoken)
                                    self._phase = "Jawab soch raha hoon"
                        finally:
                            if spoken:
                                answer = " ".join(spoken)
                                self._remember(user_text, answer)
                                self._log_add("assistant", answer, "voice")
                        if not spoken and self._is_current(token):
                            self._speak(engine, token, "Mujhe jawab nahin mila. Dobara bolain.", first=True)
                    else:
                        self._set_phase("Voice bana raha hoon")
                        self._speak(engine, token, str(payload), first=True)
                    if self._is_current(token):
                        self._set_phase("Sun raha hoon")
                except requests.RequestException:
                    if self._is_current(token):
                        self._set_phase(
                            "Sun raha hoon",
                            "Ollama offline hai — Ollama start karke call dobara karein",
                        )
                except Exception as exc:
                    if self._is_current(token):
                        detail = str(exc).strip().splitlines()[0][:110]
                        self._set_phase(
                            "Sun raha hoon", f"Local voice error: {detail or 'unknown error'}"
                        )
        finally:
            try:
                if engine is not None:
                    engine.stop()
            finally:
                if pythoncom is not None:
                    pythoncom.CoUninitialize()


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Offline AI Assistant",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at 85% -10%, #12383b 0%, #111827 42%, #0b1020 100%); color: #f7f8fb; }
    header[data-testid="stHeader"] { background: transparent; }
    .stAppDeployButton, [data-testid="stToolbar"] { display: none !important; }
    .block-container { max-width: 800px; padding-top: 1.6rem; padding-bottom: 0; }
    h1 { font-size: 1.3rem !important; letter-spacing: -0.01em; padding: 0 !important; margin: 0; }

    /* status pill */
    .status-pill { display: flex; align-items: center; justify-content: center; gap: .55rem; min-height: 38px; border: 1px solid #27406b; border-radius: 999px; background: rgba(17,28,53,.85); font-size: .86rem; margin: .1rem 0 .6rem; }
    .status-dot { width: 9px; height: 9px; border-radius: 50%; background: #64748b; }
    .status-pill.error { border-color: #ef4444; }
    .status-pill.error .status-dot { background: #ef4444; }

    /* greeting shown on an empty chat */
    .hero { text-align: center; font-size: 2rem; font-weight: 600; letter-spacing: -0.02em; margin: 16vh 0 .3rem; }
    .hero-sub { text-align: center; color: #94a3b8; margin-bottom: 1.4rem; }

    /* chat bubbles */
    [data-testid="stChatMessage"] { background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.06); border-radius: 16px; padding: .7rem 1rem; }

    /* ---------- stable composer ---------- */
    .st-key-composer_wrap {
        position: sticky; bottom: 0; z-index: 50;
        padding: 10px 0 14px;
        background: transparent !important;
    }
    .st-key-composer {
        box-sizing: border-box; width: 100%; min-height: 68px;
        display: flex; align-items: center;
        background: #1b2338; border: 1px solid #33415f;
        border-radius: 36px; padding: 9px 12px;
        box-shadow: 0 5px 18px rgba(0,0,0,.20);
        transition: border-color .18s ease, box-shadow .18s ease;
    }
    .st-key-composer:focus-within {
        border-color: #538df5; box-shadow: 0 0 0 2px rgba(59,130,246,.12);
    }
    .st-key-composer [data-testid="stHorizontalBlock"] {
        display: flex !important; flex-wrap: nowrap !important;
        align-items: center !important; gap: 8px !important;
        width: 100%;
    }
    .st-key-composer [data-testid="stColumn"] {
        min-width: 46px !important; flex: 0 0 46px !important;
        width: 46px !important; padding: 0 !important;
    }
    .st-key-composer [data-testid="stColumn"]:nth-child(2) {
        min-width: 0 !important; width: auto !important; flex: 1 1 auto !important;
    }
    .st-key-composer [data-testid="stTextInput"] { margin: 0 !important; }
    .st-key-composer [data-testid="stTextInput"] > div,
    .st-key-composer [data-testid="stTextInputRootElement"],
    .st-key-composer [data-baseweb="input"],
    .st-key-composer [data-baseweb="base-input"] {
        min-height: 46px !important; background: transparent !important;
        border: 0 !important; box-shadow: none !important;
    }
    .st-key-composer [data-testid="stTextInput"] input {
        height: 46px !important; padding: 0 5px !important;
        background: transparent !important; color: #f8fafc !important;
        font-size: 16px !important;
    }
    .st-key-composer [data-testid="InputInstructions"] { display: none !important; }

    /* Native Streamlit icon buttons: no popover chevrons or pseudo-icon overlays. */
    .st-key-composer .stButton > button {
        box-sizing: border-box !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
        width: 42px !important; min-width: 42px !important;
        height: 42px !important; min-height: 42px !important;
        padding: 0 !important; margin: 0 auto !important;
        border-radius: 50% !important; border: 0 !important;
        background: transparent !important; color: #e2e8f0 !important;
        box-shadow: none !important; transition: background .15s ease, transform .15s ease !important;
    }
    .st-key-composer .stButton > button:hover {
        background: rgba(255,255,255,.09) !important; transform: translateY(-1px);
    }
    .st-key-composer .stButton > button p { display: none !important; }
    .st-key-composer .stButton > button [data-testid="stIconMaterial"] {
        display: inline-flex !important; font-size: 23px !important; line-height: 1 !important;
        color: currentColor !important;
    }
    .st-key-composer .st-key-live_voice_btn button,
    .st-key-composer .st-key-send_btn button {
        width: 46px !important; min-width: 46px !important;
        height: 46px !important; min-height: 46px !important;
        background: #3b82f6 !important; color: #fff !important;
    }
    .st-key-composer .st-key-live_voice_btn button:hover,
    .st-key-composer .st-key-send_btn button:hover {
        background: #2563eb !important; box-shadow: 0 4px 14px rgba(59,130,246,.28) !important;
    }
    .st-key-composer .st-key-live_voice_btn button [data-testid="stIconMaterial"],
    .st-key-composer .st-key-send_btn button [data-testid="stIconMaterial"] {
        font-size: 24px !important; color: #fff !important;
    }
    /* All composer action controls use one consistent circular size. */
    .st-key-composer .st-key-attach_btn button,
    .st-key-composer .st-key-live_voice_btn button,
    .st-key-composer .st-key-send_btn button,
    .st-key-composer [data-testid="stAudioInput"] button {
        box-sizing: border-box !important;
        width: 46px !important; min-width: 46px !important; max-width: 46px !important;
        height: 46px !important; min-height: 46px !important; max-height: 46px !important;
        padding: 0 !important;
        border-radius: 50% !important;
        display: inline-flex !important; align-items: center !important; justify-content: center !important;
    }
    .st-key-composer .st-key-attach_btn button {
        background: #111827 !important; color: #f8fafc !important;
        border: 1px solid #3b4558 !important;
    }
    .st-key-composer .st-key-live_voice_btn button,
    .st-key-composer .st-key-send_btn button {
        background: #3b82f6 !important; color: #fff !important;
        border: 1px solid #60a5fa !important;
    }
    .st-key-composer .st-key-live_voice_btn button:hover,
    .st-key-composer .st-key-send_btn button:hover {
        background: #2563eb !important; box-shadow: 0 4px 14px rgba(59,130,246,.28) !important;
    }
    .st-key-composer .st-key-attach_btn button [data-testid="stIconMaterial"],
    .st-key-composer .st-key-live_voice_btn button [data-testid="stIconMaterial"],
    .st-key-composer .st-key-send_btn button [data-testid="stIconMaterial"] {
        font-size: 23px !important; line-height: 1 !important;
    }
    .st-key-composer .st-key-live_voice_btn button [data-testid="stIconMaterial"],
    .st-key-composer .st-key-send_btn button [data-testid="stIconMaterial"] {
        color: #fff !important;
    }
    .st-key-composer [data-testid="stAudioInput"],
    .st-key-composer [data-testid="stAudioInput"] > div,
    .st-key-composer [data-testid="stAudioInput"] > div > div {
        box-sizing: border-box !important;
        width: 46px !important; min-width: 46px !important; max-width: 46px !important;
        height: 46px !important; min-height: 46px !important; max-height: 46px !important;
        margin: 0 !important; padding: 0 !important;
        background: transparent !important; border: 0 !important;
        box-shadow: none !important; overflow: visible !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
    }
    .st-key-composer [data-testid="stAudioInput"] button {
        box-sizing: border-box !important;
        width: 46px !important; min-width: 46px !important; max-width: 46px !important;
        height: 46px !important; min-height: 46px !important; max-height: 46px !important;
        padding: 0 !important; margin: 0 !important;
        border-radius: 50% !important;
        background: #111827 !important; border: 1px solid #3b4558 !important;
        color: #e2e8f0 !important;
        display: inline-flex !important; align-items: center !important; justify-content: center !important;
    }
    .st-key-composer [data-testid="stAudioInput"] button:hover {
        background: #263247 !important;
    }
    .st-key-composer [data-testid="stAudioInput"] button svg {
        width: 22px !important; height: 22px !important;
    }
    .st-key-composer button:focus-visible {
        outline: 2px solid #93c5fd !important; outline-offset: 3px !important;
    }

    /* Keep the WebRTC component mounted and able to request microphone permission.
       It is tiny and transparent, but NOT display:none and NOT removed from layout. */
    .st-key-webrtc_engine {
        position: fixed !important; right: 12px !important; bottom: 12px !important;
        width: 320px !important; height: 84px !important;
        min-width: 320px !important; min-height: 84px !important;
        overflow: hidden !important; opacity: 0.01 !important;
        pointer-events: none !important; z-index: 70 !important;
        border: 1px solid rgba(96,165,250,.45) !important;
        border-radius: 14px !important; background: #111827 !important;
    }
    .st-key-webrtc_engine iframe {
        display: block !important; width: 320px !important; height: 84px !important;
        min-width: 320px !important; min-height: 84px !important; border: 0 !important;
    }
    /* Live mode is a clean, dedicated screen; do not leave the composer over it. */
    body:has(.live-overlay) .st-key-composer_wrap { display: none !important; }
    body:has(.live-overlay) .st-key-live_controls {
        bottom: 24px !important; z-index: 80 !important;
    }
    /* Upload / voice-note panels open above the composer, never inside a dropdown. */
    .st-key-attachment_panel, .st-key-voice_note_panel {
        background: rgba(27,35,56,.96); border: 1px solid #33415f;
        border-radius: 16px; padding: 12px 14px; margin: 0 0 10px;
    }
    .mic-glyph.live { background: #ef4444; color: #fff; font-size: 15px; filter: none; box-shadow: 0 0 0 4px rgba(239,68,68,.2); }

    /* ---------- LIVE voice mode (full screen, Gemini-Live style) ---------- */
    .live-overlay { position: fixed; inset: 0; z-index: 40; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1rem; padding: 2rem 1.5rem 11rem; text-align: center; background: radial-gradient(circle at 50% 34%, #16355c 0%, #0d1428 62%, #090d1a 100%); }
    .orb-wrap { position: relative; width: 190px; height: 190px; display: flex; align-items: center; justify-content: center; margin-bottom: .6rem; }
    .orb { width: 140px; height: 140px; border-radius: 50%; background: conic-gradient(from 0deg, #38bdf8, #818cf8, #c084fc, #38bdf8); box-shadow: 0 0 70px rgba(99,102,241,.55), inset 0 0 30px rgba(255,255,255,.25); animation: orb-breathe 4s ease-in-out infinite; will-change: transform; }
    .ring { position: absolute; width: 140px; height: 140px; border-radius: 50%; border: 2px solid rgba(129,140,248,.55); opacity: 0; }
    .live-overlay.listening .ring { animation: ring-out 2.6s ease-out infinite; }
    .live-overlay.listening .ring.r2 { animation-delay: 1.3s; }
    .live-overlay.working .orb { animation: orb-spin 2.4s linear infinite; will-change: transform; }
    .live-overlay.speaking .orb { animation: orb-talk 1.1s ease-in-out infinite; will-change: transform; box-shadow: 0 0 90px rgba(56,189,248,.7), inset 0 0 30px rgba(255,255,255,.3); }
    .live-overlay.error .orb { background: conic-gradient(#ef4444, #f97316, #ef4444); box-shadow: 0 0 70px rgba(239,68,68,.5); animation: none; }
    .live-overlay.muted .orb { filter: grayscale(1) brightness(.55); animation: none; box-shadow: none; }
    @keyframes orb-breathe { 0%,100% { transform: scale(1); } 50% { transform: scale(1.07); } }
    @keyframes orb-spin { to { transform: rotate(360deg); } }
    @keyframes orb-talk { 0%,100% { transform: scale(1); } 25% { transform: scale(1.16); } 50% { transform: scale(.97); } 75% { transform: scale(1.1); } }
    @keyframes ring-out { 0% { transform: scale(1); opacity: .45; } 100% { transform: scale(1.35); opacity: 0; } }
    @media (prefers-reduced-motion: reduce) { .orb, .ring { animation: none !important; } * { scroll-behavior: auto !important; } }
    .live-state { color: #cbd5e1; font-size: 1rem; letter-spacing: .01em; }
    .cap-user { color: #8b9ab3; font-size: 1rem; max-width: 640px; }
    .cap-ai { color: #f1f5f9; font-size: 1.3rem; line-height: 1.45; max-width: 680px; }

    /* Mute / End buttons floating above the composer while a call is running */
    .st-key-live_controls { position: fixed; bottom: 112px; left: 50%; transform: translateX(-50%); z-index: 60; width: min(360px, 90vw); }
    .st-key-live_controls [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .6rem !important; }
    .st-key-live_controls [data-testid="stColumn"] { min-width: 0 !important; }
    .st-key-live_controls button { border-radius: 999px; min-height: 46px; }

    div[data-testid="stAudio"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Local chat-history store (same folder as app.py).
CHAT_STORE_PATH = Path(__file__).with_name("chat_history.json")
def _read_chat_store() -> dict[str, Any]:
    try:
        raw = json.loads(CHAT_STORE_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
def _write_chat_store(store: dict[str, Any]) -> None:
    try:
        temp_path = CHAT_STORE_PATH.with_suffix(".tmp")
        temp_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(CHAT_STORE_PATH)
    except OSError:
        st.warning("Chat history file save nahin ho saki. Folder permissions check karein.")

if "_chat_store" not in st.session_state:
    st.session_state["_chat_store"] = _read_chat_store()
chat_store: dict[str, Any] = st.session_state["_chat_store"]
if not chat_store:
    first_id = uuid4().hex
    chat_store[first_id] = {"title": "New chat", "updated": time.time(), "history": [], "messages": []}
    st.session_state["_active_chat_id"] = first_id
    st.session_state["_loaded_chat_id"] = None
if "_active_chat_id" not in st.session_state:
    first_id = max(chat_store, key=lambda key: chat_store[key].get("updated", 0))
    st.session_state["_active_chat_id"] = first_id
    st.session_state["_loaded_chat_id"] = None


with st.sidebar:
    st.title("Chats")
    if st.button("＋ New chat", key="sidebar_new_chat", use_container_width=True, type="primary"):
        new_id = uuid4().hex
        chat_store[new_id] = {"title": "New chat", "updated": time.time(), "history": [], "messages": []}
        st.session_state["_active_chat_id"] = new_id
        _write_chat_store(chat_store)
        st.rerun()

    st.caption("Recent conversations")
    ordered_chats = sorted(
        chat_store.items(), key=lambda pair: pair[1].get("updated", 0), reverse=True
    )
    for chat_id, chat_data in ordered_chats[:30]:
        title = str(chat_data.get("title") or "New chat")[:34]
        is_active = chat_id == st.session_state.get("_active_chat_id")
        label = ("● " if is_active else "　") + title
        if st.button(label, key=f"open_chat_{chat_id}", use_container_width=True, help="Open this conversation"):
            st.session_state["_active_chat_id"] = chat_id
            st.rerun()

    st.divider()
    with st.expander("⚙️ Voice settings", expanded=False):
        selected_model = st.selectbox("Ollama model", [DEFAULT_MODEL], index=0)
        selected_whisper = st.selectbox("Voice understanding", ["tiny", "base"], index=0)
        selected_temperature = st.slider("Creativity", 0.0, 0.8, 0.25, 0.05)
        speak_text_replies = st.toggle(
            "Text ke jawab bhi bol kar sunao",
            value=False,
            help="Sirf tab kaam karta hai jab voice call (mic button) chal rahi ho.",
        )
    st.caption("Chat history isi computer par chat_history.json mein save hoti hai.")
    st.caption("WebRTC + Whisper + Ollama + Windows voice — sab local machine par.")

# Stop the obsolete local listener left by the previous build.
if not st.session_state.get("_legacy_voice_cleaned_v8", False):
    legacy_mic = st.session_state.get("local_microphone")
    if legacy_mic is not None:
        try:
            legacy_mic.stop()
        except Exception:
            pass
    for legacy_key in (
        "local_microphone",
        "live_voice_enabled",
        "pending_voice_audio",
        "voice_output_audio",
    ):
        st.session_state.pop(legacy_key, None)
    st.session_state["_legacy_voice_cleaned_v8"] = True

session_token = st.session_state.setdefault("_webrtc_session_token_v8", uuid4().hex)
pcm_source = create_pcm_audio_source_track(
    key=f"assistant-speaker-{session_token}",
    sample_rate=OUTPUT_RATE,
    ptime=0.02,
    lifecycle_scope="streamlit-session",
)
bridge = st.session_state.get("_realtime_voice_bridge_v8")
if getattr(bridge, "runtime_version", None) != APP_RUNTIME_VERSION:
    try:
        if bridge is not None:
            shutdown = getattr(bridge, "shutdown", bridge.pause)
            shutdown()
    except Exception:
        pass
    bridge = RealtimeVoiceBridge(pcm_source)
    st.session_state["_realtime_voice_bridge_v8"] = bridge
else:
    bridge.source = pcm_source
bridge.configure(selected_model, selected_whisper, selected_temperature)

# On chat switch, save the previous bridge state, then load the selected chat.
active_chat_id = st.session_state["_active_chat_id"]
loaded_chat_id = st.session_state.get("_loaded_chat_id")
if loaded_chat_id != active_chat_id:
    if loaded_chat_id in chat_store:
        chat_store[loaded_chat_id].update(bridge.export_conversation())
        chat_store[loaded_chat_id]["updated"] = time.time()
    bridge.load_conversation(chat_store.get(active_chat_id))
    st.session_state["_loaded_chat_id"] = active_chat_id

microphone_sink = create_audio_sink_track(
    bridge.on_audio_frame,
    key=f"browser-microphone-{session_token}",
    lifecycle_scope="streamlit-session",
)

# WebRTC lifecycle callback: start/stop the local audio bridge as soon as
# the browser component changes connection state.
def on_webrtc_change() -> None:
    ctx = st.session_state.get("offline-realtime-voice-v11")
    if ctx is not None and getattr(ctx.state, "playing", False):
        bridge.start()
        bridge.set_muted(st.session_state.get("_muted", False))
    elif ctx is not None and not getattr(ctx.state, "signalling", False):
        bridge.pause()


# Hidden WebRTC engine controlled by the visible Streamlit button.
st.session_state.setdefault("_voice_requested", False)
with st.container(key="webrtc_engine"):
    call = webrtc_streamer(
        key="offline-realtime-voice-v11",
        mode=WebRtcMode.SENDRECV,
        on_change=on_webrtc_change,
        rtc_configuration={"iceServers": []},
        media_stream_constraints={"audio": True, "video": False},
        source_audio_track=pcm_source.track,
        sink_audio_track=microphone_sink,
        sendback_audio=True,
        sendback_video=False,
        desired_playing_state=bool(st.session_state.get("_voice_requested", False)),
        video_html_attrs={"hidden": True, "style": {"display": "none", "height": 0, "width": 0}},
        audio_html_attrs={"autoPlay": True, "controls": False, "muted": False, "playsInline": True},
        translations={
            "start": "Start",
            "stop": "Stop",
            "select_device": "Mic",
            "device_ask_permission": "Browser mein microphone Allow karein",
            "device_not_available": "Microphone available nahin hai",
            "device_access_denied": "Microphone permission deny ho gayi",
            "select_microphone": "Microphone",
            "mute_microphone": "Mute",
            "unmute_microphone": "Unmute",
        },
        media_toggle_controls=False,
        backend="aiortc",
    )

# Uploaded documents live in the session: {file name: extracted text}
documents: dict[str, str] = st.session_state.setdefault("_docs", {})
bridge.sync_documents(documents)
st.session_state.setdefault("_up_n", 0)


def ingest_files(files: list[Any]) -> list[str]:
    """Read attached PDF/Word files and store their text. Returns status lines."""
    notes: list[str] = []
    for item in files:
        try:
            text = extract_document_text(item.name, item.getvalue())
            if text:
                documents[item.name] = text
                notes.append(f"✅ **{item.name}** load ho gayi ({len(text):,} characters). Ab is ke baare mein poochein.")
            else:
                notes.append(f"⚠️ **{item.name}** mein text nahin mila (shayad scanned/image PDF hai).")
        except Exception as exc:
            notes.append(f"⚠️ **{item.name}** parh nahin saka: {exc}")
    return notes


def submit_text() -> None:
    """Called by Enter in the text box or by the blue send button."""
    text = (st.session_state.get("composer_text") or "").strip()
    if text:
        st.session_state["_outbox"] = text
        st.session_state["composer_text"] = ""


def toggle_mute() -> None:
    st.session_state["_muted"] = not st.session_state.get("_muted", False)


def start_call() -> None:
    st.session_state["_voice_requested"] = True
    st.session_state["_end_call"] = False


def end_call() -> None:
    st.session_state["_voice_requested"] = False
    st.session_state["_end_call"] = True


def _one_line(text: str) -> str:
    return " ".join(text.split())


# ---- Header -----------------------------------------------------------------
head_left, head_right = st.columns([5, 1.5], vertical_alignment="center")
with head_left:
    st.title("🎙️ Offline AI Assistant")
with head_right:
    clear_clicked = st.button("🧹 Clear chat", use_container_width=True)
if clear_clicked:
    bridge.forget()
    current_id = st.session_state.get("_active_chat_id")
    if current_id in chat_store:
        chat_store[current_id].update({"title": "New chat", "updated": time.time(), "history": [], "messages": []})
        _write_chat_store(chat_store)
    st.toast("Current conversation clear ho gayi.")
    st.rerun()


@st.fragment(run_every=1.2)
def render_live_status() -> None:
    message, css_class = bridge.snapshot()
    if css_class == "off":
        return

    # Call is running: full-screen "Live" view with an animated orb and live captions appears.
    user_caption, ai_caption = bridge.live_captions()
    overlay = (
        f'<div class="live-overlay {css_class}">'
        '<div class="orb-wrap"><span class="ring r1"></span><span class="ring r2"></span>'
        '<div class="orb"></div></div>'
        f'<div class="live-state">{html.escape(message)}</div>'
    )
    if user_caption:
        overlay += f'<div class="cap-user">{html.escape(_one_line(user_caption))}</div>'
    if ai_caption:
        overlay += f'<div class="cap-ai">{html.escape(_one_line(ai_caption))}</div>'
    overlay += "</div>"
    st.markdown(overlay, unsafe_allow_html=True)


render_live_status()

# ---- Loaded documents -------------------------------------------------------
if documents:
    with st.expander(f"📎 Loaded documents ({len(documents)})", expanded=False):
        for doc_name in list(documents):
            info_col, remove_col = st.columns([6, 1], vertical_alignment="center")
            info_col.caption(f"{doc_name} — {len(documents[doc_name]):,} characters")
            if remove_col.button("✕", key=f"remove-{doc_name}", help="Hata dein"):
                del documents[doc_name]
                bridge.sync_documents(documents)
                st.rerun()


# ---- Chat history -----------------------------------------------------------
@st.fragment(run_every=4.0)
def render_chat() -> None:
    messages = bridge.messages()
    _cid = st.session_state.get("_active_chat_id")
    if _cid in chat_store:
        _snap = bridge.export_conversation()
        _first = next((m.get("content", "") for m in _snap["messages"] if m.get("role") == "user"), "")
        chat_store[_cid].update(_snap)
        chat_store[_cid]["title"] = " ".join(str(_first).split())[:42] or "New chat"
        chat_store[_cid]["updated"] = time.time()
        _write_chat_store(chat_store)
    if not messages:
        st.markdown(
            '<div class="hero">Aaj kya poochna hai?</div>'
            '<div class="hero-sub">Likhein, ＋ se PDF/Word attach karein, ya mic dabake live bolain.</div>',
            unsafe_allow_html=True,
        )
        return
    for item in messages:
        with st.chat_message(item["role"]):
            if item["source"] == "voice":
                st.caption("🎙️ voice")
            st.markdown(item["content"])


render_chat()
pending_area = st.container()  # a freshly typed message + spinner appear here
live_slot = st.container()     # Mute / End buttons appear here during a call

# ---- Composer: attachment / voice note panels + stable icon row ----------------
st.session_state.setdefault("_show_upload_panel", False)
st.session_state.setdefault("_show_voice_note_panel", False)
uploaded_files = []
voice_note = None

if st.session_state["_show_upload_panel"]:
    with st.container(key="attachment_panel"):
        panel_col, close_col = st.columns([8, 1], vertical_alignment="center")
        with panel_col:
            uploaded_files = st.file_uploader(
                "PDF ya Word file attach karein",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key=f"uploader-{st.session_state['_up_n']}",
            )
        with close_col:
            if st.button(" ", key="close_upload_panel", icon=":material/close:", help="Band karein"):
                st.session_state["_show_upload_panel"] = False
                st.rerun()

with st.container(key="composer_wrap"):
    with st.container(key="composer"):
        attach_col, text_col, note_col, live_col, send_col = st.columns(
            [0.62, 7.4, 1.05, 0.82, 0.82], vertical_alignment="center"
        )
        with attach_col:
            if st.button(" ", key="attach_btn", icon=":material/add:", help="File attach karein"):
                st.session_state["_show_upload_panel"] = not st.session_state["_show_upload_panel"]
                st.rerun()
        with text_col:
            st.text_input(
                "Message",
                key="composer_text",
                placeholder="Kuch bhi poochein...",
                label_visibility="collapsed",
                on_change=submit_text,
            )
        with note_col:
            # One click on this native mic starts recording. Stop returns the
            # audio bytes, which are transcribed and sent automatically below.
            voice_note = st.audio_input(
                "Voice note",
                key="voice_note_input",
                label_visibility="collapsed",
            )
        with live_col:
            st.button(
                " ",
                key="live_voice_btn",
                icon=":material/graphic_eq:",
                on_click=end_call if call.state.playing else start_call,
                help="Live voice call band karein" if call.state.playing else "Live voice call shuru karein",
                type="primary",
            )
        with send_col:
            st.button(
                " ",
                key="send_btn",
                icon=":material/arrow_upward:",
                on_click=submit_text,
                help="Message bhejein",
                type="primary",
            )

if call.state.playing:
    bridge.start()
    bridge.set_muted(st.session_state.get("_muted", False))
    with live_slot:
        with st.container(key="live_controls"):
            mute_col, end_col = st.columns(2)
            muted_now = st.session_state.get("_muted", False)
            mute_col.button(
                "🎤 Unmute" if muted_now else "🔇 Mute",
                key="mute_btn",
                on_click=toggle_mute,
                use_container_width=True,
            )
            end_col.button(
                "⏹ End call",
                key="end_btn",
                on_click=end_call,
                use_container_width=True,
                type="primary",
            )
            with bridge._state_lock:
                _mic_frames = bridge._audio_frames
                _mic_rms = bridge._last_audio_rms
                _mic_age = time.monotonic() - bridge._last_audio_at if bridge._last_audio_at else None
            with bridge._state_lock:
                _mic_error = bridge._audio_error
            if _mic_error:
                st.error(_mic_error)
            elif _mic_age is None or _mic_age > 4.0:
                st.warning("Browser se microphone audio receive nahi ho rahi. WebRTC panel mein Start/permission prompt complete karein, Chrome site settings mein Microphone = Allow karein, aur correct input device select karein.")
            else:
                st.caption(f"Microphone connected · input level {int(_mic_rms)} · frames {_mic_frames}")
else:
    st.session_state["_muted"] = False
    if st.session_state.get("_end_call"):
        st.session_state["_voice_requested"] = False
        st.session_state["_end_call"] = False
    bridge.pause()

# ---- Handle attached files / a submitted message ------------------------------
outbox = st.session_state.pop("_outbox", "")
if uploaded_files or outbox or voice_note:
    with pending_area:
        if uploaded_files:
            with st.spinner("Files parh raha hoon..."):
                notes = ingest_files(list(uploaded_files))
            bridge.sync_documents(documents)
            bridge.log_message("user", "📎 " + ", ".join(f.name for f in uploaded_files))
            bridge.log_message("assistant", "\n\n".join(notes))
            st.session_state["_up_n"] += 1  # resets the uploader
        if voice_note:
            audio_bytes = voice_note.getvalue()
            fingerprint = hashlib.sha256(audio_bytes).hexdigest()
            if fingerprint != st.session_state.get("_last_voice_note_hash"):
                st.session_state["_last_voice_note_hash"] = fingerprint
                st.session_state["_show_voice_note_panel"] = False
                with st.spinner("Voice note samajh raha hoon..."):
                    try:
                        pcm = recorded_audio_to_pcm(audio_bytes)
                        transcript = bridge._transcribe(pcm) if pcm.size else ""
                        if transcript:
                            bridge.log_message("user", transcript, "voice")
                            with st.chat_message("assistant"):
                                with st.spinner("Jawab soch raha hoon..."):
                                    bridge.ask_text(transcript, speak=speak_text_replies)
                        else:
                            bridge.log_message(
                                "assistant",
                                "Voice note samajh nahin aayi. Dobara record karein.",
                                "text",
                            )
                    except Exception as exc:
                        detail = str(exc).strip().splitlines()[0][:140] if str(exc).strip() else "unknown error"
                        bridge.log_message("assistant", f"Voice note process nahin ho saki: {detail}", "text")
        if outbox:
            with st.chat_message("user"):
                st.markdown(outbox)
            with st.chat_message("assistant"):
                with st.spinner("Jawab soch raha hoon..."):
                    bridge.ask_text(outbox, speak=speak_text_replies)
    st.rerun()

# Persist the current conversation after every completed interaction.
_current_id = st.session_state.get("_active_chat_id")
if _current_id in chat_store:
    _snapshot = bridge.export_conversation()
    _messages = _snapshot["messages"]
    _first_user = next((m.get("content", "") for m in _messages if m.get("role") == "user"), "")
    _title = " ".join(str(_first_user).split())[:42] or "New chat"
    chat_store[_current_id].update(_snapshot)
    chat_store[_current_id]["title"] = _title
    chat_store[_current_id]["updated"] = time.time()
    _write_chat_store(chat_store)
