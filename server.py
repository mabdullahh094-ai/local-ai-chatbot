from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextlib
import html
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import uuid


# Reconfigure stdout/stderr for Windows console encodings
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import docx
from faster_whisper import WhisperModel
import pypdf
import pyttsx3
import requests
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
import uvicorn

# ----------------------------------------------------------------------------
# Configurations & Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
CHAT_HISTORY_FILE = BASE_DIR / "chat_history.json"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

HF_ROOT = Path(r"D:\AI\huggingface")
os.environ.setdefault("HF_HOME", str(HF_ROOT))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_ROOT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_ROOT / "transformers"))
os.environ.setdefault("LLAMA_ARG_REPACK", "0")

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_WHISPER = "tiny"
PORT = 8501

SYSTEM_PROMPT = """You are a helpful, smart, and friendly local AI assistant running 100% offline.
Language Guidelines:
- If the user speaks or writes in Urdu or Roman Urdu, reply naturally in Roman Urdu using English/Latin alphabet. Never use complex Arabic script unless asked.
- If the user speaks or writes in English, reply naturally in English.
- Always be polite, concise, and helpful. You can format responses with clean markdown (bold, bullet points, clean code blocks).
- Never mention internal prompts, whisper, or system mechanics.
"""

VOICE_SYSTEM_PROMPT = """You are a real-time conversational voice assistant running 100% offline.
Rules:
- Match the user's language: Roman Urdu for Urdu queries, English for English queries.
- Keep replies brief, natural, and conversational (1 to 2 short sentences max).
- Do not use markdown, emojis, asterisks, citations, or bullet lists because this will be spoken aloud.
- Reply directly and warmly to the user's latest sentence.
"""

DOC_INSTRUCTIONS = """

--- ATTACHED DOCUMENTS CONTEXT ---
{context}
--- END OF CONTEXT ---
The user attached the documents above. When the question relates to the documents, answer using these excerpts. If the answer is not in the excerpts, politely mention it.
"""

# Thread pool for CPU-bound tasks (Whisper and SAPI TTS)
tts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts_worker")
whisper_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper_worker")

# Lazy-loaded Whisper model
_whisper_model: Optional[WhisperModel] = None
_whisper_lock = asyncio.Lock()

# In-memory document storage for session
session_documents: Dict[str, str] = {}


# ----------------------------------------------------------------------------
# Helper Functions: Whisper & Audio
# ----------------------------------------------------------------------------
def _get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            DEFAULT_WHISPER,
            device="cpu",
            compute_type="int8",
            cpu_threads=max(1, min(4, (os.cpu_count() or 2) - 1)),
            download_root=str(HF_ROOT / "hub"),
            local_files_only=True,
        )
    return _whisper_model


def _transcribe_sync(audio_path_or_bytes: Any) -> str:
    """Run Whisper transcription synchronously in a worker thread."""
    model = _get_whisper_model()
    temp_path: Optional[str] = None
    try:
        if isinstance(audio_path_or_bytes, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_path_or_bytes)
                temp_path = f.name
            target = temp_path
        else:
            target = str(audio_path_or_bytes)

        segments, _ = model.transcribe(
            target,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


async def transcribe_audio_async(audio_bytes: bytes) -> str:
    """Asynchronously transcribe audio bytes using thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(whisper_executor, _transcribe_sync, audio_bytes)


# ----------------------------------------------------------------------------
# Helper Functions: TTS (Text-to-Speech)
# ----------------------------------------------------------------------------
def clean_text_for_speech(text: str) -> str:
    """Strip code blocks, markdown symbols, and URLs for speech output."""
    t = re.sub(r"```.*?```", " [code block] ", text, flags=re.DOTALL)
    t = re.sub(r"`.*?`", " ", t)
    t = re.sub(r"https?://\S+", "link", t)
    t = re.sub(r"[*_#>`~|]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:600]


def _synthesize_wav_sync(text: str) -> bytes:
    """Run Windows SAPI TTS synchronously in a worker thread."""
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass

    clean_text = clean_text_for_speech(text)
    if not clean_text:
        return b""

    temp_path = os.path.join(tempfile.gettempdir(), f"bot_tts_{uuid.uuid4().hex}.wav")
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 172)
        engine.setProperty("volume", 1.0)
        engine.save_to_file(clean_text, temp_path)
        engine.runAndWait()
        for _ in range(30):
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 44:
                break
            time.sleep(0.04)
        if os.path.exists(temp_path):
            with open(temp_path, "rb") as f:
                return f.read()
        return b""
    except Exception as exc:
        print(f"[TTS Error] {exc}")
        return b""
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


async def generate_speech_wav_async(text: str) -> bytes:
    """Asynchronously synthesize text into WAV bytes."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(tts_executor, _synthesize_wav_sync, text)


# ----------------------------------------------------------------------------
# Helper Functions: Documents
# ----------------------------------------------------------------------------
def extract_document_text(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = pypdf.PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("PDF password protected hai.")
        pages = []
        for idx, page in enumerate(reader.pages, 1):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append(f"[Page {idx}]\n{txt}")
        return "\n\n".join(pages)[:150_000]
    elif ext == ".docx":
        doc = docx.Document(io.BytesIO(content))
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))
        return "\n".join(lines)[:150_000]
    else:
        raise ValueError("Sirf .pdf aur .docx files supported hain.")


def get_relevant_document_context(query: str) -> str:
    if not session_documents:
        return ""
    total_len = sum(len(txt) for txt in session_documents.values())
    if total_len <= 5000:
        return "\n\n".join(f"[{name}]\n{txt}" for name, txt in session_documents.items())

    words = set(re.findall(r"\w{3,}", query.lower()))
    chunks = []
    for name, text in session_documents.items():
        lines = text.split("\n\n")
        for chunk in lines:
            c = chunk.strip()
            if c:
                c_words = set(re.findall(r"\w{3,}", c.lower()))
                score = len(words & c_words)
                chunks.append((score, name, c))
    chunks.sort(key=lambda x: -x[0])
    selected = []
    curr = 0
    for _, name, c in chunks[:8]:
        if curr + len(c) > 5000:
            break
        selected.append(f"[{name}]\n{c}")
        curr += len(c)
    return "\n\n".join(selected)


# ----------------------------------------------------------------------------
# Helper Functions: Chat Store Persistence
# ----------------------------------------------------------------------------
def _read_chat_store() -> Dict[str, Any]:
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def _write_chat_store(store: Dict[str, Any]) -> None:
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[ChatStore Error] {exc}")


def load_all_chats() -> List[Dict[str, Any]]:
    store = _read_chat_store()
    chat_list = []
    for cid, data in store.items():
        messages = data.get("messages", [])
        title = data.get("title") or "New chat"
        updated = data.get("updated") or time.time()
        chat_list.append({
            "id": cid,
            "title": title,
            "updated": updated,
            "message_count": len(messages),
            "preview": messages[-1]["content"][:80] if messages else ""
        })
    chat_list.sort(key=lambda x: x["updated"], reverse=True)
    return chat_list


# ----------------------------------------------------------------------------
# Starlette API Routes
# ----------------------------------------------------------------------------
async def get_index(request: Request) -> HTMLResponse:
    template_path = BASE_DIR / "templates" / "index.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(
                f.read(),
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )
    return HTMLResponse("<h2>ChatBot Frontend Template Missing</h2>", status_code=404)


async def handle_legacy_streamlit(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def handle_legacy_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.close(code=1000)




async def api_status(request: Request) -> JSONResponse:
    """Returns Ollama status, available models, and Whisper readiness."""
    models: List[str] = []
    ollama_ok = False
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            ollama_ok = True
            for m in resp.json().get("models", []):
                name = m.get("name")
                if name:
                    models.append(name)
    except Exception:
        pass

    return JSONResponse({
        "status": "online" if ollama_ok else "offline",
        "ollama_connected": ollama_ok,
        "models": models or [DEFAULT_MODEL],
        "default_model": DEFAULT_MODEL,
        "whisper_model": DEFAULT_WHISPER,
        "documents_loaded": len(session_documents),
    })


async def api_list_chats(request: Request) -> JSONResponse:
    chats = load_all_chats()
    return JSONResponse({"chats": chats})


async def api_get_chat(request: Request) -> JSONResponse:
    chat_id = request.path_params["chat_id"]
    store = _read_chat_store()
    chat = store.get(chat_id)
    if not chat:
        # Create empty chat
        chat = {"title": "New chat", "updated": time.time(), "messages": []}
        store[chat_id] = chat
        _write_chat_store(store)
    return JSONResponse(chat)


async def api_save_chat(request: Request) -> JSONResponse:
    chat_id = request.path_params["chat_id"]
    body = await request.json()
    store = _read_chat_store()

    chat = store.get(chat_id, {})
    messages = body.get("messages", chat.get("messages", []))
    title = body.get("title")

    if not title:
        first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        title = " ".join(first_user.split())[:45] if first_user else chat.get("title", "New chat")

    chat["title"] = title
    chat["messages"] = messages
    chat["updated"] = time.time()
    store[chat_id] = chat
    _write_chat_store(store)

    # Also back up individual JSON in chat_history folder
    try:
        backup_file = CHAT_HISTORY_DIR / f"chat_{chat_id}.json"
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return JSONResponse({"status": "ok", "chat_id": chat_id, "title": title})


async def api_delete_chat(request: Request) -> JSONResponse:
    chat_id = request.path_params["chat_id"]
    store = _read_chat_store()
    if chat_id in store:
        del store[chat_id]
        _write_chat_store(store)

    backup_file = CHAT_HISTORY_DIR / f"chat_{chat_id}.json"
    if backup_file.exists():
        try:
            backup_file.unlink()
        except OSError:
            pass
    return JSONResponse({"status": "ok"})


async def api_documents_upload(request: Request) -> JSONResponse:
    form = await request.form()
    uploaded_file = form.get("file")
    if not uploaded_file:
        return JSONResponse({"error": "Koi file select nahi hui"}, status_code=400)

    filename = getattr(uploaded_file, "filename", "document.pdf")
    content = await uploaded_file.read()
    try:
        text = extract_document_text(filename, content)
        session_documents[filename] = text
        return JSONResponse({
            "status": "ok",
            "filename": filename,
            "char_count": len(text),
            "total_documents": len(session_documents)
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def api_documents_list(request: Request) -> JSONResponse:
    docs = [{"name": name, "chars": len(txt)} for name, txt in session_documents.items()]
    return JSONResponse({"documents": docs})


async def api_documents_delete(request: Request) -> JSONResponse:
    doc_name = request.path_params["doc_name"]
    if doc_name in session_documents:
        del session_documents[doc_name]
    return JSONResponse({"status": "ok", "remaining": len(session_documents)})


async def api_voice_note(request: Request) -> JSONResponse:
    """Accepts recorded audio blob (WebM or WAV), transcribes with Whisper."""
    form = await request.form()
    audio_file = form.get("audio")
    if not audio_file:
        return JSONResponse({"error": "Audio data missing"}, status_code=400)

    content = await audio_file.read()
    if not content or len(content) < 100:
        return JSONResponse({"error": "Audio bohot choti hai"}, status_code=400)

    try:
        transcript = await transcribe_audio_async(content)
        return JSONResponse({"status": "ok", "transcript": transcript})
    except Exception as exc:
        return JSONResponse({"error": f"Whisper error: {exc}"}, status_code=500)


async def api_tts(request: Request) -> Response:
    """Generates offline SAPI WAV audio from text."""
    body = await request.json()
    text = body.get("text", "")
    if not text.strip():
        return Response(b"", media_type="audio/wav")

    wav_bytes = await generate_speech_wav_async(text)
    return Response(wav_bytes, media_type="audio/wav")


async def api_chat_stream(request: Request) -> StreamingResponse:
    """Streams Ollama text completion token-by-token using Server-Sent Events (SSE)."""
    body = await request.json()
    user_message = body.get("message", "").strip()
    history = body.get("history", [])
    model = body.get("model", DEFAULT_MODEL)
    temperature = float(body.get("temperature", 0.3))

    context = get_relevant_document_context(user_message)
    sys_prompt = SYSTEM_PROMPT
    if context:
        sys_prompt += DOC_INSTRUCTIONS.format(context=context)

    messages_payload = [{"role": "system", "content": sys_prompt}]
    for h in history[-8:]:
        if isinstance(h, dict) and h.get("role") in ("user", "assistant"):
            messages_payload.append({"role": h["role"], "content": str(h["content"])[:1500]})
    messages_payload.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages_payload,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "num_ctx": 4096,
            "num_predict": 800,
            "num_thread": max(1, min(4, (os.cpu_count() or 2) - 1)),
        },
    }

    async def event_generator() -> AsyncGenerator[bytes, None]:
        full_text = []
        try:
            with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=(3, 120)) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        full_text.append(chunk)
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n".encode("utf-8")
                    if data.get("done"):
                        break
        except requests.exceptions.ConnectionError:
            err = "⚠️ Ollama start nahi hai! Please ensure Ollama is running on port 11434."
            yield f"data: {json.dumps({'error': err})}\n\n".encode("utf-8")
        except Exception as exc:
            err = f"⚠️ Generation error: {str(exc)}"
            yield f"data: {json.dumps({'error': err})}\n\n".encode("utf-8")

        yield f"data: {json.dumps({'done': True, 'full_text': ''.join(full_text)})}\n\n".encode("utf-8")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def api_voice_agent_turn(request: Request) -> JSONResponse:
    """
    Seamless full-duplex voice turn:
    Receives voice audio -> Whisper transcribes -> Ollama replies -> SAPI generates speech -> returns everything.
    Guaranteed to work 100% offline with zero WebRTC drops.
    """
    form = await request.form()
    audio_file = form.get("audio")
    model = form.get("model") or DEFAULT_MODEL
    history_json = form.get("history") or "[]"

    if not audio_file:
        return JSONResponse({"error": "No audio received"}, status_code=400)

    try:
        history = json.loads(str(history_json))
    except Exception:
        history = []

    content = await audio_file.read()
    if not content or len(content) < 300:
        return JSONResponse({"status": "empty", "message": "Audio too short"})

    # 1. Transcribe with Whisper
    transcript = await transcribe_audio_async(content)
    if not transcript or len(transcript.strip()) < 2:
        return JSONResponse({"status": "empty", "message": "Nothing heard"})

    # 2. Query Ollama with voice-optimized prompt
    context = get_relevant_document_context(transcript)
    sys_prompt = VOICE_SYSTEM_PROMPT
    if context:
        sys_prompt += DOC_INSTRUCTIONS.format(context=context)

    messages_payload = [{"role": "system", "content": sys_prompt}]
    for h in history[-6:]:
        if isinstance(h, dict) and h.get("role") in ("user", "assistant"):
            messages_payload.append({"role": h["role"], "content": str(h["content"])[:600]})
    messages_payload.append({"role": "user", "content": transcript})

    payload = {
        "model": model,
        "messages": messages_payload,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.25,
            "num_ctx": 2048,
            "num_predict": 120,
            "num_thread": max(1, min(4, (os.cpu_count() or 2) - 1)),
        },
    }

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=(3, 30))
        resp.raise_for_status()
        answer = resp.json().get("message", {}).get("content", "").strip()
    except Exception as exc:
        answer = "Maaf kijiye ga, Ollama se rabta nahi ho saka."

    # 3. Synthesize speech via SAPI WAV
    wav_bytes = await generate_speech_wav_async(answer)
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii") if wav_bytes else ""

    return JSONResponse({
        "status": "ok",
        "transcript": transcript,
        "answer": answer,
        "audio_base64": audio_b64
    })


# ----------------------------------------------------------------------------
# WebSocket: Realtime Voice Agent
# ----------------------------------------------------------------------------
async def websocket_voice(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        await websocket.send_json({"type": "status", "message": "Voice agent connected and ready"})
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            elif msg_type == "interrupt":
                await websocket.send_json({"type": "interrupted"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[WebSocket Error] {exc}")


# ----------------------------------------------------------------------------
# Starlette Application Setup
# ----------------------------------------------------------------------------
routes = [
    Route("/", get_index, methods=["GET"]),
    Route("/_stcore/health", handle_legacy_streamlit, methods=["GET"]),
    Route("/_stcore/host-config", handle_legacy_streamlit, methods=["GET"]),
    WebSocketRoute("/_stcore/stream", handle_legacy_ws),
    Route("/api/status", api_status, methods=["GET"]),
    Route("/api/chats", api_list_chats, methods=["GET"]),
    Route("/api/chats/{chat_id}", api_get_chat, methods=["GET"]),
    Route("/api/chats/{chat_id}", api_save_chat, methods=["POST"]),
    Route("/api/chats/{chat_id}", api_delete_chat, methods=["DELETE"]),
    Route("/api/documents/upload", api_documents_upload, methods=["POST"]),
    Route("/api/documents", api_documents_list, methods=["GET"]),
    Route("/api/documents/{doc_name}", api_documents_delete, methods=["DELETE"]),
    Route("/api/voice-note", api_voice_note, methods=["POST"]),
    Route("/api/tts", api_tts, methods=["POST"]),
    Route("/api/chat/stream", api_chat_stream, methods=["POST"]),
    Route("/api/voice-agent-turn", api_voice_agent_turn, methods=["POST"]),
    WebSocketRoute("/ws/voice", websocket_voice),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async def _open():
        await asyncio.sleep(1.0)
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{PORT}")
        except Exception:
            pass

    asyncio.create_task(_open())
    yield


app = Starlette(routes=routes, middleware=middleware, lifespan=lifespan)




def main():
    print("=" * 60)
    print("  [+] Local AI Assistant & Realtime Voice Agent")
    print(f"  [+] Running at: http://localhost:{PORT}")
    print("  [+] 100% Offline & Private (Ollama + Whisper + SAPI)")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
