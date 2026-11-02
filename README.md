# Local AI Chatbot & Realtime Voice Agent (100% Offline)

A modern, responsive, and completely private AI Assistant featuring real-time bidirectional voice conversation, voice note messaging, document Q&A, and full chat history persistence.

---

## 🌟 Key Features

1. **💬 Fast Text Chat**
   - Natural conversations in Roman Urdu, Urdu, and English.
   - Real-time token streaming powered by local Ollama (`qwen2.5:3b` default, `qwen2.5:7b` optional).
   - Clean markdown formatting (code blocks with copy buttons, lists, bold).

2. **🎙️ Realtime Voice Agent**
   - Immersive full-screen voice agent mode.
   - Interactive 3D glowing orb that responds to conversational states:
     - **Listening** (cyan/emerald pulse reacting to your voice)
     - **Thinking** (radiant spinning glow)
     - **Speaking** (vibrant soundwave ripple as the assistant talks)
   - Live subtitles for both your voice and the assistant's speech.
   - Voice Activity Detection (VAD) & Barge-in support (interrupt the assistant at any time).
   - 100% offline voice synthesis via Windows SAPI.

3. **🎤 Voice Note Messaging**
   - One-click microphone button in the message composer.
   - Visual recording bar with live timer, waveform animation, Cancel, and Send buttons.
   - Transcribed instantly using offline `faster-whisper` (`tiny` model).

4. **🗂️ Chat History & Sessions**
   - **+ New Chat** button creates fresh conversations instantly.
   - Sidebar list of past chats with search bar and 1-click deletion.
   - Auto-saved locally in `chat_history.json` and `chat_history/`.

5. **📎 Document Q&A (Knowledge Base)**
   - Upload `.pdf` or `.docx` documents.
   - The assistant extracts relevant excerpts to answer questions based on your files.

6. **🔒 100% Offline & Private**
   - No external APIs, cloud services, or CDN dependencies.
   - Ollama runs locally on `http://localhost:11434`.
   - Whisper runs locally using models cached in `D:\AI\huggingface`.
   - Windows speech engine runs locally via SAPI.

---

## 🚀 Quick Start

Simply double-click `start_chatbot.bat` or run in PowerShell:

```powershell
.\.venv\Scripts\python.exe server.py
```

The app will start at `http://localhost:8501` and automatically open in your default browser.

---

## 📁 Project Structure

- `server.py`: Starlette & Uvicorn asynchronous backend (Ollama, Whisper, SAPI TTS, REST & WebSocket endpoints).
- `app.py`: Entrypoint wrapper that launches `server.py`.
- `templates/index.html`: Modern, self-contained single-page UI with embedded SVG icons and animations.
- `start_chatbot.bat`: One-click Windows launcher.
- `chat_history.json` & `chat_history/`: Local conversation storage.
- `knowledge/`: Local documents folder.
- `requirements.txt`: Python package requirements.
- `app_streamlit_backup.py`: Safe backup of previous Streamlit implementation.
