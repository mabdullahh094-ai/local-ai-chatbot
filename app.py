"""
Local AI Assistant & Realtime Voice Agent Launcher
Runs the offline server directly.
"""
import sys
from pathlib import Path

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from server import main

if __name__ in ("__main__", "__page__") or "streamlit" in sys.modules:
    main()
