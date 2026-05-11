"""
Streamlit entry point — the conversational UI for the IMDB agent.

Run with:
    streamlit run app/main.py

STARTUP SEQUENCE (first run):
  1. Build indexes if not present (build_and_save_indexes)
  2. Init SQLite DB from CSV (init_sqlite_db)
  3. Init user_preferences table (init_preferences_table)
  4. All subsequent queries go through agents/orchestrator.process_query()

SESSION MANAGEMENT:
  - Each browser tab gets its own session_id (UUID stored in st.session_state)
  - LangGraph MemorySaver uses session_id as thread_id to isolate conversation state
  - Long-term preferences are stored per session_id in SQLite

VOICE (optional):
  - Sidebar toggle enables microphone input and TTS playback
  - Requires ENABLE_VOICE=true in .env (to avoid accidental API cost)
  - Uses OpenAI Whisper for STT, OpenAI TTS for playback
"""
import sys
import os

# Ensure the project root is on sys.path regardless of where Streamlit is invoked from.
# Without this, `streamlit run app/main.py` sets cwd to app/ and Python can't find
# the agents/, config/, rag/ etc. packages at the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import asyncio
import logging
import uuid

import streamlit as st

from agents.orchestrator import process_query
from app.ui_components import (
    render_chat_history,
    render_sidebar,
    render_sources_expander,
    render_voice_input,
)
from app.voice import text_to_speech, transcribe_audio
from config.settings import settings
from memory.long_term import extract_and_save_preferences, init_preferences_table
from rag.indexer import build_and_save_indexes
from tools.sqlite_tools import init_sqlite_db

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Streamlit page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMDB Movie Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource  # Run once per Streamlit server process (not per user session)
def startup() -> None:
    """
    One-time startup: build indexes, init DB.
    st.cache_resource ensures this runs once even with multiple concurrent users.
    """
    logger.info("Running startup tasks…")
    build_and_save_indexes()          # Skip if indexes already exist
    init_sqlite_db()                   # Skip if DB already exists
    asyncio.run(init_preferences_table())
    logger.info("Startup complete.")


def get_session_id() -> str:
    """Return a stable session ID for this browser tab."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


_WELCOME = (
    "👋 Welcome! I'm your **IMDB Movie Assistant**. I can help you with:\n\n"
    "- 🎬 **Movie facts** — ratings, release years, box office, directors, cast\n"
    "- 📊 **Rankings & stats** — top movies by genre, decade, or director\n"
    "- 🔍 **Recommendations** — films similar to one you love, or matching a mood\n"
    "- 🎭 **Plot & theme search** — find movies by what happens in them\n\n"
    "What would you like to explore?"
)


def init_message_store() -> None:
    """Initialise the in-Streamlit-session message list on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": _WELCOME}]
    if "sources" not in st.session_state:
        st.session_state.sources = {}  # message_index → list[str]


def run_query(user_text: str, session_id: str) -> str:
    """Run the orchestrator synchronously from Streamlit's sync context."""
    return asyncio.run(process_query(user_text, session_id))


def save_prefs_background(user_text: str, session_id: str, llm) -> None:
    """Fire-and-forget preference extraction (non-blocking for UX)."""
    asyncio.run(extract_and_save_preferences(session_id, user_text, llm))


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    startup()
    session_id = get_session_id()
    init_message_store()

    sidebar_settings = render_sidebar(session_id)
    voice_enabled = sidebar_settings["voice_enabled"] and settings.enable_voice
    show_sources = sidebar_settings["show_sources"]

    # ── Header ─────────────────────────────────────────────────────────────────
    st.title("🎬 IMDB Movie Assistant")
    st.caption(
        "Ask me about the top 1000 IMDB movies — facts, rankings, plot themes, or comparisons."
    )

    # ── Conversation history ────────────────────────────────────────────────────
    render_chat_history(st.session_state.messages)

    # ── Input (voice or text) ───────────────────────────────────────────────────
    user_text: str = ""

    if voice_enabled:
        st.divider()
        voice_result = render_voice_input()
        if voice_result:
            audio_bytes, audio_filename = voice_result
            with st.spinner("Transcribing audio…"):
                user_text, stt_error = asyncio.run(
                    transcribe_audio(audio_bytes, filename=audio_filename)
                )
            if stt_error:
                st.error(f"Transcription failed: {stt_error}")
            elif user_text:
                st.info(f"🎤 You said: _{user_text}_")
            else:
                st.warning("Could not transcribe audio — please speak clearly and try again.")
    else:
        user_text = st.chat_input("Ask about movies…")

    # ── Process query ───────────────────────────────────────────────────────────
    if user_text:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = run_query(user_text, session_id)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        # Show retrieved sources if this was a semantic query
        if show_sources and "context" in st.session_state:
            render_sources_expander(
                st.session_state.get("last_sources", [])
            )

        # Optional: voice playback of the response
        if voice_enabled and response:
            with st.spinner("Generating audio…"):
                audio = asyncio.run(text_to_speech(response))
            if audio:
                st.audio(audio, format="audio/mp3", autoplay=True)

        # Background preference extraction (does not block the response)
        # In production this would be a proper async task / background worker
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=0,
                api_key=settings.openai_api_key,
            )
            save_prefs_background(user_text, session_id, llm)
        except Exception:
            pass  # Non-critical — never block the conversation


if __name__ == "__main__":
    main()
