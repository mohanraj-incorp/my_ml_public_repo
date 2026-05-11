"""
Reusable Streamlit UI components.

Keeping these in a separate module means app/main.py stays readable and
each component is independently testable (render logic ≠ app logic).

SCALE NOTE: For a production Streamlit app, these components would live in
a component library shared across multiple pages. Streamlit custom components
(via streamlit-component-lib) allow packaging as proper React components
with better state management and styling control.
"""
import streamlit as st


def render_chat_history(messages: list[dict]) -> None:
    """
    Render the full conversation history using st.chat_message.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str} dicts
    """
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_sources_expander(sources: list[str]) -> None:
    """
    Show retrieved source passages in a collapsed expander.
    Only rendered when sources are non-empty (semantic search queries).

    Args:
        sources: List of movie plot passages used to generate the answer
    """
    if not sources:
        return
    with st.expander("🔍 Retrieved Sources", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**Source {i}:**")
            st.text(source[:500] + ("…" if len(source) > 500 else ""))
            st.divider()


def render_sidebar(session_id: str) -> dict:
    """
    Render the sidebar with session info, voice toggle, and settings.

    Returns:
        Dict of sidebar settings: {"voice_enabled": bool, "show_sources": bool}
    """
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown(f"**Session:** `{session_id[:8]}…`")
        st.divider()

        voice_enabled = st.toggle(
            "🎤 Voice Mode",
            value=False,
            help="Enable microphone input and audio responses (requires ENABLE_VOICE=true in .env)",
        )

        show_sources = st.toggle(
            "📄 Show Sources",
            value=True,
            help="Show retrieved movie passages used to generate semantic search answers",
        )

        st.divider()
        st.markdown("**About**")
        st.markdown(
            "Conversational agent over the IMDB top-1000 dataset. "
            "Powered by LangGraph + hybrid RAG."
        )

        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

    return {"voice_enabled": voice_enabled, "show_sources": show_sources}


def render_voice_input() -> tuple[bytes, str] | None:
    """
    Show Streamlit's audio recorder. Returns (audio_bytes, filename) if recording
    submitted, so the caller can pass the correct filename to Whisper for format detection.
    Only called when voice mode is enabled.
    """
    audio = st.audio_input("🎤 Record your question")
    if audio is not None:
        data = audio.read()
        if not data:
            st.warning("Recording was empty — please try again.")
            return None
        return data, getattr(audio, "name", "audio.webm")
    return None
