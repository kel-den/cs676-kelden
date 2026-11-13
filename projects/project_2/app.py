"""Streamlit front end for PersonaForge. It holds no logic of its own: each message goes to the orchestrator."""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from personaforge.llm import default_llm  # noqa: E402
from personaforge.orchestrator import Orchestrator  # noqa: E402

PERSONA_DIR = Path("web_personas")  # cloud storage is reset when the app restarts


def load_key_from_secrets() -> None:
    """Copy the key from Streamlit's Secrets box into the environment variable the package reads."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass  # no key configured: the app runs in offline demo mode


load_key_from_secrets()
offline = not os.getenv("ANTHROPIC_API_KEY")

st.set_page_config(page_title="PersonaForge", page_icon="🎭")
st.title("PersonaForge")
st.caption("Create personas, then have them talk. Try: create a persona doctor who is direct and busy")
if offline:
    st.info("Offline demo mode: no API key is set, so replies come from a simple built-in model.")

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator(llm=default_llm(offline=offline), persona_dir=PERSONA_DIR)
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if text := st.chat_input("Type a request"):
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            result = st.session_state.orchestrator.handle(text)
        st.markdown(result.reply)
    st.session_state.messages.append({"role": "assistant", "content": result.reply})
