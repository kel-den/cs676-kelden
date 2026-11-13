"""
PersonaForge — an agent-to-agent persona simulation package.

The idea in four lines:

    spec  = generate_persona("a patient with chronic back pain", llm)
    path  = save_persona(spec)              # writes temp/<name>.md
    agent = agent_from_spec(spec, llm)      # markdown + model = a mind
    run_conversation([doctor, patient], topic="the test results")

A persona is a markdown file. An agent is that file plus a model. A conversation
is agents taking turns. Everything else is plumbing.

Two additions in 0.2.0:

- ``Orchestrator.handle(text)`` routes a request through model tool use (create,
  list, or start a conversation) and returns a ``Result``.
- Each agent keeps private memory in ``<persona_dir>/memory/`` and recalls it in
  later conversations. Nothing one agent remembers is shown to another.

Errors name the next step. For example, ``AmbiguousPersonaError`` lists the
matching personas so the caller can ask which one is meant.
"""

from .agent import PersonaAgent, agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .llm import LLM, ClaudeLLM, ScriptedLLM, default_llm
from .orchestrator import Orchestrator, Result
from .persona import (
    AmbiguousPersonaError,
    PersonaSpec,
    find_persona,
    generate_persona,
    list_personas,
    load_persona,
    save_persona,
)

__version__ = "0.2.0"

__all__ = [
    "PersonaAgent",
    "agent_from_spec",
    "Conversation",
    "Turn",
    "run_conversation",
    "save_transcript",
    "LLM",
    "ClaudeLLM",
    "ScriptedLLM",
    "default_llm",
    "Orchestrator",
    "Result",
    "PersonaSpec",
    "AmbiguousPersonaError",
    "find_persona",
    "generate_persona",
    "list_personas",
    "load_persona",
    "save_persona",
    "__version__",
]
