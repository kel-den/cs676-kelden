"""
orchestrator.py — the agent you talk to.

You type "I want to create a persona patient" and the model decides you meant
`create_persona("patient")`. The model does this by calling a tool. We describe
the tools, run whatever it asks for, send back the result, and repeat until it
replies in plain text.

Anything the model answers without calling a tool is ordinary conversation. So
it can ask a clarifying question ("What kind of persona?") instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agent import agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .llm import LLM, ToolCall
from .persona import (
    DEFAULT_PERSONA_DIR,
    AmbiguousPersonaError,
    PersonaSpec,
    find_persona,
    generate_persona,
    list_personas,
    save_persona,
    surname_of,
)

# Guards against a model that keeps calling tools forever.
MAX_TOOL_ROUNDS = 5

# How many messages of conversation with the user the model sees. Older messages
# are dropped, so the history stays bounded.
MAX_HISTORY = 40

DEFAULT_TURNS = 6
MIN_TURNS = 2
MAX_TURNS = 20

_SYSTEM = """You are the stage manager of a persona simulation workshop.

The user builds fictional characters and has them talk to each other. Use the
tools to act on what they ask for.

- If a request is missing something the tools need (a persona with no
  description, a conversation with no topic), ask one short question. Do not guess.
- If the user names personas for a conversation, pass those names exactly.
- If they do not name any and more than two personas exist, ask which two should talk.
- Keep your replies brief: two or three sentences."""


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "create_persona",
        "description": "Create and save a new persona character from a short description.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What kind of person to create, e.g. 'a skeptical patient with back pain'.",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "list_personas",
        "description": "List every persona currently saved.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "start_conversation",
        "description": (
            "Have saved personas talk about a topic. Name the participants when the user "
            "names them; omit them only when exactly two personas exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "What the personas should discuss."},
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names or roles of the personas to use.",
                },
                "turns": {
                    "type": "integer",
                    "minimum": MIN_TURNS,
                    "maximum": MAX_TURNS,
                    "description": f"Total lines of dialogue. Default {DEFAULT_TURNS}.",
                },
            },
            "required": ["topic"],
        },
    },
]


@dataclass
class Result:
    """What the orchestrator did, in a form the UI can render."""

    reply: str
    events: List[str] = field(default_factory=list)
    conversation: Optional[Conversation] = None


class Orchestrator:
    """Routes what the user typed to the right part of the package, using tools."""

    def __init__(self, llm: LLM, persona_dir: Path = DEFAULT_PERSONA_DIR) -> None:
        """Store the model, the persona folder, and an empty conversation history."""
        self.llm = llm
        self.persona_dir = Path(persona_dir)
        self.history: List[Dict[str, Any]] = []

    # -- tools --------------------------------------------------------------

    def _create_persona(self, description: str, result: Result) -> str:
        """Generate a persona from a description, save it, and return a summary for the model."""
        description = description.strip()
        if not description:
            return "No description was given. Ask the user what kind of persona they want."

        spec = generate_persona(description, self.llm, avoid_surnames=self._taken_surnames())
        path = save_persona(spec, self.persona_dir)
        result.events.append(f"wrote {path}")
        return f"Created {spec.name} ({spec.role}) — {spec.summary}. Saved to {path}."

    def _taken_surnames(self) -> frozenset:
        """Lower-case surnames of the saved personas, so a new persona does not repeat one."""
        return frozenset(s for s in (surname_of(p.name) for p in list_personas(self.persona_dir)) if s)

    def _list_personas(self) -> str:
        """Describe every saved persona for the model, or say that none exist."""
        specs = list_personas(self.persona_dir)
        if not specs:
            return "No personas are saved yet."
        lines = [f"- {s.name} ({s.role}) — {s.summary}" for s in specs]
        return f"{len(specs)} persona(s) saved:\n" + "\n".join(lines)

    def _start_conversation(
        self,
        tool_input: Dict[str, Any],
        result: Result,
        on_turn: Optional[Callable[[Turn], None]],
        on_start: Optional[Callable[[str], None]],
    ) -> str:
        """Resolve the participants, run the conversation, save its transcript, and return the final reply."""
        topic = str(tool_input.get("topic", "")).strip()
        if not topic:
            return "No topic was given. Ask the user what the personas should talk about."

        turns = _clamp_turns(tool_input.get("turns"))
        names = tool_input.get("participants") or []
        available = list_personas(self.persona_dir)

        specs: List[PersonaSpec] = []
        if names:
            for name in names:
                try:
                    found = find_persona(str(name), self.persona_dir)
                except AmbiguousPersonaError as exc:
                    return f"{exc} Ask the user which one they mean."
                if found is None:
                    saved = ", ".join(s.name for s in available) or "none"
                    return f"No saved persona matches '{name}'. Saved personas: {saved}."
                if found not in specs:
                    specs.append(found)
        elif len(available) == 2:
            specs = available
        elif len(available) > 2:
            saved = ", ".join(s.name for s in available)
            return f"Several personas are saved ({saved}). Ask the user which two should talk."
        else:
            return "At least two personas are needed before they can talk. Create another one first."

        if len(specs) < 2:
            return "A conversation needs two different personas. Ask the user which two to use."

        memory_dir = self.persona_dir / "memory"
        agents = [agent_from_spec(s, self.llm, memory_dir=memory_dir) for s in specs]
        if on_start is not None:
            on_start(topic)
        try:
            conversation = run_conversation(agents, topic=topic, turns=turns, on_turn=on_turn)
        except Exception as exc:  # noqa: BLE001 — report the failure; never let the model invent the rest
            result.reply = f"The conversation stopped early ({type(exc).__name__}: {exc}). No transcript was saved."
            return result.reply
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = save_transcript(conversation, self.persona_dir, filename=f"conversation-{stamp}.md")

        for agent in agents:
            others = [a.name for a in agents if a is not agent]
            own_lines = [t.text for t in conversation.turns if t.speaker == agent.name]
            agent.remember(topic, others, own_lines)

        result.conversation = conversation
        result.events.append(f"wrote {path}")
        names_text = " and ".join(s.name for s in specs)
        # The reply is fixed here, not written by the model. The model never sees
        # the dialogue, so it would otherwise invent one.
        result.reply = (
            f"Done: {len(conversation.turns)} turns between {names_text} about {topic}. "
            f"Transcript saved to {path}."
        )
        return result.reply

    def _run_tool(
        self,
        call: ToolCall,
        result: Result,
        on_turn: Optional[Callable[[Turn], None]],
        on_start: Optional[Callable[[str], None]],
    ) -> Tuple[str, bool]:
        """Run one tool. Returns (output for the model, whether it failed)."""
        try:
            if call.name == "create_persona":
                return self._create_persona(str(call.input.get("description", "")), result), False
            if call.name == "list_personas":
                return self._list_personas(), False
            if call.name == "start_conversation":
                return self._start_conversation(call.input, result, on_turn, on_start), False
            return f"Unknown tool: {call.name}", True
        except Exception as exc:  # noqa: BLE001 — report the failure to the model, not the user's terminal
            return f"{type(exc).__name__}: {exc}", True

    # -- the loop -----------------------------------------------------------

    def handle(
        self,
        text: str,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """Send the user's message to the model, run any tools it asks for, and return the reply."""
        self._trim_history()
        start = len(self.history)
        self.history.append({"role": "user", "content": text})

        result = Result(reply="")
        tool_outputs: List[str] = []

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                response = self.llm.complete_with_tools(_SYSTEM, self.history, TOOLS, max_tokens=1024)
            except Exception as exc:  # noqa: BLE001 — a missing key, network error or quota problem all end the same way
                del self.history[start:]  # forget this request so the next one starts clean
                result.reply = (
                    f"I could not reach the model ({type(exc).__name__}: {exc}). "
                    "Check your API key and connection, then try again."
                )
                return result
            self.history.append({"role": "assistant", "content": response.content or response.text})

            if not response.tool_calls:
                result.reply = response.text or "\n".join(tool_outputs)
                return result

            blocks: List[Dict[str, Any]] = []
            for call in response.tool_calls:
                output, failed = self._run_tool(call, result, on_turn, on_start)
                tool_outputs.append(output)
                block: Dict[str, Any] = {"type": "tool_result", "tool_use_id": call.id, "content": output}
                if failed:
                    block["is_error"] = True
                blocks.append(block)
            self.history.append({"role": "user", "content": blocks})

            if result.reply:
                # A tool has already written the final reply (a finished or failed
                # conversation). Stop here, so the model cannot write its own version.
                self.history.append({"role": "assistant", "content": result.reply})
                return result

        result.reply = "I stopped after several steps without finishing. Try a simpler request."
        return result

    def _trim_history(self) -> None:
        """Keep the last MAX_HISTORY messages, starting on a plain user message.

        A tool_result must follow the tool_use it answers, so we never start the
        window on one.
        """
        window = self.history[-MAX_HISTORY:]
        while window and not (window[0]["role"] == "user" and isinstance(window[0]["content"], str)):
            window = window[1:]
        self.history = window


def _clamp_turns(value: Any) -> int:
    """Turn count from the model, forced into the allowed range."""
    try:
        turns = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TURNS
    return max(MIN_TURNS, min(MAX_TURNS, turns))
