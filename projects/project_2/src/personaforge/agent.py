"""
agent.py — markdown file + model = an agent with its own mind.

A `PersonaAgent` holds a `PersonaSpec` (which came from a `.md` file), an `LLM`,
and optionally a folder where it keeps its own memory. When you ask it to speak,
it builds a system prompt from the persona markdown and the agent's private
memory, then sends the most recent part of the transcript.

Memory design (answers "where should memory live?"):
  - The persona `.md` is the character you wrote. It stays as you wrote it.
  - Memory is what the character has lived through. It lives in a sidecar file,
    `<persona_dir>/memory/<slug>.md`, which `list_personas()` never reads
    (it only looks at the top level of the persona folder).

Privacy (answers "what does an agent know privately?"):
  - An agent's memory is loaded only into that agent's own system prompt.
  - The transcript is the only thing agents share, and every agent sees it
    as public dialogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import re
from typing import TYPE_CHECKING, List, Optional

from .llm import LLM
from .persona import PersonaSpec

if TYPE_CHECKING:  # imported only for type hints, to avoid a circular import
    from .conversation import Turn

# Returned by the model when it chooses to stay silent this turn.
PASS = "[PASS]"

# Only the most recent turns go to the model, so long conversations stay cheap.
# The scene turn (the topic) is always kept.
MAX_CONTEXT_TURNS = 12

# How much of an agent's memory file is injected into its prompt, and how many of
# its own lines are saved per conversation.
MEMORY_CHARS = 3000
MAX_REMEMBERED_LINES = 6
REMEMBERED_LINE_CHARS = 200

# Wrapped around the persona markdown to put the model in character and keep it
# from narrating. Without the stage-direction rules models tend to write stage
# directions and answer for everyone in the room.
_IN_CHARACTER_TEMPLATE = """You are role-playing a character in a simulated conversation.

Everything between the markers below defines who you are. Inhabit it completely:
speak in this person's voice, hold their opinions, want what they want.

=== BEGIN CHARACTER ===
{persona}
=== END CHARACTER ===

{memory}Rules:
- Speak only as {name}. Never speak for anyone else.
- Reply with dialogue only. No stage directions, no asterisks, no narration.
- Stay in character even if the conversation goes somewhere unexpected.
- Keep replies to a few sentences unless the moment genuinely calls for more.
- If you have nothing useful to add right now, reply with exactly [PASS] and nothing else.
- You are {name}. Do not mention that you are an AI or a simulation."""

_MEMORY_TEMPLATE = """=== WHAT YOU REMEMBER (private: only {name} knows this) ===
{memory}
=== END MEMORY ===

"""

_AI_SLIP = re.compile(r"\b(as an ai|language model|large language model|chatbot|simulation)\b", re.IGNORECASE)
_STAGE_ASTERISK = re.compile(r"\*[^*\n]+\*")
_STAGE_BRACKET = re.compile(r"\[[^\]\n]+\]")
_OTHER_SPEAKER = re.compile(r"^[A-Z][\w .'-]{1,40}:\s")
_MAX_REPLY_CHARS = 1200

# First-person physical actions written as plain text, e.g. "I pause, choosing my words."
# Only common stage-direction verbs are listed, so ordinary dialogue such as
# "I look forward to it" is not flagged.
_NARRATED_ACTION = re.compile(
    r"(?:^|[.!?]\s+)I (?:nod|pause|sigh|shrug|smile|frown|lean|sip|wipe|rub|glance|look up"
    r"|take a (?:drink|breath|moment|sip)|cross my arms|close my eyes|shake my head"
    r"|set (?:it |the \w+ )?down|put (?:it |the \w+ )?down|turn to|step (?:back|closer|away|forward)"
    r"|stand up|sit down|sit back|pick up)\b",
    re.MULTILINE,
)


def check_in_character(text: str, name: str) -> List[str]:
    """
    Return a list of ways a reply broke character. Empty means it looks fine.

    This is a set of heuristics, not a judge. It catches the common slips:
    stage directions, mentioning being an AI, speaking as someone else, and
    replies that run far too long.
    """
    issues: List[str] = []
    if _STAGE_ASTERISK.search(text):
        issues.append("stage direction in asterisks")
    if _STAGE_BRACKET.search(text):
        issues.append("bracketed stage direction")
    if _NARRATED_ACTION.search(text):
        issues.append("narrates its own physical action in plain text")
    if _AI_SLIP.search(text):
        issues.append("refers to being an AI or a simulation")
    for line in text.splitlines():
        if _OTHER_SPEAKER.match(line.strip()) and not line.strip().startswith(name):
            issues.append("speaks as someone else")
            break
    if len(text) > _MAX_REPLY_CHARS:
        issues.append("reply is far too long")
    return issues


def _shorten(text: str, limit: int = REMEMBERED_LINE_CHARS) -> str:
    """Collapse whitespace and cut text to at most `limit` characters, ending with an ellipsis if cut."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class PersonaAgent:
    """One character in a conversation."""

    spec: PersonaSpec
    llm: LLM
    memory_dir: Optional[Path] = None
    last_issues: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """The character's name, taken from the persona file."""
        return self.spec.name

    # -- memory -------------------------------------------------------------

    def memory_path(self) -> Optional[Path]:
        """Where this agent's private memory lives, or None if memory is off."""
        if self.memory_dir is None:
            return None
        return Path(self.memory_dir) / f"{self.spec.slug()}.md"

    def recall(self, max_chars: int = MEMORY_CHARS) -> str:
        """The most recent part of this agent's memory. Empty if there is none."""
        path = self.memory_path()
        if path is None or not path.exists():
            return ""
        text = path.read_text(encoding="utf-8").strip()
        if len(text) > max_chars:
            text = text[-max_chars:]
            cut = text.find("\n## ")  # start on an entry boundary, not mid-line
            if cut != -1:
                text = text[cut + 1 :]
        return text

    def remember(self, topic: str, others: List[str], own_lines: List[str]) -> None:
        """Record what this agent said in a conversation. Nothing else is stored."""
        path = self.memory_path()
        if path is None or not own_lines:
            return
        entry = [
            f"## {date.today().isoformat()} — {topic}",
            f"Talked with: {', '.join(others) or 'no one'}",
        ]
        entry += [f"- I said: {_shorten(line)}" for line in own_lines[:MAX_REMEMBERED_LINES]]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(entry) + "\n")

    # -- speaking -----------------------------------------------------------

    def system_prompt(self) -> str:
        """The persona markdown and private memory, wrapped in in-character instructions."""
        memory = self.recall()
        memory_block = _MEMORY_TEMPLATE.format(name=self.name, memory=memory) if memory else ""
        return _IN_CHARACTER_TEMPLATE.format(persona=self.spec.body, name=self.name, memory=memory_block)

    def respond(self, transcript: List["Turn"], max_tokens: int = 1024) -> Optional[str]:
        """
        Produce this character's next line, or None if it passes this turn.

        Only the scene and the most recent turns are sent. Each turn is rewritten
        from this agent's point of view: its own past lines become `assistant`
        turns and everyone else's become `user` turns.
        """
        window = transcript
        omitted = len(transcript) > MAX_CONTEXT_TURNS
        if omitted:
            window = transcript[:1] + transcript[-(MAX_CONTEXT_TURNS - 1):]

        messages: List[dict] = []
        for turn in window:
            if turn.speaker == self.name:
                messages.append({"role": "assistant", "content": turn.text})
            else:
                messages.append({"role": "user", "content": f"{turn.speaker}: {turn.text}"})

        # The API requires the first message to be from the user. An empty or
        # assistant-first transcript needs an opener.
        if not messages or messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "(begin the conversation)"})
        if omitted:
            messages[0]["content"] = "(earlier discussion omitted)\n" + messages[0]["content"]

        text = self.llm.complete(self.system_prompt(), messages, max_tokens=max_tokens).strip()
        self.last_issues = check_in_character(text, self.name)

        if not text or text.startswith(PASS):
            return None
        return text


def agent_from_spec(spec: PersonaSpec, llm: LLM, memory_dir: Optional[Path] = None) -> PersonaAgent:
    """Convenience constructor, so callers don't import the dataclass directly."""
    return PersonaAgent(spec=spec, llm=llm, memory_dir=memory_dir)
