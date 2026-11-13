"""
conversation.py — put two or more agents in a room and let them talk.

Agents take strict turns in the order given, for a fixed number of turns. An
agent may pass its turn with [PASS]. Otherwise nothing decides who speaks next,
and no agent can interrupt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from .agent import PersonaAgent


@dataclass
class Turn:
    """One line of dialogue."""

    speaker: str
    text: str


@dataclass
class Conversation:
    """A completed dialogue, plus the setup that produced it."""

    topic: str
    participants: List[str]
    turns: List[Turn] = field(default_factory=list)

    def transcript(self) -> str:
        """Render as plain text, suitable for saving or printing."""
        header = f"# Conversation: {self.topic}\n\nParticipants: {', '.join(self.participants)}\n\n"
        return header + "\n\n".join(f"**{t.speaker}:** {t.text}" for t in self.turns)


def run_conversation(
    agents: List[PersonaAgent],
    topic: str,
    turns: int = 6,
    on_turn: Optional[Callable[[Turn], None]] = None,
) -> Conversation:
    """
    Run a round-robin conversation between the given agents.

    :param agents:  two or more PersonaAgents. They speak in list order.
    :param topic:   seeds the opening; each agent sees it as the situation.
    :param turns:   total lines of dialogue, split across the agents.
    :param on_turn: called after each line, so a UI can print as it goes rather
                    than waiting for the whole conversation to finish.
    """
    if len(agents) < 2:
        raise ValueError("a conversation needs at least two agents")

    conversation = Conversation(topic=topic, participants=[a.name for a in agents])

    # The topic is injected as a scene-setting turn from a narrator. Every agent
    # sees it as context but nobody has to answer it directly.
    history: List[Turn] = [Turn(speaker="Scene", text=topic)]

    for i in range(turns):
        speaker = agents[i % len(agents)]
        text = speaker.respond(history)
        if text is None:  # the agent passed: it has nothing to add this turn
            continue
        turn = Turn(speaker=speaker.name, text=text)

        history.append(turn)
        conversation.turns.append(turn)

        if on_turn is not None:
            on_turn(turn)

    return conversation


def save_transcript(conversation: Conversation, directory: Path, filename: Optional[str] = None) -> Path:
    """Write the transcript to disk and return its path.

    Without a filename, the name includes a timestamp so earlier transcripts are kept.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = filename or f"conversation-{datetime.now():%Y%m%d-%H%M%S-%f}.md"
    path = directory / name
    path.write_text(conversation.transcript(), encoding="utf-8")
    return path


# =============================================================================
# KNOWN LIMITATIONS
# =============================================================================
#  - Turn order is strict round-robin. An agent can pass with [PASS], but it
#    cannot interrupt, and the conversation does not end early once the question
#    is answered. These are the open parts of Task 3.
#  - Nobody observes the conversation. This is Task 4.
#  - The "Scene" turn is a fake speaker that carries the topic. A cleaner way to
#    set context would remove it.
#  - If an agent's reply raises, the conversation stops and the lines already
#    said are not saved as a transcript. The orchestrator reports this to the user.
# =============================================================================
