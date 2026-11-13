"""Step 2 tests: no overwriting, stricter persona matching, unique transcript names."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from personaforge.llm import DemoLLM, ToolCall, ToolResponse
from personaforge.orchestrator import Orchestrator
from personaforge.persona import (
    AmbiguousPersonaError,
    PersonaSpec,
    find_persona,
    list_personas,
    save_persona,
)


def spec(name: str, role: str, body: str = "b") -> PersonaSpec:
    return PersonaSpec(name=name, role=role, summary=f"a {role}", body=body)


# -- no overwriting ---------------------------------------------------------

def test_saving_the_same_name_twice_keeps_both(tmp_path):
    first = save_persona(spec("Maria Delgado", "patient", body="first"), tmp_path)
    second = save_persona(spec("Maria Delgado", "patient", body="second"), tmp_path)

    assert first.name == "maria-delgado.md"
    assert second.name == "maria-delgado-2.md"
    assert "first" in first.read_text(encoding="utf-8")
    assert "second" in second.read_text(encoding="utf-8")


def test_third_save_gets_a_third_name(tmp_path):
    for _ in range(3):
        last = save_persona(spec("Maria", "patient"), tmp_path)
    assert last.name == "maria-3.md"
    assert len(list_personas(tmp_path)) == 3


# -- stricter matching ------------------------------------------------------

@pytest.fixture
def cast(tmp_path):
    save_persona(spec("Dorothy Vance", "nurse"), tmp_path)
    save_persona(spec("Dr Ada Doyle", "doctor"), tmp_path)
    save_persona(spec("Maria Delgado", "patient"), tmp_path)
    save_persona(spec("Maria Ortiz", "patient"), tmp_path)
    return tmp_path


def test_exact_name_wins(cast):
    assert find_persona("Dorothy Vance", cast).role == "nurse"


def test_exact_role_with_one_match_is_found(cast):
    assert find_persona("nurse", cast).name == "Dorothy Vance"


def test_substring_does_not_match_inside_another_word(cast):
    # "doc" is in "doctor" but not in "Dorothy": only the doctor should match.
    assert find_persona("doc", cast).name == "Dr Ada Doyle"


def test_ambiguous_first_name_raises_with_the_options(cast):
    with pytest.raises(AmbiguousPersonaError) as info:
        find_persona("maria", cast)
    names = [m.name for m in info.value.matches]
    assert names == ["Maria Delgado", "Maria Ortiz"]
    assert "Maria Delgado" in str(info.value)


def test_ambiguous_role_raises(cast):
    with pytest.raises(AmbiguousPersonaError):
        find_persona("patient", cast)


def test_unknown_name_returns_none(cast):
    assert find_persona("Zaphod", cast) is None


def test_empty_query_returns_none(cast):
    assert find_persona("   ", cast) is None


# -- orchestrator uses the new matching ------------------------------------

class TurnLLM:
    """Asks for start_conversation with the given participants, then finishes."""

    def __init__(self, participants: List[str]) -> None:
        self.participants = participants
        self.calls = 0

    def complete(self, system, messages, max_tokens=2048):
        return "line"

    def complete_with_tools(self, system, messages, tools, max_tokens=2048) -> ToolResponse:
        self.calls += 1
        if self.calls == 1:
            call = ToolCall(id="t1", name="start_conversation", input={"topic": "x", "participants": self.participants})
            return ToolResponse(text="", tool_calls=[call], content=[{"type": "tool_use", "id": "t1", "name": "start_conversation", "input": {}}])
        return ToolResponse(text="done", content=[{"type": "text", "text": "done"}])


def test_orchestrator_asks_when_a_participant_is_ambiguous(cast):
    output = _tool_output(cast, ["maria", "Dorothy"])
    assert "Ask the user which one they mean" in output
    assert "Maria Delgado" in output and "Maria Ortiz" in output


def _tool_output(persona_dir, participants) -> str:
    """Run the orchestrator and return the text the model received for the tool call."""
    seen: Dict[str, Any] = {}

    class Spy(TurnLLM):
        def complete_with_tools(self, system, messages, tools, max_tokens=2048):
            response = super().complete_with_tools(system, messages, tools, max_tokens)
            if self.calls == 2:
                seen["output"] = messages[-1]["content"][0]["content"]
            return response

    Orchestrator(Spy(participants), persona_dir=persona_dir).handle("talk")
    return seen["output"]


# -- unique transcript names ------------------------------------------------

def test_each_conversation_gets_its_own_transcript(tmp_path):
    save_persona(spec("Alice Adams", "doctor"), tmp_path)
    save_persona(spec("Bob Brown", "patient"), tmp_path)
    orch = Orchestrator(DemoLLM(), persona_dir=tmp_path)

    orch.handle("have them talk about the results")
    orch.handle("have them talk about the results again")

    transcripts = sorted(tmp_path.glob("conversation-*.md"))
    assert len(transcripts) == 2
    assert transcripts[0].name != transcripts[1].name
