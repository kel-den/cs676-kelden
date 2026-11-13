"""Tests for the tool-use router in orchestrator.py. All offline: no API key."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from personaforge.llm import DemoLLM, ToolCall, ToolResponse
from personaforge.orchestrator import MAX_HISTORY, MAX_TOOL_ROUNDS, Orchestrator
from personaforge.persona import PersonaSpec, list_personas, save_persona


class QueueLLM:
    """Returns a fixed list of ToolResponses, one per call. Records every call."""

    def __init__(self, responses: List[ToolResponse]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def complete(self, system, messages, max_tokens=2048):
        return "unused"

    def complete_with_tools(self, system, messages, tools, max_tokens=2048):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            return ToolResponse(text="done")
        return self.responses.pop(0)


def tool(name: str, **tool_input: Any) -> ToolResponse:
    call = ToolCall(id=f"id_{name}", name=name, input=tool_input)
    return ToolResponse(text="", tool_calls=[call], content=[{"type": "tool_use", "id": call.id, "name": name, "input": tool_input}])


def text(reply: str) -> ToolResponse:
    return ToolResponse(text=reply, content=[{"type": "text", "text": reply}])


@pytest.fixture
def persona_dir(tmp_path):
    return tmp_path / "temp"


def seed(persona_dir, name: str, role: str) -> PersonaSpec:
    spec = PersonaSpec(name=name, role=role, summary=f"a {role}", body=f"# {name}\n\nBackground.")
    save_persona(spec, persona_dir)
    return spec


# -- create / list ----------------------------------------------------------

def test_create_persona_writes_a_file(persona_dir):
    o = Orchestrator(DemoLLM(), persona_dir=persona_dir)
    result = o.handle("create a persona doctor who is direct and busy")
    assert len(list_personas(persona_dir)) == 1
    assert any("wrote" in e for e in result.events)


def test_list_personas_reports_saved_personas(persona_dir):
    seed(persona_dir, "Maria Delgado", "patient")
    o = Orchestrator(DemoLLM(), persona_dir=persona_dir)
    result = o.handle("list personas")
    assert "Maria Delgado" in result.reply


def test_create_with_no_description_asks_instead_of_guessing(persona_dir):
    llm = QueueLLM([tool("create_persona", description=""), text("What kind of persona should I create?")])
    o = Orchestrator(llm, persona_dir=persona_dir)
    result = o.handle("create a persona")
    assert list_personas(persona_dir) == []
    assert "What kind" in result.reply


# -- conversations ----------------------------------------------------------

def test_talk_without_topic_asks_a_question(persona_dir):
    seed(persona_dir, "Maria", "patient")
    seed(persona_dir, "Dr Reyes", "doctor")
    llm = QueueLLM([tool("start_conversation", topic=""), text("What should they talk about?")])
    result = Orchestrator(llm, persona_dir=persona_dir).handle("have them talk")
    assert result.conversation is None
    assert "talk about" in result.reply


def test_talk_with_two_personas_runs_a_conversation(persona_dir):
    seed(persona_dir, "Maria", "patient")
    seed(persona_dir, "Dr Reyes", "doctor")
    result = Orchestrator(DemoLLM(), persona_dir=persona_dir).handle("have them talk about the test results")
    assert result.conversation is not None
    assert len(result.conversation.turns) == 6
    assert list(persona_dir.glob("conversation-*.md"))


def test_three_personas_and_no_names_asks_which_two(persona_dir):
    seed(persona_dir, "Maria", "patient")
    seed(persona_dir, "Dr Reyes", "doctor")
    seed(persona_dir, "Nurse Kim", "nurse")
    llm = QueueLLM([tool("start_conversation", topic="the results"), text("Which two should talk?")])
    result = Orchestrator(llm, persona_dir=persona_dir).handle("have them talk about the results")
    assert result.conversation is None
    # The tool result sent back to the model should explain why nothing ran.
    tool_result = llm.calls[1]["messages"][-1]["content"][0]["content"]
    assert "Ask the user which two" in tool_result


def test_unknown_participant_is_reported_back(persona_dir):
    seed(persona_dir, "Maria", "patient")
    seed(persona_dir, "Dr Reyes", "doctor")
    llm = QueueLLM([tool("start_conversation", topic="x", participants=["Ghost", "Maria"]), text("No one called Ghost.")])
    o = Orchestrator(llm, persona_dir=persona_dir)
    o.handle("have Ghost talk to Maria about x")
    tool_result = llm.calls[1]["messages"][-1]["content"][0]
    assert tool_result.get("is_error") is None  # reported as text, not as a crash
    assert "No saved persona matches 'Ghost'" in tool_result["content"]


def test_turns_are_clamped(persona_dir):
    seed(persona_dir, "Maria", "patient")
    seed(persona_dir, "Dr Reyes", "doctor")
    too_many = QueueLLM([tool("start_conversation", topic="x", turns=500), text("ok")])
    o = Orchestrator(too_many, persona_dir=persona_dir)
    res = o.handle("talk about x")
    assert len(res.conversation.turns) <= 20


# -- loop safety ------------------------------------------------------------

def test_unknown_tool_is_reported_as_an_error(persona_dir):
    llm = QueueLLM([tool("delete_everything"), text("Sorry, I can't do that.")])
    o = Orchestrator(llm, persona_dir=persona_dir)
    o.handle("wipe the folder")
    block = llm.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "Unknown tool" in block["content"]


def test_endless_tool_requests_are_stopped(persona_dir):
    llm = QueueLLM([tool("list_personas") for _ in range(MAX_TOOL_ROUNDS + 3)])
    result = Orchestrator(llm, persona_dir=persona_dir).handle("list")
    assert "stopped" in result.reply
    assert len(llm.calls) == MAX_TOOL_ROUNDS


# -- history ----------------------------------------------------------------

def test_history_stays_bounded(persona_dir):
    o = Orchestrator(DemoLLM(), persona_dir=persona_dir)
    for i in range(60):
        o.handle(f"hello {i}")
    assert len(o.history) <= MAX_HISTORY + 2


def test_history_never_starts_on_a_tool_result(persona_dir):
    o = Orchestrator(DemoLLM(), persona_dir=persona_dir)
    for i in range(30):
        o.handle(f"create a persona nurse number {i}")
    first = o.history[0]
    assert first["role"] == "user" and isinstance(first["content"], str)


def test_no_model_call_after_a_conversation(persona_dir):
    """After a conversation runs, the reply is fixed in code and the model is not asked again."""
    seed(persona_dir, "Maria", "patient")
    seed(persona_dir, "Dr Reyes", "doctor")
    llm = QueueLLM([tool("start_conversation", topic="the shift", turns=2), text("should never be used")])
    o = Orchestrator(llm, persona_dir=persona_dir)
    result = o.handle("have Maria and Dr Reyes talk about the shift")
    assert len(llm.calls) == 1  # only the call that asked for the conversation
    assert "should never be used" not in result.reply
    assert result.reply.startswith("Done: 2 turns between")
    assert "Transcript saved to" in result.reply
    assert result.conversation is not None


class FailingLLM(QueueLLM):
    """Every model call raises, as it would with a missing key or no network."""

    def complete(self, system, messages, max_tokens=2048):
        raise RuntimeError("network down")

    def complete_with_tools(self, system, messages, tools, max_tokens=2048):
        raise RuntimeError("network down")


def test_model_failure_gives_a_clear_reply_and_keeps_history_clean(persona_dir):
    o = Orchestrator(FailingLLM([]), persona_dir=persona_dir)
    result = o.handle("list personas")
    assert "could not reach the model" in result.reply
    assert "API key" in result.reply
    assert o.history == []  # the failed request is forgotten, so the next one starts clean


def test_failed_conversation_is_reported_and_not_invented(persona_dir):
    seed(persona_dir, "Maria", "patient")
    seed(persona_dir, "Dr Reyes", "doctor")

    class DialogueFails(QueueLLM):
        def complete(self, system, messages, max_tokens=2048):
            raise RuntimeError("reply failed")

    llm = DialogueFails([tool("start_conversation", topic="the results", turns=2), text("invented")])
    o = Orchestrator(llm, persona_dir=persona_dir)
    result = o.handle("have Maria and Dr Reyes talk about the results")
    assert result.reply.startswith("The conversation stopped early")
    assert "No transcript was saved" in result.reply
    assert "invented" not in result.reply
    assert len(llm.calls) == 1
    assert list(persona_dir.glob("conversation-*.md")) == []
