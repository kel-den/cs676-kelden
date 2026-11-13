"""Tests for agent.py, memory, and the list_personas fix. All offline."""

from __future__ import annotations


import pytest

from personaforge.agent import MAX_CONTEXT_TURNS, PASS, PersonaAgent, check_in_character
from personaforge.conversation import Turn, run_conversation
from personaforge.llm import ScriptedLLM
from personaforge.persona import PersonaSpec, list_personas, save_persona


@pytest.fixture
def maria() -> PersonaSpec:
    return PersonaSpec(name="Maria", role="patient", summary="pain", body="Body of Maria.")


@pytest.fixture
def reyes() -> PersonaSpec:
    return PersonaSpec(name="Dr Reyes", role="doctor", summary="busy", body="Body of the doctor.")


# -- character checks -------------------------------------------------------

@pytest.mark.parametrize(
    "reply, expected",
    [
        ("*sighs* fine.", "stage direction in asterisks"),
        ("Well [laughs] okay.", "bracketed stage direction"),
        ("As an AI, I can't feel pain.", "refers to being an AI or a simulation"),
        ("Dr Reyes: let me look.", "speaks as someone else"),
        ("x" * 1500, "reply is far too long"),
    ],
)
def test_character_checks_flag_breaks(reply, expected):
    assert expected in check_in_character(reply, "Maria")


def test_clean_reply_has_no_issues():
    assert check_in_character("Coffee makes it worse, and lying down helps.", "Maria") == []


# -- passing ---------------------------------------------------------------

def test_pass_returns_none(maria):
    agent = PersonaAgent(maria, ScriptedLLM(replies=[PASS]))
    assert agent.respond([Turn("Scene", "the results")]) is None


def test_empty_reply_is_treated_as_pass(maria):
    agent = PersonaAgent(maria, ScriptedLLM(replies=["   "]))
    assert agent.respond([Turn("Scene", "the results")]) is None


def test_run_conversation_skips_passed_turns(maria, reyes):
    replies = ["Ok.", PASS, "Still here.", PASS, "Fine.", PASS]
    agents = [PersonaAgent(maria, ScriptedLLM(replies=replies)), PersonaAgent(reyes, ScriptedLLM(replies=["unused"]))]
    convo = run_conversation(agents, topic="x", turns=6)
    assert all(t.text != PASS for t in convo.turns)
    assert len(convo.turns) < 6


# -- context window ---------------------------------------------------------

def test_long_transcript_is_trimmed_and_keeps_the_scene(maria):
    llm = ScriptedLLM(replies=["ok"])
    transcript = [Turn("Scene", "the topic")] + [Turn("Dr Reyes", f"line {i}") for i in range(40)]
    PersonaAgent(maria, llm).respond(transcript)
    messages = llm.calls[-1]["messages"]
    assert len(messages) == MAX_CONTEXT_TURNS
    assert messages[0]["content"].startswith("(earlier discussion omitted)")
    assert "the topic" in messages[0]["content"]


def test_short_transcript_is_not_trimmed(maria):
    llm = ScriptedLLM(replies=["ok"])
    PersonaAgent(maria, llm).respond([Turn("Scene", "topic"), Turn("Dr Reyes", "hi")])
    assert len(llm.calls[-1]["messages"]) == 2
    assert "omitted" not in llm.calls[-1]["messages"][0]["content"]


# -- memory -----------------------------------------------------------------

def test_memory_is_off_by_default(maria):
    agent = PersonaAgent(maria, ScriptedLLM())
    agent.remember("topic", ["Dr Reyes"], ["hello"])  # must not raise or write anything
    assert agent.recall() == ""
    assert agent.memory_path() is None


def test_remember_then_recall(maria, tmp_path):
    agent = PersonaAgent(maria, ScriptedLLM(), memory_dir=tmp_path / "memory")
    agent.remember("the MRI", ["Dr Reyes"], ["I have been waiting three weeks."])
    recalled = agent.recall()
    assert "the MRI" in recalled
    assert "I have been waiting three weeks." in recalled


def test_memory_goes_into_own_prompt_only(maria, reyes, tmp_path):
    mem = tmp_path / "memory"
    PersonaAgent(maria, ScriptedLLM(), memory_dir=mem).remember("topic", ["Dr Reyes"], ["Maria secret line"])
    PersonaAgent(reyes, ScriptedLLM(), memory_dir=mem).remember("topic", ["Maria"], ["Doctor line"])

    maria_prompt = PersonaAgent(maria, ScriptedLLM(), memory_dir=mem).system_prompt()
    doctor_prompt = PersonaAgent(reyes, ScriptedLLM(), memory_dir=mem).system_prompt()

    assert "Maria secret line" in maria_prompt
    assert "Doctor line" not in maria_prompt
    assert "Maria secret line" not in doctor_prompt


def test_memory_is_capped(maria, tmp_path):
    agent = PersonaAgent(maria, ScriptedLLM(), memory_dir=tmp_path / "memory")
    for i in range(200):
        agent.remember(f"topic {i}", ["Dr"], ["a line " * 20])
    assert len(agent.recall()) <= 3000


# -- persona listing --------------------------------------------------------

def test_list_personas_skips_transcripts(tmp_path, maria):
    save_persona(maria, tmp_path)
    (tmp_path / "conversation.md").write_text("# Conversation: x\n\n**Maria:** hi", encoding="utf-8")
    names = [p.name for p in list_personas(tmp_path)]
    assert names == ["Maria"]
