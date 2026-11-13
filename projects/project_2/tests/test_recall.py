"""Tests that an agent recalls its own memory, and only its own. All offline."""

from __future__ import annotations

from personaforge.agent import agent_from_spec
from personaforge.conversation import Turn
from personaforge.llm import ScriptedLLM
from personaforge.persona import PersonaSpec

# A fact that appears only in memory, never in the topic or the transcript.
PLANTED_FACT = "Mrs. Patterson's pressure dropped sharply overnight."


def make_agent(name, memory_dir, llm):
    spec = PersonaSpec(name=name, role="staff", summary="a test persona", body=f"# {name}\n\nBackground.")
    return agent_from_spec(spec, llm, memory_dir=memory_dir)


def test_agent_recalls_a_fact_that_is_only_in_memory(tmp_path):
    llm = ScriptedLLM(replies=["Understood."])
    memory_dir = tmp_path / "memory"
    margaret = make_agent("Margaret Chen", memory_dir, llm)

    # The topic says nothing about Mrs. Patterson, so the fact can only come from memory.
    margaret.remember("the shift", ["Dr. Marcus Chen"], [PLANTED_FACT])
    margaret.respond([Turn(speaker="Scene", text="the shift")])

    system_prompt = llm.calls[-1]["system"]
    assert PLANTED_FACT in system_prompt
    assert "WHAT YOU REMEMBER" in system_prompt


def test_memory_is_not_shared_with_another_agent(tmp_path):
    llm = ScriptedLLM(replies=["Understood."])
    memory_dir = tmp_path / "memory"
    margaret = make_agent("Margaret Chen", memory_dir, llm)
    marcus = make_agent("Dr. Marcus Chen", memory_dir, llm)

    margaret.remember("the shift", ["Dr. Marcus Chen"], [PLANTED_FACT])
    marcus.respond([Turn(speaker="Scene", text="the shift")])

    assert PLANTED_FACT not in llm.calls[-1]["system"]
