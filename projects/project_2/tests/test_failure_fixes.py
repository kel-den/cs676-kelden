"""Tests for two failures from the report: plain-text stage direction and shared surnames. All offline."""

from personaforge.agent import check_in_character
from personaforge.llm import ScriptedLLM, ToolCall, ToolResponse
from personaforge.orchestrator import Orchestrator
from personaforge.persona import generate_persona, list_personas, save_persona, surname_of, PersonaSpec


# -- plain-text stage direction ---------------------------------------------

def test_narrated_actions_from_the_transcript_are_flagged():
    lines = [
        "I look up from the medication cart and sigh.",
        "I pause, choosing my words.",
        "I nod once.",
        "I take a drink of the cold coffee.",
        "I glance back once as I pass the monitor station.",
        "Fine. I nod to you. Let's go.",
    ]
    for line in lines:
        assert "narrates its own physical action in plain text" in check_in_character(line, "Margaret"), line


def test_ordinary_dialogue_with_i_is_not_flagged():
    lines = [
        "I understand. Could you tell me more about when the symptoms started?",
        "I would like to run a couple of tests.",
        "I look forward to hearing the results.",
        "I think we should check triage first.",
    ]
    for line in lines:
        assert check_in_character(line, "Margaret") == [], line


# -- shared surnames ----------------------------------------------------------

def test_surname_of():
    assert surname_of("Margaret Chen") == "chen"
    assert surname_of("Dr. Marcus Chen") == "chen"
    assert surname_of("Maria") == ""


def persona_md(name):
    return f"---\nname: {name}\nrole: doctor\nsummary: a doctor\n---\n\n# {name}\n\nBackground."


def test_generation_retries_when_the_surname_is_taken():
    llm = ScriptedLLM(replies=[persona_md("Ana Chen"), persona_md("Ana Okafor")])
    spec = generate_persona("a doctor", llm, avoid_surnames=frozenset({"chen"}))
    assert spec.name == "Ana Okafor"
    assert len(llm.calls) == 2
    assert "chen" in llm.calls[1]["messages"][0]["content"]


def test_no_retry_when_nothing_is_taken():
    llm = ScriptedLLM(replies=[persona_md("Ana Chen")])
    spec = generate_persona("a doctor", llm)
    assert spec.name == "Ana Chen"
    assert len(llm.calls) == 1


def test_last_attempt_is_kept_even_if_it_still_clashes():
    llm = ScriptedLLM(replies=[persona_md("Ana Chen"), persona_md("Ben Chen")])
    spec = generate_persona("a doctor", llm, avoid_surnames=frozenset({"chen"}))
    assert spec.name == "Ben Chen"
    assert len(llm.calls) == 2


class AuthorLLM:
    """Routes one create_persona request, then answers the authoring call from a fixed script."""

    def __init__(self, first_name, second_name):
        self.names = [first_name, second_name]
        self.author_prompts = []
        self.routed = False

    def complete(self, system, messages, max_tokens=2048):
        self.author_prompts.append(messages[-1]["content"])
        return persona_md(self.names.pop(0))

    def complete_with_tools(self, system, messages, tools, max_tokens=2048):
        if not self.routed:
            self.routed = True
            call = ToolCall(id="t1", name="create_persona", input={"description": "a doctor"})
            return ToolResponse(text="", tool_calls=[call], content=[{"type": "tool_use", "id": "t1", "name": "create_persona", "input": {"description": "a doctor"}}])
        return ToolResponse(text="done", content=[{"type": "text", "text": "done"}])


def test_orchestrator_avoids_the_surname_of_a_saved_persona(tmp_path):
    save_persona(PersonaSpec(name="Margaret Chen", role="nurse", summary="s", body="b"), tmp_path)
    llm = AuthorLLM(first_name="Dr. Marcus Chen", second_name="Dr. Marcus Okafor")
    Orchestrator(llm, persona_dir=tmp_path).handle("create a persona doctor")

    names = sorted(p.name for p in list_personas(tmp_path))
    assert names == ["Dr. Marcus Okafor", "Margaret Chen"]
    assert "chen" in llm.author_prompts[0].lower() or "Do not use" in llm.author_prompts[0]
