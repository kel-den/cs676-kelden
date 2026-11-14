"""Tests for writing character-check findings into the transcript, and for the persona-dir setting. All offline."""

from pathlib import Path

from personaforge import PersonaSpec, agent_from_spec
from personaforge.cli import build_parser
from personaforge.conversation import Conversation, Turn, run_conversation, save_transcript
from personaforge.llm import ScriptedLLM
from personaforge.persona import DEFAULT_PERSONA_DIR


def make_agent(name, replies):
    spec = PersonaSpec(name=name, role="nurse", summary="tired", body=f"# {name}\n\nA nurse.")
    return agent_from_spec(spec, ScriptedLLM(replies=replies))


# -- transcript rendering ------------------------------------------------------

def test_transcript_without_issues_has_no_check_section():
    convo = Conversation(topic="staffing", participants=["A", "B"], turns=[Turn("A", "Hello.")])
    assert "## Character check" not in convo.transcript()


def test_transcript_lists_each_issue_under_a_check_heading():
    convo = Conversation(
        topic="staffing",
        participants=["A", "B"],
        turns=[Turn("A", "Hello.")],
        issues=["Turn 1, A: stage direction in asterisks", "Turn 2, B: reply is far too long"],
    )
    text = convo.transcript()
    assert "**A:** Hello." in text
    # The findings come after the dialogue, one bullet per finding.
    assert text.index("## Character check") > text.index("**A:** Hello.")
    assert "- Turn 1, A: stage direction in asterisks" in text
    assert "- Turn 2, B: reply is far too long" in text


# -- run_conversation records what the check found ----------------------------

def test_clean_conversation_records_no_issues():
    a = make_agent("Margaret", ["We are three nurses short tonight."])
    b = make_agent("Marcus", ["Then we triage the admit queue first."])
    convo = run_conversation([a, b], "staffing", turns=4)
    assert convo.issues == []
    assert "## Character check" not in convo.transcript()


def test_issues_are_recorded_with_turn_number_and_speaker():
    a = make_agent("Margaret", ["We are short. *sighs*"])
    b = make_agent("Marcus", ["Then we triage first."])
    convo = run_conversation([a, b], "staffing", turns=3)

    # Margaret speaks on turns 1 and 3; Marcus's clean lines add nothing.
    assert convo.issues == [
        "Turn 1, Margaret: stage direction in asterisks",
        "Turn 3, Margaret: stage direction in asterisks",
    ]


def test_saved_transcript_contains_the_check_section(tmp_path):
    a = make_agent("Margaret", ["I nod. Fine."])
    b = make_agent("Marcus", ["Go."])
    convo = run_conversation([a, b], "staffing", turns=2)
    path = save_transcript(convo, tmp_path, filename="t.md")

    saved = path.read_text(encoding="utf-8")
    assert "## Character check" in saved
    assert "Turn 1, Margaret: narrates its own physical action in plain text" in saved


# -- persona directory setting (cli.py) ---------------------------------------

def test_persona_dir_defaults_to_temp(monkeypatch):
    monkeypatch.delenv("PERSONAFORGE_PERSONA_DIR", raising=False)
    args = build_parser().parse_args([])
    assert Path(args.persona_dir) == Path(DEFAULT_PERSONA_DIR)


def test_persona_dir_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("PERSONAFORGE_PERSONA_DIR", "/somewhere/else")
    args = build_parser().parse_args([])
    assert args.persona_dir == "/somewhere/else"


def test_persona_dir_flag_overrides_environment_variable(monkeypatch):
    monkeypatch.setenv("PERSONAFORGE_PERSONA_DIR", "/somewhere/else")
    args = build_parser().parse_args(["--persona-dir", "/explicit"])
    assert args.persona_dir == "/explicit"
