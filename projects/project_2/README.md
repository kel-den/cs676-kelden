# Project 2 — PersonaForge: An Agent-to-Agent Package

**CS676 Algorithms for Data Science • Pace University, Seidenberg School**
**Kelden Tshomo**

## 📌 Quick Links

| Link | Purpose |
|------|---------|
| **[🚀 Live App](https://personaforge.streamlit.app/)** | Try PersonaForge live on Streamlit |
| **[📄 Technical Report](./CS676_report.pdf)** | Design, transcript commentary, failure analysis, coverage (9 pages) |
| **[🎬 Video Demo](./livedemo.mp4)** | Live walkthrough of the app |
| **[📊 Professor's Repo](https://github.com/yiqiao-yin/pace-u-cs676)** | Original project template |

---

## 🎯 What It Does

PersonaForge is a Python package where fictional characters (personas) talk to each
other. You describe a character in plain English, and the model writes it as a markdown
file. Then you ask two characters to discuss something and watch the conversation.

- **A persona is a markdown file.** Edit the file and the character changes.
- **An agent is that file plus a model.**
- **A conversation is agents taking turns.**

```
you › make me a tired night-shift nurse and a brisk attending physician
you › have them talk about whether the unit is short-staffed
```

---

## 🔧 My Implementation

The course skeleton routed commands with regular expressions, kept no state between
conversations, and ran fixed-length conversations. I chose to do two extensions in depth
plus part of a third:

### 1. Model-driven routing with tool use (`orchestrator.py`)
I replaced the regex router with a tool-use loop. The model gets three tools,
`create_persona`, `list_personas` and `start_conversation`, and decides which one to call.
- Understands varied phrasing and **asks a clarifying question** when something is missing
  (no description, no topic, or more than two personas and none named).
- The loop is capped at 5 rounds. Tool errors go back to the model instead of crashing the UI.
- **Fixed reply after a conversation.** In my first live run the model made up its own
  version of the dialogue. Now the loop stops once a conversation finishes and returns a
  fixed message with the participants and the transcript path.

### 2. Private agent memory (`agent.py`)
- Each agent has its own memory file at `<persona_dir>/memory/<slug>.md`, kept separate
  from the persona file so the character description is never overwritten.
- After each conversation, the agent adds a dated entry with the topic, who it talked
  with, and up to 6 of its own lines.
- Before the next conversation, the last 3,000 characters of the file go into **that
  agent's prompt only**. Other agents never see it (tested in `tests/test_recall.py`).

### 3. Packaging and robustness
- A clean public API in `__init__.py`. Version `0.2.0`.
- `AmbiguousPersonaError` lists the matching personas, so the caller can ask which one
  was meant.
- Saving a persona never overwrites an existing file. A numeric suffix is added instead.
- A missing API key or network error returns a message saying what to check.

### Fixes found in live runs
- **Plain-text stage directions** ("I nod", "I pause") are now flagged by the
  in-character check, not just `*asterisks*` and `[brackets]`.
- **Duplicate surnames.** Two personas both named Chen confused each other. New personas
  now avoid surnames that are already saved (with one retry).

See the [report](./CS676_report.pdf) for the alternatives I considered, the transcript
commentary, and the failures that are still open.

---

## 🚀 Setup

You need **Python 3.10+** and **[uv](https://docs.astral.sh/uv/)**.

```bash
git clone https://github.com/kel-den/cs676-kelden.git
cd cs676-kelden/projects/project_2

uv sync
cp .env.example .env      # then add your ANTHROPIC_API_KEY
```

On Windows, use `copy .env.example .env`.

## ▶️ Run It

```bash
uv run main.py --offline                  # scripted model: no key, no cost
uv run main.py                            # live model (needs ANTHROPIC_API_KEY)
uv run main.py --model claude-haiku-4-5   # cheaper model while testing
```

| Flag | Effect |
|---|---|
| `--offline` | Scripted replies, no API key needed |
| `--model <id>` | Choose the Claude model |
| `--persona-dir <dir>` | Store personas somewhere other than `temp/` |

Inside the app: `/help`, `/clear`, `/quit`.

**Web UI (Streamlit):**

```bash
uv run --with streamlit streamlit run app.py
```

`app.py` is a thin front end over the orchestrator. If no key is set, it runs in offline
demo mode. When deployed, it reads `ANTHROPIC_API_KEY` from Streamlit Secrets.

**Cost:** one model call per conversation turn, plus routing. Creating a persona takes
about 3 calls, and a 6-turn conversation takes about 7.

---

## 🧪 Tests

```bash
uv run pytest
```

All tests run offline using `ScriptedLLM`, so no key is needed. Currently: **81 passed,
7 skipped**. The 7 skipped tests are the original regex-router tests. I kept them with a
skip reason instead of deleting them.

---

## 📁 Project Layout

```
project_2/
├── main.py                 entry point (CLI)
├── app.py                  Streamlit web front end
├── pyproject.toml / uv.lock
├── requirements.txt        for deployment hosts
├── src/personaforge/
│   ├── __init__.py         public API
│   ├── llm.py              ClaudeLLM / ScriptedLLM / offline demo model
│   ├── persona.py          generate, save, load, find personas
│   ├── agent.py            in-character prompt, character check, private memory
│   ├── conversation.py     turn loop and transcripts
│   ├── orchestrator.py     tool-use routing loop
│   ├── ui.py, cli.py       terminal interface
├── tests/                  pytest, all offline
├── temp/                   personas, memory and transcripts (gitignored)
├── CS676_report.pdf
└── livedemo.mp4
```

---

## ⚠️ Known Limitations

- Turn-taking is still round-robin. Agents can `[PASS]` but cannot interrupt or end early.
- There is no observer or judge agent.
- Memory recall uses the most recent text, not the most relevant.
- Agents see only the last 12 turns, so a detail from early in a long conversation can be
  lost.
- The character check is heuristic. It flags problems in the transcript but does not retry
  the turn.
