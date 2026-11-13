"""
persona.py — a persona is a markdown file on disk.

That is the whole idea of this package. There is no persona database and no
in-memory registry. A persona is a `.md` file in `temp/`, and anything that can
read that file can wake the persona up. You can open one in a text editor,
change a sentence, and the agent behaves differently on the next run.

File format — frontmatter for the machine, markdown for the model:

    ---
    name: Maria Delgado
    role: patient
    summary: 58-year-old with poorly controlled type 2 diabetes
    ---

    # Maria Delgado

    ## Background
    ...

The frontmatter parser here is about fifteen lines and handles exactly the
three keys above. That is intentional. See KNOWN LIMITATIONS at the bottom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .llm import LLM

# Where persona files live. Relative to wherever you launched the app.
DEFAULT_PERSONA_DIR = Path("temp")


@dataclass
class PersonaSpec:
    """One persona: the metadata we index on, plus the markdown that is its mind."""

    name: str
    role: str
    summary: str
    body: str  # markdown; becomes the agent's system prompt

    def slug(self) -> str:
        """Filesystem-safe stem, e.g. 'Maria Delgado' -> 'maria-delgado'."""
        s = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        return s or "unnamed"

    def to_markdown(self) -> str:
        """Serialize to the frontmatter + body format shown above."""
        return (
            "---\n"
            f"name: {self.name}\n"
            f"role: {self.role}\n"
            f"summary: {self.summary}\n"
            "---\n\n"
            f"{self.body.strip()}\n"
        )

    @classmethod
    def from_markdown(cls, text: str) -> "PersonaSpec":
        """
        Parse a persona file back into a spec.

        Missing frontmatter is tolerated: we fall back to sensible defaults and
        treat the whole file as the body, so a hand-written persona still loads.
        """
        meta = {"name": "Unnamed", "role": "unknown", "summary": ""}
        text = text.strip()

        # Models sometimes wrap the whole document in a code fence.
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
            text = text.strip()

        body = text

        if text.startswith("---"):
            rest = text[3:].lstrip("\n")
            closing = re.search(r"^---\s*$", rest, re.MULTILINE)
            if closing:
                # Well-formed: frontmatter is everything up to the closing fence.
                front, body = rest[:closing.start()], rest[closing.end():]
            else:
                # The closing '---' is missing, which the model does drop from
                # time to time. Rather than throw the whole document away, read
                # `key: value` lines from the top and treat the first line that
                # is not one as the start of the body.
                front_lines, body_lines, in_front = [], [], True
                for line in rest.splitlines():
                    if in_front and re.match(r"^[A-Za-z_][A-Za-z0-9_ -]*:", line):
                        front_lines.append(line)
                    elif in_front and not line.strip():
                        continue            # blank line inside the header block
                    else:
                        in_front = False
                        body_lines.append(line)
                front, body = "\n".join(front_lines), "\n".join(body_lines)

            for line in front.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip().lower()
                    if key in meta:
                        meta[key] = value.strip()

        return cls(name=meta["name"], role=meta["role"], summary=meta["summary"], body=body.strip())


def save_persona(spec: PersonaSpec, directory: Path = DEFAULT_PERSONA_DIR) -> Path:
    """
    Write the persona to `<directory>/<slug>.md` and return the path.

    If that file already exists, the new persona gets `<slug>-2.md`, then
    `-3`, and so on. An earlier persona is never overwritten.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.slug()}.md"
    n = 2
    while path.exists():
        path = directory / f"{spec.slug()}-{n}.md"
        n += 1
    path.write_text(spec.to_markdown(), encoding="utf-8")
    return path


def load_persona(path: Path) -> PersonaSpec:
    """Read one persona file from disk."""
    return PersonaSpec.from_markdown(Path(path).read_text(encoding="utf-8"))


def list_personas(directory: Path = DEFAULT_PERSONA_DIR) -> List[PersonaSpec]:
    """
    Every persona currently saved, sorted by filename. Empty if none yet.

    Only files with a `name:` in their frontmatter count as personas. That skips
    conversation transcripts, which share the folder and would otherwise show up
    as a character called "Unnamed".
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    personas: List[PersonaSpec] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^name:", text.lstrip()[:500], re.MULTILINE):
            personas.append(PersonaSpec.from_markdown(text))
    return personas


class AmbiguousPersonaError(LookupError):
    """More than one saved persona matches a name, so we refuse to guess."""

    def __init__(self, query: str, matches: List["PersonaSpec"]) -> None:
        """Build the error message that lists each matching persona."""
        self.query = query
        self.matches = matches
        options = ", ".join(f"{m.name} ({m.role})" for m in matches)
        super().__init__(f"'{query}' matches more than one persona: {options}.")


def find_persona(name: str, directory: Path = DEFAULT_PERSONA_DIR) -> Optional[PersonaSpec]:
    """
    Look a persona up by name, slug, or role, case-insensitively.

    Exact matches on name, slug, or role win. Otherwise we fall back to a
    substring match. If exactly one persona matches, it is returned. If several
    match, AmbiguousPersonaError is raised, listing them, so the caller can ask
    the user which one they meant. If none match, returns None.
    """
    needle = name.strip().lower()
    if not needle:
        return None

    personas = list_personas(directory)
    exact = [s for s in personas if needle in (s.name.lower(), s.slug(), s.role.lower())]
    loose = [s for s in personas if needle in s.name.lower() or needle in s.role.lower()]
    matches = exact or loose

    if not matches:
        return None
    if len(matches) > 1:
        raise AmbiguousPersonaError(name.strip(), matches)
    return matches[0]


# -----------------------------------------------------------------------------
# Generating a persona with the model
# -----------------------------------------------------------------------------

_AUTHOR_SYSTEM = """You write character specifications for a multi-agent simulation.

Given a short description, write a persona as a markdown document. Output ONLY the
markdown — no preamble, no code fences.

Use exactly this structure:

---
name: <a specific, realistic full name>
role: <one lowercase word, e.g. patient, doctor, teacher, customer>
summary: <one line, under 15 words>
---

# <name>

## Background
Two or three sentences. Be concrete and specific — age, situation, history.

## Personality
Three or four bullet points about temperament and outlook.

## How they speak
Two or three bullets on voice: vocabulary, sentence length, verbal habits.

## What they want
One or two sentences on their goal in a conversation.

## What they avoid
One or two sentences on what they will not say or do.

Write a specific individual, not a type. Give them a quirk. Avoid stereotypes,
and do not make them a spokesperson for a demographic."""


def surname_of(name: str) -> str:
    """The last word of a name in lower case, or '' if the name has only one word."""
    parts = name.split()
    return parts[-1].lower() if len(parts) > 1 else ""


MAX_AUTHOR_ATTEMPTS = 2


def generate_persona(description: str, llm: LLM, avoid_surnames: frozenset = frozenset()) -> PersonaSpec:
    """
    Ask the model to author a persona from a one-line description.

    If the new persona reuses a surname in `avoid_surnames`, the model is asked
    again, up to MAX_AUTHOR_ATTEMPTS times. The last attempt is kept even if it
    still clashes, so a persona is always produced.

    :param description:     e.g. "a patient with type 2 diabetes who distrusts doctors"
    :param llm:             any LLM implementation (real or scripted)
    :param avoid_surnames:  lower-case surnames already in use
    :return:                the parsed PersonaSpec, not yet saved to disk
    """
    prompt = f"Write a persona for: {description}"
    if avoid_surnames:
        prompt += "\nDo not use any of these surnames: " + ", ".join(sorted(avoid_surnames)) + "."

    spec = None
    for _ in range(MAX_AUTHOR_ATTEMPTS):
        markdown = llm.complete(
            system=_AUTHOR_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        spec = PersonaSpec.from_markdown(markdown)

        # If the model ignored the frontmatter format we still want something usable,
        # so fall back to the raw description rather than failing the whole command.
        if spec.name == "Unnamed":
            spec.name = description[:40].strip().title() or "Unnamed"
            spec.summary = description.strip()

        if surname_of(spec.name) not in avoid_surnames:
            break
    return spec


# =============================================================================
# KNOWN LIMITATIONS
# =============================================================================
#  - The frontmatter parser reads three fixed keys with no quoting. A summary
#    containing a colon is kept, but its exact punctuation is not guaranteed.
#  - Model output is only checked for the frontmatter. A persona whose body is
#    prose instead of the requested sections is accepted as it is.
#  - Personas cannot be edited inside the app. Change the markdown file instead.
#  - A persona does not change during a conversation, so agents learn only
#    through their memory file.
# =============================================================================
