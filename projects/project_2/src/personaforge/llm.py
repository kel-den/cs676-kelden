"""
llm.py — the thin layer between a persona and a model.

Everything in this package that needs to "think" goes through the `LLM`
protocol below. There are two implementations:

    ClaudeLLM    real Claude calls. Needs ANTHROPIC_API_KEY.
    ScriptedLLM  returns canned strings. No key, no network, no cost.

That split is deliberate and it is the reason this project is testable. The
unit tests in `tests/` inject a ScriptedLLM, so they run in milliseconds for
free, and `main.py --offline` uses one too so you can see the whole app work
before you have configured anything.

Tool use: `complete_with_tools()` lets the model request actions (create a
persona, start a conversation) instead of the orchestrator guessing from regular
expressions. It returns a `ToolResponse` holding any tool requests plus the raw
content blocks, which must be echoed back in the assistant turn.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

# Check this ID against the current model list before a live run. Model IDs
# change, and a wrong one fails every call.
DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class ToolCall:
    """One tool the model asked us to run."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResponse:
    """
    What the model returned in one request.

    :param text:       any plain text the model wrote (may be empty)
    :param tool_calls: tools the model wants run; empty means it is finished
    :param content:    raw content blocks, echoed back as the assistant turn so
                       the API can match each tool_result to its tool_use
    """

    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    content: List[Dict[str, Any]] = field(default_factory=list)


class LLM(Protocol):
    """Anything that can turn a system prompt plus a transcript into a reply."""

    def complete(self, system: str, messages: List[Dict[str, Any]], max_tokens: int = 2048) -> str:
        """Send one request and return the model's text reply."""
        ...

    def complete_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 2048,
    ) -> ToolResponse:
        """Send one request that offers `tools`, and return any text and tool requests."""
        ...


# =============================================================================
# REAL MODEL CALLS
# =============================================================================
# ClaudeLLM is the only code that calls the Anthropic API. It needs
# ANTHROPIC_API_KEY (copy .env.example to .env). Calls are billed to the key owner:
# a conversation costs one call per turn, plus one call per persona created.
#
# Verified live: complete() for persona authoring and multi-turn dialogue, and
# complete_with_tools() for the router (create, list, start a conversation).
# Offline runs use DemoLLM and need no key.
# =============================================================================


@dataclass
class ClaudeLLM:
    """
    Real Claude. The client is created lazily so importing this module never
    requires a key.
    """

    model: str = DEFAULT_MODEL
    _client: Any = field(default=None, repr=False)

    def _get_client(self) -> Any:
        """Create the Anthropic client on first use, so importing this module never needs a key."""
        if self._client is None:
            import anthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                    "or run in offline mode: uv run main.py --offline"
                )
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, messages: List[Dict[str, Any]], max_tokens: int = 2048) -> str:
        """Send one request and return the concatenated text of the reply."""
        return self.complete_with_tools(system, messages, tools=[], max_tokens=max_tokens).text

    def complete_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 2048,
    ) -> ToolResponse:
        """Send one request offering `tools`; return text and any tool requests."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self._get_client().messages.create(**kwargs)

        # Claude can decline a request; content is empty or partial when it does.
        if response.stop_reason == "refusal":
            return ToolResponse(text="[declined to respond]")

        content = [block.model_dump(exclude_none=True) for block in response.content]
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        calls = [
            ToolCall(id=b.id, name=b.name, input=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        return ToolResponse(text=text, tool_calls=calls, content=content)


@dataclass
class ScriptedLLM:
    """
    A fake model that replays a fixed list of replies, then loops.

    Used by the tests and by `--offline`. It records every call it received in
    `.calls`, which makes it easy to assert that a persona's markdown actually
    reached the system prompt. It never requests tools.
    """

    replies: List[str] = field(default_factory=lambda: ["(scripted reply)"])
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Dict[str, Any]], max_tokens: int = 2048) -> str:
        """Send one request and return the model's text reply."""
        self.calls.append({"system": system, "messages": list(messages)})
        return self.replies[(len(self.calls) - 1) % len(self.replies)]

    def complete_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 2048,
    ) -> ToolResponse:
        """Send one request that offers `tools`, and return any text and tool requests."""
        text = self.complete(system, messages, max_tokens=max_tokens)
        return ToolResponse(text=text, content=[{"type": "text", "text": text}])


_DEMO_DIALOGUE = [
    "I understand. Could you tell me more about when the symptoms started?",
    "That has been going on for about three weeks now, mostly in the mornings.",
    "Have you noticed anything that makes it better or worse?",
    "Coffee seems to make it worse, and lying down helps a little.",
    "Thank you, that is useful. I would like to run a couple of tests.",
    "Whatever you think is best, doctor. I just want to feel normal again.",
]

_DEMO_PERSONA = """---
name: {name}
role: {role}
summary: a demo persona generated in offline mode
---

# {name}

## Background
This persona was produced by the offline stub, not by a model. Run without
`--offline` to get a real character written by Claude.

## Personality
- Placeholder
- Placeholder

## How they speak
- In whatever the scripted stub happens to return
"""


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    """The most recent plain-text user message, or '' if there is none."""
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(message["content"], str):
            return message["content"]
    return ""


@dataclass
class DemoLLM:
    """
    The offline stub used by `--offline`.

    A single ScriptedLLM cannot serve every job: asked to author a persona it
    would return a line of dialogue, and you would get a character named
    "Unnamed". So this one looks at the system prompt and answers in the shape
    the caller expects.

    For tool requests it uses simple keyword rules. That is a stub for offline
    testing only. The real router is the model.
    """

    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Dict[str, Any]], max_tokens: int = 2048) -> str:
        """Send one request and return the model's text reply."""
        self.calls.append({"system": system, "messages": list(messages)})
        request = messages[-1]["content"] if messages and isinstance(messages[-1]["content"], str) else ""

        # The persona-author prompt is the only one that asks for frontmatter.
        if "role: <one lowercase word" in system:
            description = request.replace("Write a persona for:", "").strip()
            words = [w for w in description.split() if w.isalpha()]
            role = words[0].lower() if words else "person"
            return _DEMO_PERSONA.format(name=description[:40].title() or "Demo Persona", role=role)

        if "stage manager" in system.lower():
            return "Offline mode — I'm a stub. Try: create a persona doctor"

        return _DEMO_DIALOGUE[(len(self.calls) - 1) % len(_DEMO_DIALOGUE)]

    def complete_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 2048,
    ) -> ToolResponse:
        """Send one request that offers `tools`, and return any text and tool requests."""
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})

        # Once a tool has run, the last message holds tool_result blocks.
        # Finish the turn with no further tool requests.
        if messages and isinstance(messages[-1]["content"], list):
            return ToolResponse(text="", content=[{"type": "text", "text": ""}])

        text = _last_user_text(messages)
        lowered = text.lower()

        if "persona" in lowered and re.search(r"\b(create|make|add|build|generate)\b", lowered):
            description = re.sub(r"^.*?\bpersona\b", "", text, count=1, flags=re.IGNORECASE).strip()
            return self._tool("create_persona", {"description": description})

        if "persona" in lowered and re.search(r"\b(list|show|who)\b", lowered):
            return self._tool("list_personas", {})

        if re.search(r"\b(talk|conversation|discuss|converse)\b", lowered):
            about = re.search(r"\babout\s+(.*)$", text, flags=re.IGNORECASE)
            topic = about.group(1).strip() if about else ""
            if not topic:
                # A real model would ask here. The stub mirrors that.
                return ToolResponse(text="What should they talk about?", content=[{"type": "text", "text": "What should they talk about?"}])
            return self._tool("start_conversation", {"topic": topic})

        reply = "Offline mode — I'm a stub. Try: create a persona doctor"
        return ToolResponse(text=reply, content=[{"type": "text", "text": reply}])

    def _tool(self, name: str, tool_input: Dict[str, Any]) -> ToolResponse:
        """Wrap one tool request in the ToolResponse shape the orchestrator expects."""
        call_id = f"offline_{len(self.calls)}"
        block = {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}
        return ToolResponse(
            text="",
            tool_calls=[ToolCall(id=call_id, name=name, input=tool_input)],
            content=[block],
        )


def default_llm(offline: bool = False, model: Optional[str] = None) -> LLM:
    """Pick an implementation. Central so you only change this in one place."""
    if offline:
        return DemoLLM()
    return ClaudeLLM(model=model or DEFAULT_MODEL)
