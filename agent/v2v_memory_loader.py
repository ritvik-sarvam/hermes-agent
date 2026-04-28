"""V2V system-prompt assembler (Milestone 6.1).

The v2v hackathon harness keeps three pieces of context as plain
markdown files on disk:

1. A *global* agent workflow / SOP (e.g. ``<root>/agent_workflow.md``)
   shared by every user.
2. A *per-user* memory blob (e.g. ``<root>/users/<user_id>/memory.md``)
   that the reflection cron in M9 will keep up-to-date.
3. The currently-active *skill* card text (selected by the skill router
   in M7 — empty for now).

This module concatenates them into a single string suitable for the
``system`` field of an OpenAI-style chat completion. Missing files are
skipped silently so the harness can boot without a global SOP and new
users without a memory file still get a working prompt.

We deliberately do **not** wire this into Hermes' :mod:`agent.memory_manager`
yet — for the hackathon the explicit-files-on-disk model is simpler and
what the reflection cron writes into anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def _read_text_or_none(path: PathLike) -> str | None:
    """Return the file's text contents, or ``None`` if it doesn't exist
    or can't be read. Empty files return ``""`` (not ``None``) — they
    indicate "section is intentionally empty" rather than "absent".
    """
    p = Path(path)
    try:
        if not p.is_file():
            return None
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def build_system_prompt(
    *,
    global_path: PathLike,
    user_path: PathLike,
    skill_text: str = "",
) -> str:
    """Assemble the v2v system prompt from three source pieces.

    The returned prompt has up to three sections, in this order:

    * ``## Global SOP``
    * ``## User memory``
    * ``## Active skill``

    Sections whose source is missing (or empty for the skill text) are
    omitted entirely — including their header — so a downstream LLM
    isn't asked to "respect the empty user memory above."

    Parameters
    ----------
    global_path:
        Path to the shared workflow / SOP markdown file.
    user_path:
        Path to the per-user ``memory.md`` markdown file.
    skill_text:
        Inline text of the currently-active skill card. May be empty
        (Milestone 7 produces real content).
    """
    parts: list[str] = []

    global_text = _read_text_or_none(global_path)
    if global_text is not None and global_text.strip():
        parts.append("## Global SOP\n\n" + global_text.rstrip())

    user_text = _read_text_or_none(user_path)
    if user_text is not None and user_text.strip():
        parts.append("## User memory\n\n" + user_text.rstrip())

    skill_text = (skill_text or "").strip()
    if skill_text:
        parts.append("## Active skill\n\n" + skill_text)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Voice-mode skill loader
# ---------------------------------------------------------------------------
#
# Skills (router + specialists) on disk are written for an agent runtime
# with real tool execution — they instruct the agent to call ``skills_tool``,
# ``cronjob``, ``lookup_order``, etc. The hackathon V2VAgentSession does not
# have a tool-execution loop yet (deferred to Hermes-AIAgent integration);
# letting those instructions land verbatim in the system prompt causes the
# model to emit XML tool-call markup that streams into TTS as gibberish
# (see the ClauseChunker's suppression set as the secondary defense).
#
# ``load_router_for_voice`` reads the router skill file and returns only
# the voice-safe sections — Persona, Proactivity clause, Anti-spam
# guardrails, Pitfalls — with frontmatter stripped and tool references
# rewritten as natural language. The returned string is suitable to pass
# straight into ``build_system_prompt(skill_text=...)``.

# Header phrases (lowercased) we KEEP for voice mode. Anything else
# (Procedure, When to Use, Verification, etc.) is dropped because it
# tells the agent how to operate tools.
_VOICE_SAFE_SECTIONS = (
    "persona",
    "proactivity clause",
    "anti-spam guardrails",
    "pitfalls",
)


def _strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block delimited by ``---`` lines."""
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) >= 3:
        # parts[0] is "" before the first ---; parts[1] is frontmatter;
        # parts[2] is the body.
        return parts[2].lstrip("\n")
    return text


def _section_is_voice_safe(header: str) -> bool:
    h = header.strip().lower().lstrip("#").strip()
    return any(h == safe or h.startswith(safe) for safe in _VOICE_SAFE_SECTIONS)


def load_router_for_voice(router_path: PathLike) -> str:
    """Read the router skill markdown and return a voice-safe rendering.

    Steps:

    1. Strip YAML frontmatter.
    2. Walk top-level (``##``) sections. Keep only the voice-safe ones.
    3. Soften residual tool-call references in the kept text — a few
       sentences mention ``cronjob`` by name (the proactivity clause).
       Replace with natural-language phrasing so the model doesn't
       interpret it as an instruction to emit markup.

    Returns ``""`` if the file is missing or all sections are
    voice-unsafe (callers can pass that into ``build_system_prompt``
    which will then omit the active-skill section).
    """
    text = _read_text_or_none(router_path)
    if not text:
        return ""

    body = _strip_frontmatter(text)

    kept: list[str] = []
    current_section: list[str] | None = None
    keep_current = False

    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            # Flush the previous section if it was kept.
            if current_section is not None and keep_current:
                kept.append("\n".join(current_section).rstrip())
            current_section = [line]
            keep_current = _section_is_voice_safe(stripped)
            continue
        if current_section is not None:
            current_section.append(line)
        # Pre-section content (intro paragraph, the H1) is also dropped —
        # it usually says "Always loaded" / "Routes the conversation" /
        # similar tool-meta phrasing.
    # Final flush.
    if current_section is not None and keep_current:
        kept.append("\n".join(current_section).rstrip())

    if not kept:
        return ""

    rendered = "\n\n".join(kept)

    # Soften residual tool-call references. The proactivity clause says
    # "schedule it via the `cronjob` tool" — without a tool-execution
    # loop the model would happily emit a <tool_call>cronjob...</tool_call>
    # block. Rephrase as "make a note to follow up later" — the actual
    # scheduling will land via the post-call reflection loop instead.
    rendered = rendered.replace(
        "schedule it via the `cronjob` tool.",
        "make a clear verbal commitment to follow up, and we'll record it in your memory file.",
    )
    rendered = rendered.replace(
        "via the `cronjob` tool",
        "via your memory file (a follow-up will be scheduled offline)",
    )
    rendered = rendered.replace(
        "Before creating a `cronjob`",
        "Before promising a follow-up",
    )

    return rendered


__all__ = ["build_system_prompt", "load_router_for_voice"]
