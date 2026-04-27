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


__all__ = ["build_system_prompt"]
