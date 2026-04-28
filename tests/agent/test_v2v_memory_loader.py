"""Tests for ``agent.v2v_memory_loader.build_system_prompt`` (Task 6.1).

The v2v hackathon uses explicit ``memory.md`` files rather than wiring
into Hermes' ``memory_manager``: the loader concatenates a global SOP, a
per-user memory blob, and the active skill text into a single system
prompt with clear markdown section dividers. Missing files are skipped
gracefully so the hackathon harness can boot without a global SOP and
new users without a memory file still get a working prompt.
"""

from __future__ import annotations

from pathlib import Path

from agent.v2v_memory_loader import build_system_prompt, load_router_for_voice


def test_includes_global_workflow_and_user_memory(tmp_path):
    g = tmp_path / "agent_workflow.md"
    g.write_text("GLOBAL SOP")
    u = tmp_path / "users/u1/memory.md"
    u.parent.mkdir(parents=True)
    u.write_text("USER FACTS")
    s = build_system_prompt(global_path=g, user_path=u, skill_text="SKILL")
    assert "GLOBAL SOP" in s and "USER FACTS" in s and "SKILL" in s
    assert s.index("GLOBAL SOP") < s.index("USER FACTS") < s.index("SKILL")


def test_missing_global_path_omits_section(tmp_path):
    u = tmp_path / "users/u1/memory.md"
    u.parent.mkdir(parents=True)
    u.write_text("U")
    s = build_system_prompt(global_path=tmp_path / "missing.md", user_path=u, skill_text="S")
    assert "GLOBAL" not in s.upper()  # no header for absent global
    assert "U" in s and "S" in s


def test_missing_user_path_creates_empty_section_or_omits(tmp_path):
    g = tmp_path / "g.md"
    g.write_text("G")
    s = build_system_prompt(global_path=g, user_path=tmp_path / "absent.md", skill_text="S")
    assert "G" in s and "S" in s
    # should not crash; either omit user section or include an empty placeholder


def test_section_headers_are_clear(tmp_path):
    g = tmp_path / "g.md"
    g.write_text("G")
    u = tmp_path / "u.md"
    u.write_text("U")
    s = build_system_prompt(global_path=g, user_path=u, skill_text="S")
    assert "## " in s  # has markdown section dividers


def test_missing_skill_text_still_produces_prompt(tmp_path):
    g = tmp_path / "g.md"
    g.write_text("G")
    u = tmp_path / "u.md"
    u.write_text("U")
    s = build_system_prompt(global_path=g, user_path=u, skill_text="")
    assert "G" in s and "U" in s


def test_all_missing_returns_empty_string_or_minimal(tmp_path):
    s = build_system_prompt(
        global_path=tmp_path / "no.md",
        user_path=tmp_path / "no2.md",
        skill_text="",
    )
    # Should not crash, may be empty.
    assert isinstance(s, str)


# ---------------------------------------------------------------------------
# load_router_for_voice — strip frontmatter, keep voice-safe sections,
# soften residual tool-call references.
# ---------------------------------------------------------------------------


_ROUTER_FIXTURE = """\
---
name: v2v-router
description: anything
---

# V2V Router

Intro paragraph that should be dropped.

## Persona

You are a customer support agent. Speak in short sentences.

## When to Use

This is meta-instruction (drops).

## Procedure

1. Call the skill-loader tool with action `view`.
2. schedule it via the `cronjob` tool.

## Proactivity clause

Whenever the user mentions a commitment, schedule it via the `cronjob` tool.

## Anti-spam guardrails

Per-user weekly cap is 3. Quiet hours 09:00-20:00 user-local.
Before creating a `cronjob`, list existing scheduled items.

## Pitfalls

- Don't load specialists speculatively.
- Don't read order IDs character-by-character.

## Verification

After routing, the loaded specialist runs.
"""


def test_load_router_for_voice_keeps_safe_sections(tmp_path: Path) -> None:
    p = tmp_path / "router" / "SKILL.md"
    p.parent.mkdir()
    p.write_text(_ROUTER_FIXTURE)

    out = load_router_for_voice(p)

    # Voice-safe sections kept.
    assert "## Persona" in out
    assert "customer support agent" in out
    assert "## Proactivity clause" in out
    assert "## Anti-spam guardrails" in out
    assert "## Pitfalls" in out

    # Tool-meta sections dropped entirely.
    assert "## Procedure" not in out
    assert "## When to Use" not in out
    assert "## Verification" not in out

    # Frontmatter and intro paragraph dropped.
    assert "---" not in out
    assert "Intro paragraph" not in out
    assert "name: v2v-router" not in out


def test_load_router_for_voice_softens_cronjob_references(tmp_path: Path) -> None:
    """The proactivity clause survives, but its 'schedule via cronjob tool'
    phrasing must be rewritten so the model doesn't emit tool-call markup."""
    p = tmp_path / "router" / "SKILL.md"
    p.parent.mkdir()
    p.write_text(_ROUTER_FIXTURE)

    out = load_router_for_voice(p)

    # The proactivity clause text is preserved, but the tool-call pointer
    # is rewritten away from "cronjob tool".
    assert "schedule it via the `cronjob` tool" not in out
    assert "Before creating a `cronjob`" not in out
    # Some softened replacement is present.
    assert "follow-up" in out.lower() or "memory" in out.lower()


def test_load_router_for_voice_returns_empty_for_missing_file(tmp_path: Path) -> None:
    out = load_router_for_voice(tmp_path / "missing" / "SKILL.md")
    assert out == ""


def test_load_router_for_voice_handles_no_frontmatter(tmp_path: Path) -> None:
    p = tmp_path / "SKILL.md"
    p.write_text("# Body Only\n\n## Persona\n\nBe nice.\n")
    out = load_router_for_voice(p)
    assert "## Persona" in out
    assert "Be nice." in out
