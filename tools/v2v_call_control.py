"""V2V call-control tools — terminate or hand off the LiveKit room.

These handlers can't directly disconnect the LiveKit room (they run inside
``AIAgent`` which has no awareness of the room object). Instead they return
a structured result dict; the gateway adapter watches the tool-event queue
and triggers the actual room hangup on observing a ``tool_complete`` for
``end_call`` or ``agent_handover``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def end_call(reason: str) -> Dict[str, Any]:
    """Politely terminate the call after the conversation is resolved."""
    return {
        "status": "ok",
        "action": "end_call",
        "reason": reason or "",
    }


def agent_handover(reason: str, summary: str) -> Dict[str, Any]:
    """Escalate the call to a human supervisor with a handoff summary."""
    return {
        "status": "ok",
        "action": "handover",
        "reason": reason or "",
        "summary": summary or "",
    }


END_CALL_SCHEMA = {
    "name": "end_call",
    "description": (
        "Politely end the current voice call once the user's issue is "
        "resolved or there is no further action to take. Speak a short "
        "farewell sentence first; the call will hang up a few seconds "
        "later so the farewell finishes synthesizing. Use this only when "
        "the conversation has reached a clean stopping point."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Free-form reason (e.g. 'issue resolved', 'user requested', 'no further action').",
            },
        },
        "required": ["reason"],
    },
}

AGENT_HANDOVER_SCHEMA = {
    "name": "agent_handover",
    "description": (
        "Hand the call off to a human supervisor. Use when the issue is "
        "out of scope, the caller asks for a manager, or repeated tool "
        "calls fail to resolve the situation. Speak a short bridge "
        "sentence first; the call will hang up a few seconds later. "
        "Provide a summary so the human picks up cleanly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Short tag describing why we're handing off (e.g. 'angry_customer', 'out_of_scope').",
            },
            "summary": {
                "type": "string",
                "description": "1-3 sentence summary of the call so the human supervisor has context.",
            },
        },
        "required": ["reason", "summary"],
    },
}


def _check_v2v_call_control_requirements() -> bool:
    return True


from tools.registry import registry

registry.register(
    name="end_call",
    toolset="v2v",
    schema=END_CALL_SCHEMA,
    handler=lambda args, **kw: end_call(args.get("reason", "")),
    check_fn=_check_v2v_call_control_requirements,
    emoji="📴",
)
registry.register(
    name="agent_handover",
    toolset="v2v",
    schema=AGENT_HANDOVER_SCHEMA,
    handler=lambda args, **kw: agent_handover(
        args.get("reason", ""), args.get("summary", "")
    ),
    check_fn=_check_v2v_call_control_requirements,
    emoji="🙋",
)
