"""V2V outbound room-invite tool (Milestone 9).

The v2v_outbound skill calls this tool when a scheduled follow-up
fires. The tool:

1. Mints a short-lived LiveKit access token granting the user
   ``room_join`` + publish + subscribe on a fresh room named
   ``v2v-<user_id>-<call_id>``.
2. Optionally emails the join URL to ``notify_email`` if SMTP is
   configured. Otherwise falls back to appending the URL into a
   per-user log file so the demo operator can copy-paste it.

The tool intentionally does NOT create the room on the LiveKit server
side — the first participant join (the agent or the user) creates it
implicitly. Keeping the surface to "mint a JWT, deliver a link" means
the tool runs offline in tests and during dry-runs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _data_root() -> Path:
    """Return the directory under which per-user state lives.

    Tests set ``V2V_DATA_ROOT``; production falls back to
    ``~/.hermes/v2v``.
    """
    root = os.environ.get("V2V_DATA_ROOT")
    if root:
        return Path(root)
    return Path.home() / ".hermes" / "v2v"


def _livekit_creds() -> tuple[str, str, str]:
    url = os.environ.get("LIVEKIT_URL", "")
    key = os.environ.get("LIVEKIT_API_KEY", "")
    secret = os.environ.get("LIVEKIT_API_SECRET", "")
    return url, key, secret


# ---------------------------------------------------------------------------
# JWT minting
# ---------------------------------------------------------------------------


def _mint_token(
    *,
    api_key: str,
    api_secret: str,
    identity: str,
    room: str,
    ttl_minutes: int,
) -> str:
    """Build a LiveKit access-token JWT.

    Uses the official ``livekit-api`` SDK when available; falls back to
    a hand-rolled HMAC-SHA256 JWT (sufficient for tests / environments
    without the SDK installed).
    """
    try:
        from livekit.api import AccessToken, VideoGrants
    except Exception:  # pragma: no cover - exercised only when the SDK is missing
        return _mint_token_manual(
            api_key=api_key,
            api_secret=api_secret,
            identity=identity,
            room=room,
            ttl_minutes=ttl_minutes,
        )

    grants = VideoGrants(
        room=room,
        room_join=True,
        can_publish=True,
        can_subscribe=True,
    )
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(grants)
        .with_ttl(timedelta(minutes=ttl_minutes))
    )
    return token.to_jwt()


def _mint_token_manual(
    *,
    api_key: str,
    api_secret: str,
    identity: str,
    room: str,
    ttl_minutes: int,
) -> str:
    """Hand-rolled JWT — kept as a fallback for SDK-less environments."""
    import base64
    import hashlib
    import hmac
    import json

    now = int(datetime.now(tz=timezone.utc).timestamp())
    payload = {
        "iss": api_key,
        "sub": identity,
        "iat": now,
        "exp": now + ttl_minutes * 60,
        "nbf": now,
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
        },
    }
    header = {"alg": "HS256", "typ": "JWT"}

    def _b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    h = _b64(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{h}.{p}".encode("ascii")
    sig = hmac.new(api_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{h}.{p}.{_b64(sig)}"


# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------


async def _send_email(
    *,
    to: str,
    subject: str,
    body: str,
    host: str,
    port: int,
    user: str,
    password: str,
    sender: str,
) -> bool:
    """Best-effort SMTP send. Returns True on success, False on any failure.

    Tries ``aiosmtplib`` first (truly async), falls back to ``smtplib``
    in a worker thread. Tests monkeypatch this whole function.
    """
    try:  # pragma: no cover - aiosmtplib path exercised by integration tests
        import aiosmtplib

        msg = (
            f"From: {sender}\r\nTo: {to}\r\nSubject: {subject}\r\n\r\n{body}"
        ).encode("utf-8")
        await aiosmtplib.send(
            msg,
            hostname=host,
            port=port,
            username=user,
            password=password,
            sender=sender,
            recipients=[to],
            start_tls=True,
        )
        return True
    except Exception as e:
        logger.warning("aiosmtplib send failed (%s); falling back to smtplib", e)

    try:
        def _send_sync():
            import smtplib
            from email.mime.text import MIMEText

            mime = MIMEText(body)
            mime["From"] = sender
            mime["To"] = to
            mime["Subject"] = subject
            with smtplib.SMTP(host, port, timeout=15) as s:
                s.starttls()
                s.login(user, password)
                s.sendmail(sender, [to], mime.as_string())

        await asyncio.to_thread(_send_sync)
        return True
    except Exception:
        logger.exception("smtplib fallback failed")
        return False


def _append_log(user_id: str, line: str) -> Path:
    """Append a line to ``<data_root>/users/<user_id>/outbound_log.md``."""
    user_dir = _data_root() / "users" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    log = user_dir / "outbound_log.md"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")
    return log


def _build_join_url(livekit_url: str, token: str) -> str:
    """Build a copy-paste-able join URL.

    LiveKit JS clients accept ``https://meet.livekit.io/?token=<jwt>``-
    style links. Since the LiveKit URL itself is the websocket endpoint
    we cannot redirect end-users to (it's ``wss://`` for the SDK), we
    emit ``<LIVEKIT_URL>?access_token=<jwt>`` so the consumer can either
    open it in a LiveKit web client wired to that host, or extract the
    token+url pair from the structured return value.
    """
    sep = "&" if "?" in livekit_url else "?"
    return f"{livekit_url}{sep}access_token={quote(token, safe='')}"


# ---------------------------------------------------------------------------
# Public tool surface
# ---------------------------------------------------------------------------


async def v2v_room_invite(
    user_id: str,
    reason: str,
    notify_email: Optional[str] = None,
    ttl_minutes: int = 30,
) -> dict:
    """Create a LiveKit room JWT for an outbound v2v call and (optionally)
    deliver the join URL to the user.

    Returns ``{"room", "url", "token", "expires_at", "notified"}``.

    For the demo: if SMTP_* env vars are absent, the function falls back
    to writing the join URL to ``data/users/<user_id>/outbound_log.md``
    so the test harness / human operator can copy-paste it.
    """
    if not user_id:
        raise ValueError("user_id is required")

    livekit_url, api_key, api_secret = _livekit_creds()
    if not (livekit_url and api_key and api_secret):
        raise RuntimeError(
            "LiveKit creds missing. Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET."
        )

    call_id = secrets.token_hex(4)  # 8 hex chars
    room = f"v2v-{user_id}-{call_id}"
    identity = f"agent_invite:{user_id}:{call_id}"

    expires_at = (
        datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
    ).isoformat()

    token = _mint_token(
        api_key=api_key,
        api_secret=api_secret,
        identity=identity,
        room=room,
        ttl_minutes=ttl_minutes,
    )

    url = _build_join_url(livekit_url, token)

    notified = await _deliver(
        user_id=user_id,
        reason=reason,
        notify_email=notify_email,
        url=url,
    )

    return {
        "room": room,
        "url": url,
        "token": token,
        "expires_at": expires_at,
        "notified": notified,
    }


async def _deliver(
    *,
    user_id: str,
    reason: str,
    notify_email: Optional[str],
    url: str,
) -> bool:
    """Email the link if SMTP+recipient are configured; else log; else return False."""
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_port = int(os.environ.get("SMTP_PORT", "587") or 587)
    smtp_from = os.environ.get("SMTP_FROM") or smtp_user or ""

    have_smtp = bool(smtp_host and smtp_user and smtp_pass)

    if notify_email and have_smtp:
        ok = await _send_email(
            to=notify_email,
            subject=f"Quick voice check-in: {reason}"[:120],
            body=(
                f"Hi — your assistant would like to talk briefly about: {reason}.\n"
                f"Tap to join: {url}\n\n(This link expires soon.)\n"
            ),
            host=smtp_host or "",
            port=smtp_port,
            user=smtp_user or "",
            password=smtp_pass or "",
            sender=smtp_from,
        )
        return bool(ok)

    if notify_email:
        # SMTP missing: log fallback so the demo operator can copy-paste.
        ts = datetime.now(timezone.utc).isoformat()
        line = f"- {ts} reason={reason!r} email={notify_email!r} url={url}"
        try:
            _append_log(user_id, line)
            return True
        except Exception:
            logger.exception("Failed to write outbound log fallback")
            return False

    # No email recipient and no SMTP — nothing to deliver.
    return False


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


V2V_ROOM_INVITE_SCHEMA = {
    "name": "v2v_room_invite",
    "description": (
        "Create a LiveKit room for an outbound v2v call and (optionally) email "
        "the join URL to the user. Returns the room name, JWT token, join URL, "
        "expiry timestamp, and a notified boolean. If SMTP is not configured, "
        "falls back to writing the URL into the user's outbound_log.md."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Stable per-user identifier — used in the room name and log path.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "One-line, voice-ready reason for the follow-up. Embedded in the "
                    "email body the user sees (e.g. 'Quick check on your refund')."
                ),
            },
            "notify_email": {
                "type": "string",
                "description": "Optional email address to receive the join link.",
            },
            "ttl_minutes": {
                "type": "integer",
                "description": "Token lifetime in minutes. Default 30.",
            },
        },
        "required": ["user_id", "reason"],
    },
}


def _check_v2v_room_invite_requirements() -> bool:
    """Available whenever LiveKit env vars are set. We don't fail closed
    here: the runtime ``v2v_room_invite`` call will raise a clear error
    if creds are missing — that's nicer than the tool silently disappearing
    from the toolset.
    """
    return True


def _handler(args: dict, **kwargs) -> dict:
    """Sync wrapper used by the registry — runs the async tool to completion."""
    coro = v2v_room_invite(
        user_id=args.get("user_id", ""),
        reason=args.get("reason", ""),
        notify_email=args.get("notify_email"),
        ttl_minutes=int(args.get("ttl_minutes", 30) or 30),
    )
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Already inside an event loop (e.g. agent runtime). Fall back to
        # creating a dedicated loop in a thread so we don't deadlock.
        import threading

        result: dict = {}
        exc: dict = {}

        def _run():
            try:
                result["v"] = asyncio.run(coro)
            except Exception as e:
                exc["e"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join()
        if "e" in exc:
            raise exc["e"]
        return result.get("v", {})


try:
    from tools.registry import registry

    registry.register(
        name="v2v_room_invite",
        toolset="v2v",
        schema=V2V_ROOM_INVITE_SCHEMA,
        handler=_handler,
        check_fn=_check_v2v_room_invite_requirements,
        emoji="📞",
    )
except Exception:  # pragma: no cover - registry is best-effort at import
    logger.exception("Failed to register v2v_room_invite")
