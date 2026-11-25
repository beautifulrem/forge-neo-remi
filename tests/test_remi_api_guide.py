"""The API guide (docs/remi-webui/api.md) lists every event the WebSocket sends:
a client written from it learns of each one (the pipeline events were missing)."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "remi-webui" / "api.md"


def sent_events() -> set[str]:
    """Event types the server publishes on the bus or writes on the socket itself."""
    found: set[str] = set()
    for path in (ROOT / "remi").rglob("*.py"):
        found.update(re.findall(r"\.publish\(\s*\"([a-z]+(?:\.[a-z_]+)+)\"", path.read_text(encoding="utf-8")))
    socket = (ROOT / "remi" / "server" / "ws.py").read_text(encoding="utf-8")
    found.update(re.findall(r"(?:_send|push_event)\(\{\"type\": \"([a-z_.]+)\"", socket))
    return found


def documented_events() -> set[str]:
    text = GUIDE.read_text(encoding="utf-8")
    table = text[text.index("**Text frames**") :]
    table = table[: table.index("\n\n", table.index("| type |"))]
    return set(re.findall(r"^\| `([a-z_.]+)` \|", table, flags=re.M))


def test_the_guide_lists_every_websocket_event():
    sent = sent_events()
    assert {"hello", "queue.changed", "job.progress", "resync"} <= sent  # the scan finds the known ones
    assert sorted(sent - documented_events()) == []


def test_the_guide_names_the_headers_that_mark_a_proxied_request():
    """Only these headers take the local-client exemption away (X-Forwarded-Proto and -Port do
    not): the guide names them one by one instead of a wildcard."""
    from remi.server.deps import PROXY_HEADERS

    text = GUIDE.read_text(encoding="utf-8")
    assert "X-Forwarded-*" not in text
    for name in PROXY_HEADERS:
        assert f"`{name}`" in text.lower(), name
