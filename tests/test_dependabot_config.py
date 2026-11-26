"""Dependabot proposes updates only for what Remi owns (.github/dependabot.yml).

The SPA in webui/ is Remi's own. Of the Python requirements, the model runtime is shared with
Forge Neo and its pins move with an upstream sync (security.md, "Dependency advisories"); only
the server packages Remi adds to requirements.txt or pins above Forge Neo's are proposed here,
with the workflow's actions, which check.yml pins to commit SHAs.
"""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# requirements.txt pins Remi adds to upstream's, or raises (fastapi, from upstream's 0.127.1); a sync
# adds a new one here (upstream-absorption.md, `requirements*.txt`)
REMI_PINS = {"fastapi", "python-multipart", "pytz", "requests", "starlette", "uvicorn", "websockets"}


def _config() -> dict:
    return yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))


def _updates(ecosystem: str) -> list[dict]:
    return [entry for entry in _config()["updates"] if entry["package-ecosystem"] == ecosystem]


def test_the_spa_is_updated_in_one_weekly_group():
    (spa,) = _updates("npm")
    assert spa["directory"] == "/webui"
    assert spa["schedule"]["interval"] == "weekly"
    assert any(group.get("patterns") == ["*"] for group in spa["groups"].values())


def test_redoc_is_updated_within_the_viewer_the_redoc_page_loads():
    # The /redoc page is written for ReDoc 3 (remi/server/api_docs.py: an ES module, telemetry off
    # by an attribute of its tag) and webui/vite.config.ts replaces what that bundle would load from
    # outside hosts, failing the build when an update names such a host again. Updates within 3
    # come with the weekly group; another major version is another viewer, a port of the page.
    (spa,) = _updates("npm")
    assert {"dependency-name": "redoc", "versions": [">=4.0.0"]} in spa["ignore"]
    package = json.loads((ROOT / "webui" / "package.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"3\.\d+\.\d+", package["devDependencies"]["redoc"])


def test_only_remis_own_python_pins_are_updated():
    (python,) = _updates("pip")
    assert python["directory"] == "/"
    allowed = {entry["dependency-name"] for entry in python["allow"]}
    assert allowed == REMI_PINS
    pinned = {m[1].lower() for m in re.finditer(r"^([A-Za-z0-9_.-]+)==", (ROOT / "requirements.txt").read_text(encoding="utf-8"), re.M)}
    assert REMI_PINS.issubset(pinned)


def test_fastapi_stays_below_the_release_that_hides_included_routes():
    # FastAPI 0.137 stopped listing included routes in app.routes, which extensions walk
    (python,) = _updates("pip")
    assert {"dependency-name": "fastapi", "versions": [">=0.137.0"]} in python["ignore"]


def test_the_workflows_pinned_actions_are_kept_current():
    # check.yml pins its actions to commit SHAs, which only these updates move
    (actions,) = _updates("github-actions")
    assert actions["directory"] == "/"
    workflow = (ROOT / ".github" / "workflows" / "check.yml").read_text(encoding="utf-8")
    uses = re.findall(r"uses: (\S+)", workflow)
    assert uses and all(re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", use) for use in uses), uses
