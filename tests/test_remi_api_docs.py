"""The API documentation pages (/docs, /redoc): their viewers come from the
SPA build, never from a CDN, and the pages carry the server's CSP
(remi/server/api_docs.py, docs/remi-webui/security.md "Outbound network
calls")."""

import gzip
import json
import os
import re

import pytest
from fastapi.testclient import TestClient

from remi.server.api_docs import MANIFEST, viewer_assets
from remi.server.spa import DIST
from remi.server.testing import BASE_URL, server_app
from remi.testing import webui_state

SWAGGER = "assets/swagger-ui-dist-5.0.0"
REDOC = "assets/redoc-3.0.0"
OUTSIDE_HOSTS = ("cdn.jsdelivr.net", "fonts.googleapis.com", "fastapi.tiangolo.com", "unpkg.com", "validator.swagger.io", "cdn.redoc.ly", "redocly.com")


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


def make_dist(root, manifest=None):
    dist = root / "dist"
    for folder in (SWAGGER, REDOC):
        (dist / folder).mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><html><head></head><body></body></html>")
    (dist / SWAGGER / "swagger-ui-bundle.js").write_text("window.SwaggerUIBundle = () => {}")
    (dist / SWAGGER / "swagger-ui.css").write_text("body{}")
    (dist / REDOC / "redoc.standalone.js").write_text("// redoc")
    manifest = manifest or {"swaggerJs": f"{SWAGGER}/swagger-ui-bundle.js", "swaggerCss": f"{SWAGGER}/swagger-ui.css", "redocJs": f"{REDOC}/redoc.standalone.js"}
    (dist / MANIFEST).write_text(json.dumps(manifest))
    return dist


def urls(page):
    return re.findall(r'(?:src|href)="([^"]*)"', page)


def assert_locked_down(response):
    policy = response.headers["content-security-policy"]
    directives = dict(item.strip().split(" ", 1) for item in policy.split(";"))
    assert directives["default-src"] == "'self'" and directives["object-src"] == "'none'"
    # scripts only from the server, the page's inline script by its hash
    assert re.fullmatch(r"'self'( 'sha256-[A-Za-z0-9+/=]+')*", directives["script-src"])
    assert directives["connect-src"].split()[0] == "'self'"
    assert not any(host in response.text for host in OUTSIDE_HOSTS)


def test_swagger_ui_loads_the_builds_copy_under_the_csp(webui, tmp_path):
    with server_app(tmp_path / "data", spa_dir=make_dist(tmp_path)) as server:
        response = server.client.get("/docs")
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert_locked_down(response)
        assert set(urls(response.text)) == {f"/{SWAGGER}/swagger-ui-bundle.js", f"/{SWAGGER}/swagger-ui.css", "/favicon.svg"}
        assert "url: '/openapi.json'" in response.text and '"validatorUrl": null' in response.text
        assert "'sha256-" in response.headers["content-security-policy"]  # the init script
        # the viewer's files are build assets, served by the server itself
        assert server.client.get(f"/{SWAGGER}/swagger-ui-bundle.js").text == "window.SwaggerUIBundle = () => {}"
        redirect = server.client.get("/docs/oauth2-redirect")
        assert redirect.status_code == 200 and "'sha256-" in redirect.headers["content-security-policy"]
        assert_locked_down(redirect)


def test_redoc_loads_the_builds_copy_as_a_module_with_telemetry_off(webui, tmp_path):
    with server_app(tmp_path / "data", spa_dir=make_dist(tmp_path)) as server:
        response = server.client.get("/redoc")
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert_locked_down(response)
        assert set(urls(response.text)) == {f"/{REDOC}/redoc.standalone.js", "/favicon.svg", "/openapi.json"}
        # ReDoc 3's bundle is an ES module (a classic script tag fails on its `export`), and it
        # reads its options from the tag: without this one it reports usage to Redocly
        assert f'<script type="module" src="/{REDOC}/redoc.standalone.js"></script>' in response.text
        assert '<redoc spec-url="/openapi.json" disable-telemetry="true"></redoc>' in response.text
        # no inline script: nothing for the CSP to allow by hash
        assert response.text.count("<script") == 1 and "'sha256-" not in response.headers["content-security-policy"]
        # a browser runs a module only when it is served as JavaScript
        bundle = server.client.get(f"/{REDOC}/redoc.standalone.js")
        assert bundle.status_code == 200 and bundle.headers["content-type"].startswith("text/javascript")


def test_the_redoc_page_escapes_what_it_writes_into_the_markup(webui, tmp_path):
    with server_app(tmp_path / "data", spa_dir=make_dist(tmp_path)) as server:
        client = TestClient(server.app, base_url=BASE_URL, root_path='/a"><script>alert(1)</script>')
        response = client.get('/a"><script>alert(1)</script>/redoc')
        assert response.status_code == 200 and "&lt;script&gt;alert(1)" in response.text
        assert "<script>alert(1)" not in response.text and response.text.count("<script") == 1


def test_the_pages_follow_the_subpath(webui, tmp_path):
    with server_app(tmp_path / "data", spa_dir=make_dist(tmp_path)) as server:
        client = TestClient(server.app, base_url=BASE_URL, root_path="/remi")
        docs = client.get("/remi/docs").text
        assert f'src="/remi/{SWAGGER}/swagger-ui-bundle.js"' in docs and "url: '/remi/openapi.json'" in docs
        redoc = client.get("/remi/redoc").text
        assert 'spec-url="/remi/openapi.json"' in redoc and f'src="/remi/{REDOC}/redoc.standalone.js"' in redoc


@pytest.mark.parametrize(
    "manifest",
    [
        None,  # no build
        {"swaggerJs": "assets/x/missing.js", "swaggerCss": f"{SWAGGER}/swagger-ui.css", "redocJs": f"{REDOC}/redoc.standalone.js"},
        {"swaggerJs": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js", "swaggerCss": f"{SWAGGER}/swagger-ui.css", "redocJs": f"{REDOC}/redoc.standalone.js"},
        {"swaggerJs": "assets/../../secret.js", "swaggerCss": f"{SWAGGER}/swagger-ui.css", "redocJs": f"{REDOC}/redoc.standalone.js"},
        {"swaggerJs": f"{SWAGGER}/swagger-ui-bundle.js"},
    ],
)
def test_without_a_complete_build_the_pages_say_how_to_build_it(webui, tmp_path, manifest):
    dist = make_dist(tmp_path, manifest)
    if manifest is None:
        (dist / MANIFEST).unlink()
    assert viewer_assets(dist) is None
    with server_app(tmp_path / "data", spa_dir=dist) as server:
        for path in ("/docs", "/redoc"):
            response = server.client.get(path)
            assert response.status_code == 503 and "pnpm build" in response.text
            assert urls(response.text) == ["/openapi.json"]
            assert_locked_down(response)


def test_without_the_spa_the_server_still_serves_the_viewer_files(webui, tmp_path):
    """``--nowebui``: no SPA to serve ``/assets``, so the docs serve their
    viewers' folders themselves (and nothing else of the build)."""

    import threading

    from remi.jobs import run_inline
    from remi.server import Services, create_app
    from remi.server.auth import Auth
    from remi.server.testing import FakeState, StubExecutor

    dist = make_dist(tmp_path)
    (dist / "assets" / "app-1a2b.js").write_text("app")
    (dist / SWAGGER / "swagger-ui-bundle.js.gz").write_bytes(gzip.compress(b"packed"))
    state = FakeState()
    services = Services.create(tmp_path / "data", realtime=False, housekeeping=False, executor=StubExecutor(state), dispatch=run_inline, lock=threading.Lock(), state=lambda: state)
    try:
        app = create_app(services=services, auth=Auth(), sdapi=False, webui=False, spa_dir=dist)
        with TestClient(app, base_url=BASE_URL) as client:
            assert client.get("/docs").status_code == 200 and client.get("/redoc").status_code == 200
            bundle = client.get(f"/{SWAGGER}/swagger-ui-bundle.js", headers={"Accept-Encoding": "identity"})
            assert bundle.status_code == 200 and bundle.text == "window.SwaggerUIBundle = () => {}" and "immutable" in bundle.headers["cache-control"]
            packed = client.get(f"/{SWAGGER}/swagger-ui-bundle.js", headers={"Accept-Encoding": "gzip"})
            assert packed.headers.get("content-encoding") == "gzip" and packed.text == "packed"
            module = client.get(f"/{REDOC}/redoc.standalone.js")
            assert module.text == "// redoc" and module.headers["content-type"].startswith("text/javascript")
            assert client.get("/assets/app-1a2b.js").status_code == 404
            assert client.get(f"/{SWAGGER}/missing.js").status_code == 404
            assert client.get("/assets/../index.html").status_code == 404
    finally:
        services.close()


def test_the_openapi_description_stays_and_lists_no_docs_pages(webui, tmp_path):
    with server_app(tmp_path / "data") as server:
        document = server.client.get("/openapi.json").json()
        assert not {"/docs", "/redoc", "/docs/oauth2-redirect"} & set(document["paths"])


@pytest.mark.spa_build
@pytest.mark.skipif(not (DIST / MANIFEST).is_file() and not os.environ.get("REMI_SPA_BUILT"), reason="the SPA is not built (cd webui && pnpm build)")
def test_the_build_vendors_both_viewers():
    """``apiDocs`` in webui/vite.config.ts: pinned versions, the license
    files next to them, no references to source maps that are not shipped,
    and no outside host left for a viewer to load from or report to.

    Reads the real build: tools/check.sh runs it after its own build (step
    ``api-docs``, which sets REMI_SPA_BUILT: then a missing build fails), never
    beside it."""

    assets = viewer_assets(DIST)
    assert assets is not None
    for key, name in (("swaggerJs", "swagger-ui-dist"), ("swaggerCss", "swagger-ui-dist"), ("redocJs", "redoc")):
        file = DIST / assets[key]
        assert file.parent.name.startswith(f"{name}-") and (file.parent / "LICENSE").is_file()
        source = file.read_text(encoding="utf8")
        assert "sourceMappingURL" not in source[-300:]
        # redocly.com itself stays: the attribution's link and the links to the OpenAPI reference
        remote = re.search(r"https://(?:[a-z0-9.-]+\.redoc\.ly|[a-z0-9.-]+\.redocly\.com|cdn\.jsdelivr\.net|unpkg\.com|fonts\.(?:googleapis|gstatic)\.com)\b[^`'\"\s]*", source)
        assert remote is None, f"{assets[key]} names {remote[0]}"
    redoc = DIST / assets["redocJs"]
    source = redoc.read_text(encoding="utf8")
    # the bundle the /redoc page loads as a module, with the option that page sets
    assert re.search(r"export\s*\{[^}]*\}\s*;?\s*$", source) and "disableTelemetry" in source
    # ReDoc 3 lists what its bundle includes nowhere: the build writes the license comments the bundle keeps
    notices = redoc.with_name(redoc.name + ".LICENSE.txt").read_text(encoding="utf8")
    assert "Redocly" in notices and "React" in notices


def test_the_not_built_pages_ask_for_the_node_version_the_launcher_names():
    # the launcher's NODE_REQUIREMENT (also the README's); npm would ignore pnpm-lock.yaml, so
    # neither page offers it
    from modules import launch_utils
    from remi.server import api_docs, spa

    for page in (spa.NOT_BUILT, api_docs.NOT_BUILT):
        assert launch_utils.NODE_REQUIREMENT in page
        assert "npm" not in page.replace("pnpm", "")
        # the launcher's own commands: the lockfile as committed, never a re-resolved one
        assert "pnpm install --frozen-lockfile\n" in page and "pnpm install\n" not in page
        # readable on a phone (a server reached over the network)
        assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page
    install, _build = launch_utils.webui_build_commands(which={"node": "/n/node", "pnpm": "/n/pnpm"}.get)
    assert install == ["/n/pnpm", "install", "--frozen-lockfile"]
