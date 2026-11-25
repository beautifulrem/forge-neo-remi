"""The source offer of AGPL-3.0 section 13 (docs/remi-webui/licensing.md): every
user of the server, the sign-in screen included, gets a link to the source of
the version it runs. ``--source-url`` / ``REMI_SOURCE_URL`` name the source of
a modified copy."""

import logging
import os
import subprocess

import pytest

from remi.server.auth import Auth
from remi.server.source import DEFAULT_SOURCE_URL, SOURCE_URL_ENV, SourceOffer, configured_source_url, resolve_source
from remi.server.testing import server_app
from remi.testing import webui_state

BRANCH_LINK = "https://github.com/beautifulrem/forge-neo-remi/tree/remi"


def _git(*args, cwd):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
    )


@pytest.fixture
def checkout(tmp_path):
    """A checkout with one commit; no remote knows it yet."""

    repo = tmp_path / "checkout"
    repo.mkdir()
    _git("init", "-q", "-b", "remi", cwd=repo)
    _git("-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", "init", cwd=repo)
    return repo, _git("rev-parse", "HEAD", cwd=repo).stdout.strip()


# region Resolving the link


def test_the_default_is_the_public_repository_and_its_branch(tmp_path):
    assert DEFAULT_SOURCE_URL == "https://github.com/beautifulrem/forge-neo-remi"
    # not a checkout (a Docker image has no .git): the commit is unknown
    assert resolve_source(None, root=tmp_path) == SourceOffer(public=BRANCH_LINK, running=BRANCH_LINK)


def test_a_commit_the_repository_has_is_linked_to_its_tree(checkout):
    repo, head = checkout
    _git("remote", "add", "origin", "git@github.com:beautifulrem/forge-neo-remi.git", cwd=repo)
    _git("update-ref", "refs/remotes/origin/remi", head, cwd=repo)
    offer = resolve_source(None, root=repo)
    assert offer.running == f"{DEFAULT_SOURCE_URL}/tree/{head}"
    # before a login only the branch: the running commit is not told to anyone who reaches the server
    assert offer.public == BRANCH_LINK


def test_a_commit_the_repository_does_not_have_gets_the_branch(checkout):
    repo, head = checkout
    # a local commit: no remote-tracking ref of the repository contains it (the link would be a 404)
    _git("remote", "add", "origin", "https://github.com/beautifulrem/forge-neo-remi.git", cwd=repo)
    assert resolve_source(None, root=repo).running == BRANCH_LINK
    # published, but to another repository (a fork that did not set --source-url)
    _git("remote", "add", "fork", "https://github.com/someone/fork.git", cwd=repo)
    _git("update-ref", "refs/remotes/fork/main", head, cwd=repo)
    assert resolve_source(None, root=repo).running == BRANCH_LINK


def test_the_operator_of_a_modified_copy_names_its_source(checkout):
    repo, head = checkout
    _git("remote", "add", "origin", "https://github.com/beautifulrem/forge-neo-remi.git", cwd=repo)
    _git("update-ref", "refs/remotes/origin/remi", head, cwd=repo)
    url = "https://git.example.com/me/remi/src/branch/main"
    # used as given: Remi cannot tell where the operator publishes which version
    assert resolve_source(url, root=repo) == SourceOffer(public=url, running=url)


def test_only_a_web_address_names_the_source(tmp_path, caplog):
    for value in ("file:///srv/remi", "javascript:alert(1)", "git@github.com:me/remi.git", "https://", "remi"):
        with caplog.at_level(logging.WARNING, logger="remi.server.source"):
            caplog.clear()
            assert resolve_source(value, root=tmp_path) == SourceOffer(public=BRANCH_LINK, running=BRANCH_LINK), value
        assert "--source-url" in caplog.text, value


def test_the_flag_wins_over_the_environment_variable():
    from modules import cmd_args

    flagged, _ = cmd_args.parser.parse_known_args(["--source-url", "https://example.org/remi"])
    plain, _ = cmd_args.parser.parse_known_args([])
    assert SOURCE_URL_ENV == "REMI_SOURCE_URL"
    assert configured_source_url(flagged, {SOURCE_URL_ENV: "https://example.net/r"}) == "https://example.org/remi"
    assert configured_source_url(plain, {SOURCE_URL_ENV: " https://example.net/r "}) == "https://example.net/r"
    assert configured_source_url(plain, {SOURCE_URL_ENV: ""}) is None
    assert configured_source_url(plain, {}) is None


# endregion

# region The server's answers


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


OFFER = SourceOffer(public="https://example.org/remi", running="https://example.org/remi/tree/0123abc")


def test_the_sign_in_screen_gets_the_source_without_the_running_commit(webui, tmp_path):
    with server_app(tmp_path, auth=Auth({"ann": "s3cret"}, b"k" * 32)) as server:
        server.app.state.source = OFFER
        client = server.client
        # a public path: the sign-in screen links it
        assert client.get("/api/v2/auth/status").json() == {"required": True, "user": None, "source_url": OFFER.public}
        client.post("/api/v2/auth/login", json={"username": "ann", "password": "s3cret"})
        assert client.get("/api/v2/auth/status").json()["source_url"] == OFFER.running
        assert client.get("/api/v2/app").json()["auth"]["source_url"] == OFFER.running


def test_without_a_login_every_user_gets_the_running_version(webui, tmp_path):
    with server_app(tmp_path) as server:
        server.app.state.source = OFFER
        assert server.client.get("/api/v2/auth/status").json()["source_url"] == OFFER.running
        assert server.client.get("/api/v2/app").json()["auth"]["source_url"] == OFFER.running


def test_the_app_resolves_its_source_when_it_is_built(webui, tmp_path):
    with server_app(tmp_path) as server:
        assert server.app.state.source.running.startswith(DEFAULT_SOURCE_URL + "/tree/")


# endregion
