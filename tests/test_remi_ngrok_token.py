"""``--ngrok``: a connection that fails prints why, but not the authtoken
(nor the basic-auth password of a ``token:user:password`` value), which
went to the console and any log that captures it (security.md, "Secrets in
logs"). ``modules/ngrok.py`` is upstream's (upstream-absorption.md §1.2)."""

import importlib.util
import sys
import types
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "modules" / "ngrok.py"


def _module(monkeypatch, error: Exception):
    def connect(address, **options):
        raise error

    monkeypatch.setitem(sys.modules, "ngrok", types.SimpleNamespace(connect=connect))
    spec = importlib.util.spec_from_file_location("remi_test_ngrok", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_failed_connection_prints_no_token(monkeypatch, capsys):
    token = "2abcDEFghiJKLmnoPQRstu_secretpart"
    module = _module(monkeypatch, RuntimeError(f"authentication failed: {token} is not valid; basic auth ann:hunter22 / hunter22"))
    module.connect(f"{token}:ann:hunter22", 7860, {})
    out = capsys.readouterr().out
    assert "ngrok connection aborted" in out
    assert token not in out and "secretpart" not in out and "hunter22" not in out
    assert "Your token: ...part," in out  # its end only, to tell which token it was (ASCII: any console prints it)


def test_no_token_still_says_so(monkeypatch, capsys):
    module = _module(monkeypatch, RuntimeError("no authtoken"))
    module.connect(None, 7860, {})
    out = capsys.readouterr().out
    assert "ngrok connection aborted" in out and "Your token: None" in out


def test_a_short_token_is_not_shown(monkeypatch, capsys):
    module = _module(monkeypatch, RuntimeError("bad token"))
    module.connect("abcd1234", 7860, {})
    out = capsys.readouterr().out
    assert "abcd1234" not in out and "1234" not in out


def test_secrets_from_the_options_and_the_environment_stay_out(monkeypatch, capsys):
    """``--ngrok-options '{"basic_auth": [...]}'`` (the help's own example) and the token
    ngrok reads from ``NGROK_AUTHTOKEN`` (``authtoken_from_env``) are secrets too."""
    monkeypatch.setenv("NGROK_AUTHTOKEN", "ENVTOKENabcdefghijkl_env")
    module = _module(monkeypatch, RuntimeError("basic_auth user:optionsSecret99 rejected; authtoken ENVTOKENabcdefghijkl_env"))
    module.connect(None, 7860, {"authtoken_from_env": True, "basic_auth": ["user:optionsSecret99"]})
    out = capsys.readouterr().out
    assert "optionsSecret99" not in out and "ENVTOKENabcdefghijkl_env" not in out


def test_every_secret_named_option_stays_out(monkeypatch, capsys):
    """ngrok takes more secrets than basic auth: ``verify_webhook_secret``, ``oidc_client_secret``
    and the like. A value under a name that says it is a secret is masked, whatever ngrok calls it."""
    module = _module(monkeypatch, RuntimeError("webhook whsecVALUE_abc123 and oidc OIDCsecretValue42 refused for example.ngrok.app"))
    module.connect(None, 7860, {"verify_webhook_secret": "whsecVALUE_abc123", "oidc_client_secret": "OIDCsecretValue42", "domain": "example.ngrok.app"})
    out = capsys.readouterr().out
    assert "whsecVALUE_abc123" not in out and "OIDCsecretValue42" not in out
    assert "example.ngrok.app" in out  # a name that is not a secret still helps to tell what failed
