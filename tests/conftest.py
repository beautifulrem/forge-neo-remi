import atexit
import os
import shutil
import sys
import tempfile

# The WebUI parses sys.argv when modules.shared is first imported; keep
# pytest's own arguments away from it, and give it a settings file of its
# own: a test that saved options without redirecting shared.config_filename
# once rewrote the owner's config.json (test_settings_file_isolation.py).
_SETTINGS_DIR = tempfile.mkdtemp(prefix="remi-tests-settings-")
atexit.register(shutil.rmtree, _SETTINGS_DIR, True)
sys.argv = sys.argv[:1] + ["--ui-settings-file", os.path.join(_SETTINGS_DIR, "config.json")]
# As modules_forge/initialization.py does at startup (huggingface_guess, k_diffusion, ...)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules_forge", "packages"))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_api_auth_throttle():
    """Failed --api-auth logins are counted process-wide; tests start clean."""
    from remi.server.auth import API_AUTH_THROTTLE

    API_AUTH_THROTTLE._failures.clear()
    yield
