"""pytest's arguments kept away from the WebUI's argv parser.

``networks`` scans the LoRA folder when imported: an empty one keeps the
tests from reading (and hashing) the LoRAs of the checkout they run in."""

import sys
import tempfile

sys.argv = [*sys.argv[:1], "--lora-dir", tempfile.mkdtemp(prefix="remi-lora-test-")]
