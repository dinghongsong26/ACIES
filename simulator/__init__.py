"""Physical-layer simulators that execute synthesized GR(1) FSMs (warehouse, patrol, delivery, dock)."""

import os
import warnings

# Silence pygame's import-time "Hello from the pygame community" banner and its
# pkg_resources deprecation warning; both must be set before pygame is first imported.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
